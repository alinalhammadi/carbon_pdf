import io
import os
import re
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fitz  # PyMuPDF
import pymupdf4llm  # PyMuPDF4LLM
import requests
from rapidfuzz import process, fuzz


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="CarbonSpec - LLM-first PDF MVP", layout="wide")


# -----------------------------
# Minimal factor library and mapping prompts
# These are hints for the LLM to ground its estimation
# -----------------------------
FACTOR_HINTS = [
    {"category":"Concrete",  "material_key":"concrete_cem1_30mpa", "unit":"m3", "factor_kgco2e_per_unit":330, "notes":"ICE v3.0 generic CEM I"},
    {"category":"Concrete",  "material_key":"concrete_cem2_30mpa", "unit":"m3", "factor_kgco2e_per_unit":260, "notes":"ICE v3.0 generic CEM II"},
    {"category":"Steel",     "material_key":"rebar_virgin",        "unit":"t",  "factor_kgco2e_per_unit":1900, "notes":"ICE v3.0 primary steel"},
    {"category":"Steel",     "material_key":"rebar_recycled",      "unit":"t",  "factor_kgco2e_per_unit":950,  "notes":"ICE v3.0 high recycled content"},
    {"category":"Aluminum",  "material_key":"al_extruded_primary", "unit":"t",  "factor_kgco2e_per_unit":9500, "notes":"ICE v3.0 generic"},
    {"category":"Aluminum",  "material_key":"al_extruded_recycled","unit":"t",  "factor_kgco2e_per_unit":5000, "notes":"ICE v3.0 high recycled content"},
    {"category":"Glass",     "material_key":"float_glass",         "unit":"t",  "factor_kgco2e_per_unit":1200, "notes":"ICE v3.0 generic"},
    {"category":"Plastics",  "material_key":"hdpe",                 "unit":"t",  "factor_kgco2e_per_unit":1900, "notes":"ICE v3.0 generic"},
    {"category":"Gypsum",    "material_key":"gypsum_board_12mm",    "unit":"m2", "factor_kgco2e_per_unit":8,    "notes":"ICE v3.0 approx per m2"},
    {"category":"MEP",       "material_key":"copper_cable",         "unit":"t",  "factor_kgco2e_per_unit":4000, "notes":"ICE v3.0 generic cable mix"},
    {"category":"PV",        "material_key":"pv_module",            "unit":"kwp","factor_kgco2e_per_unit":600,  "notes":"Meta study avg per kWp"}
]

# Regex hints for the LLM prompt
MAPPING_HINTS = [
    {"regex":"concrete.*cem\\s*i|\\bcem\\s*i\\b", "material_key":"concrete_cem1_30mpa"},
    {"regex":"concrete.*cem\\s*ii|\\bcem\\s*ii\\b","material_key":"concrete_cem2_30mpa"},
    {"regex":"\\brebar\\b|\\bsteel\\b",           "material_key":"rebar_virgin"},
    {"regex":"recycled.*steel|rebar.*recycled",   "material_key":"rebar_recycled"},
    {"regex":"aluminum|aluminium",                "material_key":"al_extruded_primary"},
    {"regex":"aluminum.*recycled|aluminium.*recycled","material_key":"al_extruded_recycled"},
    {"regex":"float.*glass|\\bglass\\b",          "material_key":"float_glass"},
    {"regex":"\\bhdpe\\b|polyethylene",           "material_key":"hdpe"},
    {"regex":"gypsum|drywall|plasterboard",       "material_key":"gypsum_board_12mm"},
    {"regex":"copper.*cable|\\bcable\\b",         "material_key":"copper_cable"},
    {"regex":"\\bpv\\b|solar|\\bkwp\\b",          "material_key":"pv_module"},
]


# -----------------------------
# HF token and API helper
# -----------------------------
def hf_token() -> str:
    tok = os.getenv("HF_API_TOKEN")
    if not tok:
        try:
            tok = st.secrets.get("HF_API_TOKEN", None)  # type: ignore[attr-defined]
        except Exception:
            tok = None
    return tok or ""

def hf_generate(model_repo: str, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str | None:
    token = hf_token()
    if not token:
        return None
    url = f"https://api-inference.huggingface.co/models/{model_repo}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code != 200:
            st.warning(f"Hugging Face API returned {resp.status_code}")
            return None
        data = resp.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)
    except Exception as e:
        st.error(f"HF inference failed: {e}")
        return None


# -----------------------------
# PDF extraction with PyMuPDF4LLM
# -----------------------------
def extract_markdown_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    md = pymupdf4llm.to_markdown(doc)
    return md

def parse_items_from_markdown(md: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    rows = []
    qty_rx = re.compile(
        r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>t|ton|tons|tonne|tonnes|mt|kg|m3|m2|kwp|sqm|square meters?)?\b",
        re.I
    )
    for ln in lines:
        if ln.startswith("#") or ln.startswith("|"):
            continue
        m = qty_rx.search(ln)
        if not m:
            continue
        qty = float(m.group("qty"))
        unit = (m.group("unit") or "").strip()
        item_text = re.sub(qty_rx, "", ln).strip(" -:•|")
        if not item_text:
            item_text = ln
        rows.append({"item": item_text[:140], "quantity": qty, "unit": unit})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["item","quantity","unit"])
    return df


# -----------------------------
# Step 1 - LLM JSON estimator
# The model returns machine-readable JSON with per-line estimates and totals
# -----------------------------
def make_json_estimator_prompt(extracted_md: str, items_df: pd.DataFrame) -> str:
    schema = {
        "schema_note": "Return pure JSON only. No prose. Use this exact schema.",
        "total_kgco2e": 0.0,
        "uncertainty_low_kgco2e": 0.0,
        "uncertainty_high_kgco2e": 0.0,
        "lines": [
            {
                "item": "string",
                "quantity": 0.0,
                "input_unit": "string",
                "matched_material_key": "string",
                "assumed_db_unit": "string",
                "factor_kgco2e_per_unit": 0.0,
                "kgco2e": 0.0,
                "category": "string",
                "notes": "short note on assumptions or unit fixes"
            }
        ],
        "by_category": [
            {"category":"Concrete","kgco2e":0.0}
        ],
        "flags": [
            "short warnings, e.g. unit uncertainty on line 3"
        ]
    }
    prompt = (
        "You are a sustainability professional. Estimate embodied CO2 for the extracted invoice or BOM. "
        "You must output valid JSON only. No extra text. "
        "Use the factor hints and mapping hints when possible. "
        "If units do not match, make a conservative assumption and note it in 'notes'. "
        "For totals, include a conservative range of ±15% unless there are clear unit mismatches. "
        "Never invent items that are not present. "
        "\n\nFACTOR_HINTS:\n" + json.dumps(FACTOR_HINTS) +
        "\n\nMAPPING_HINTS:\n" + json.dumps(MAPPING_HINTS) +
        "\n\nEXTRACTED_MARKDOWN_EXCERPT:\n" + extracted_md[:6000] +
        "\n\nPARSED_ITEMS_TABLE:\n" + items_df.to_json(orient="records") +
        "\n\nJSON_SCHEMA:\n" + json.dumps(schema)
    )
    return prompt

# -----------------------------
# Step 2 - LLM pro analysis
# -----------------------------
def make_pro_analysis_prompt(json_payload: dict) -> str:
    task = (
        "You are a senior sustainability consultant. Based on the JSON results, write a concise analysis. "
        "1) Identify the top drivers and why. "
        "2) Provide a defensible total range. "
        "3) Give 3 practical substitutions or procurement actions that maintain function. "
        "4) Keep it under 160 words. "
        "5) Avoid buzzwords."
    )
    return task + "\n\nRESULTS_JSON:\n" + json.dumps(json_payload)[:8000]


# -----------------------------
# Fallback local estimator if HF not available
# -----------------------------
def local_estimator(items_df: pd.DataFrame) -> dict:
    # Very simple keyword mapping and factors
    keys = [h["material_key"] for h in FACTOR_HINTS]
    name_map = {k:k for k in keys}

    def fuzzy_key(text: str):
        cand = process.extractOne(text.lower(), keys, scorer=fuzz.WRatio)
        if not cand:
            return None, 0
        k, score, _ = cand
        return k, score

    def factor_for(key: str):
        for h in FACTOR_HINTS:
            if h["material_key"] == key:
                return h
        return None

    lines = []
    total = 0.0
    for _, r in items_df.iterrows():
        item = str(r["item"])
        qty = float(r["quantity"])
        unit = str(r["unit"]).lower()

        k, score = fuzzy_key(item)
        h = factor_for(k) if k else None
        db_unit = h["unit"] if h else ""
        factor = float(h["factor_kgco2e_per_unit"]) if h else 0.0

        qty_conv = qty
        # Minimal conversion for kg <-> t
        if unit == "kg" and db_unit == "t":
            qty_conv = qty / 1000.0
        elif unit == "t" and db_unit == "kg":
            qty_conv = qty * 1000.0
        elif unit and db_unit and unit != db_unit:
            # Unknown conversion
            qty_conv = np.nan

        kgco2e = qty_conv * factor if not pd.isna(qty_conv) else np.nan
        if not pd.isna(kgco2e):
            total += kgco2e

        lines.append({
            "item": item,
            "quantity": qty,
            "input_unit": unit,
            "matched_material_key": k or "",
            "assumed_db_unit": db_unit,
            "factor_kgco2e_per_unit": factor,
            "kgco2e": 0.0 if pd.isna(kgco2e) else float(kgco2e),
            "category": h["category"] if h else "",
            "notes": "" if not pd.isna(kgco2e) else "Unit mismatch - manual review"
        })

    by_cat = {}
    for ln in lines:
        c = ln.get("category","") or "Unknown"
        by_cat[c] = by_cat.get(c, 0.0) + float(ln.get("kgco2e", 0.0))
    by_category = [{"category":k, "kgco2e":v} for k,v in by_cat.items()]

    return {
        "total_kgco2e": float(total),
        "uncertainty_low_kgco2e": float(total*0.85),
        "uncertainty_high_kgco2e": float(total*1.15),
        "lines": lines,
        "by_category": by_category,
        "flags": []
    }

def local_pro_analysis(payload: dict) -> str:
    total = payload.get("total_kgco2e", 0.0)
    low = payload.get("uncertainty_low_kgco2e", 0.0)
    high = payload.get("uncertainty_high_kgco2e", 0.0)
    lines = payload.get("lines", [])
    df = pd.DataFrame(lines)
    df = df.sort_values("kgco2e", ascending=False)
    top = df.head(3)
    items = "; ".join([f"{r['item'][:28]} ({r['kgco2e']:.0f} kg)" for _, r in top.iterrows()])
    recs = []
    for _, r in top.iterrows():
        k = r.get("matched_material_key","")
        if "concrete_cem1" in k:
            recs.append("Switch to CEM II or reduce cement content")
        if "rebar_virgin" in k:
            recs.append("Specify high recycled content rebar")
        if "al_extruded_primary" in k:
            recs.append("Use recycled content aluminum")
    if not recs:
        recs = ["Prefer suppliers with verified low carbon EPDs", "Check unit consistency", "Right-size specs for top drivers"]
    return (
        f"Estimated total {total:,.0f} kgCO2e "
        f"(range {low:,.0f} to {high:,.0f}). "
        f"Top drivers: {items}. "
        f"Actions: {', '.join(recs[:3])}."
    )


# -----------------------------
# UI layout
# -----------------------------
st.title("CarbonSpec - LLM-first PDF MVP")
st.caption("Left: upload PDFs. Right: LLM Step 1 - JSON estimation and analytics. LLM Step 2 - pro analysis.")

left, right = st.columns([1, 1])

with left:
    st.subheader("Upload PDFs")
    files = st.file_uploader("Upload invoices or BOM PDFs", type=["pdf"], accept_multiple_files=True)
    st.markdown("**Hugging Face**")
    model_repo = st.selectbox(
        "Hosted model",
        ["meta-llama/Llama-3-8b-instruct", "meta-llama/Llama-3-70b-instruct"],
        index=0
    )
    if hf_token():
        st.success("HF token detected")
    else:
        st.info("Set HF_API_TOKEN in env or Streamlit secrets")
    run_btn = st.button("Process")

with right:
    st.subheader("Results")

if run_btn and files:
    # Extract text
    md_blobs = []
    for f in files:
        try:
            md = extract_markdown_from_pdf(f.read())
        except Exception as e:
            md = f"[Error extracting {f.name}: {e}]"
        md_blobs.append(f"# File: {f.name}\n\n{md}")
    combined_md = "\n\n---\n\n".join(md_blobs)

    # Parse items
    items_df = parse_items_from_markdown(combined_md)

    with right:
        st.markdown("**Extracted items**")
        if items_df.empty:
            st.info("No line items detected. Adjust parsing or try a different PDF.")
        else:
            st.dataframe(items_df, use_container_width=True)

    # Step 1 - LLM JSON estimator
    estimator_prompt = make_json_estimator_prompt(combined_md, items_df)
    json_text = hf_generate(model_repo, estimator_prompt, max_new_tokens=700, temperature=0.2)

    payload = None
    if json_text:
        # Try to isolate last JSON block if the model returns extra text
        try:
            # Simple heuristic: find first { and last } and parse
            i, j = json_text.find("{"), json_text.rfind("}")
            payload = json.loads(json_text[i:j+1]) if i != -1 and j != -1 else json.loads(json_text)
        except Exception:
            payload = None

    if payload is None:
        st.caption("HF estimator unavailable or returned invalid JSON. Using local estimator.")
        payload = local_estimator(items_df)

    # Show analytics from payload
    with right:
        st.subheader("LLM Step 1 - Estimated CO2 and analytics")
        total = float(payload.get("total_kgco2e", 0.0))
        low = float(payload.get("uncertainty_low_kgco2e", max(0.0, total*0.85)))
        high = float(payload.get("uncertainty_high_kgco2e", total*1.15))
        st.metric("Estimated total embodied CO2", f"{total:,.0f} kgCO2e")
        st.caption(f"Range: {low:,.0f} to {high:,.0f} kgCO2e")

        lines = pd.DataFrame(payload.get("lines", []))
        if not lines.empty:
            st.markdown("**Per-line results**")
            st.dataframe(lines, use_container_width=True)

            # Hotspots
            df_hot = lines.dropna(subset=["kgco2e"]).sort_values("kgco2e", ascending=False).head(10)
            if not df_hot.empty:
                fig = plt.figure()
                plt.bar(df_hot["item"].astype(str).str[:24], df_hot["kgco2e"])
                plt.title("Top items by kgCO2e")
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("kgCO2e")
                st.pyplot(fig)

        by_cat = pd.DataFrame(payload.get("by_category", []))
        if not by_cat.empty:
            st.markdown("**By category**")
            st.dataframe(by_cat.sort_values("kgco2e", ascending=False), use_container_width=True)

        flags = payload.get("flags", [])
        if flags:
            st.warning("Notes: " + " | ".join(flags))

        # Download
        csv_io = io.StringIO()
        if not lines.empty:
            lines.to_csv(csv_io, index=False)
            st.download_button("Download line results CSV", data=csv_io.getvalue(),
                               file_name="carbonspec_line_results.csv", mime="text/csv")

    # Step 2 - LLM pro analysis
    with right:
        st.subheader("LLM Step 2 - Professional analysis")
        analysis_prompt = make_pro_analysis_prompt(payload)
        analysis_text = hf_generate(model_repo, analysis_prompt, max_new_tokens=400, temperature=0.2)
        if analysis_text:
            st.write(analysis_text.strip())
        else:
            st.caption("HF analysis unavailable. Using local analysis.")
            st.write(local_pro_analysis(payload))

elif run_btn and not files:
    st.warning("Please upload at least one PDF.")
