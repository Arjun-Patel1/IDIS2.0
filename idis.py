import streamlit as st
from idis_pipeline import run_pipeline

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="IDIS 2.0 – Intelligent Document Intelligence System",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.title("🧠 IDIS 2.0")
st.subheader("Intelligent Document Intelligence System")
st.write(
    "Classifies customer support issues, retrieves similar historical cases, "
    "and generates AI-based explanations with business signals."
)

st.divider()

# ------------------------------------------------------
# USER INPUT
# ------------------------------------------------------
user_input = st.text_area(
    "Enter customer issue / support ticket:",
    placeholder="e.g. I have not received my debit card yet",
    height=120
)

analyze_btn = st.button("🔍 Analyze Ticket")

# ------------------------------------------------------
# RUN PIPELINE
# ------------------------------------------------------
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter a customer issue before analysis.")
    else:
        with st.spinner("Running IDIS pipeline..."):
            try:
                result = run_pipeline(user_input)

                st.success("✅ Analysis Complete")

                # ---------------- CATEGORY ----------------
                st.markdown("### 🏷️ Category")
                st.write(result["predicted_category"])

                # ---------------- CONFIDENCE ----------------
                st.markdown("### 📊 Confidence")
                st.write(result["confidence"])

                st.markdown("### 🧮 Calibrated Confidence")
                st.write(result["calibrated_confidence"])

                st.markdown("### 🔐 Trust Level")
                st.write(result["trust_level"])

                # ---------------- BUSINESS SIGNALS ----------------
                st.markdown("### ⚡ Priority")
                st.write(result["priority"])

                st.markdown("### ⏱️ SLA Risk")
                st.write(result["sla_risk"])

                st.markdown("### 📌 Recommended Action")
                st.write(result["recommended_action"])

                st.markdown("### 👤 Human Review Required")
                st.write(result["human_review_required"])

                # ---------------- RAG EXPLANATION ----------------
                st.markdown("### 🧠 RAG Explanation")
                st.write(result["explanation"])

                # ---------------- SIMILAR CASES ----------------
                st.markdown("### 📚 Similar Historical Cases")
                for i, case in enumerate(result["similar_cases"], start=1):
                    st.write(f"**Case {i}:** {case}")

                # ---------------- RAW JSON ----------------
                with st.expander("🔍 View Raw JSON Output"):
                    st.json(result)

            except Exception as e:
                st.error("❌ An error occurred while running the pipeline.")
                st.exception(e)
