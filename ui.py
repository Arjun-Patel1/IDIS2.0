import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="IDIS – Intelligent Document Intelligence System",
    layout="wide"
)

st.title("🧠 Intelligent Document Intelligence System (IDIS)")
st.caption(
    "AI-powered customer support issue classification with "
    "Retrieval-Augmented Generation (RAG)"
)

# ---------------- INPUT ----------------
st.subheader("📩 Enter Customer Support Ticket")

user_input = st.text_area(
    "Paste the customer issue text below:",
    height=120,
    placeholder="Example: I still haven't received my debit card after two weeks"
)

analyze_btn = st.button("🚀 Analyze Issue")

# ---------------- PROCESS ----------------
if analyze_btn and user_input.strip():
    with st.spinner("Analyzing ticket..."):
        response = requests.post(
            API_URL,
            json={"text": user_input}
        )

    if response.status_code != 200:
        st.error("❌ API Error")
        st.code(response.text)
    else:
        data = response.json()

        # ---------------- RESULTS ----------------
        st.divider()
        st.subheader("✅ Analysis Complete")

        # ---- Category & Confidence ----
        col1, col2, col3 = st.columns(3)
        col1.metric("🏷️ Category", data["predicted_category"])
        col2.metric("📊 Confidence", data["confidence"])
        col3.metric(
            "🧮 Calibrated Confidence",
            data["calibrated_confidence"]
        )

        # ---- Trust & Priority ----
        col4, col5, col6 = st.columns(3)
        col4.metric("🔐 Trust Level", data["trust_level"])
        col5.metric("⚡ Priority", data["priority"])
        col6.metric("⏱️ SLA Risk", str(data["sla_risk"]))

        # ---- Action ----
        st.success(f"📌 Recommended Action: {data['recommended_action']}")

        if data["human_review_required"]:
            st.warning("👤 Human Review Required")

        # ---- Explanation ----
        st.subheader("🧠 RAG Explanation")
        st.write(data["explanation"])

        # ---- Similar Cases ----
        st.subheader("📚 Similar Historical Cases")
        if data["similar_cases"]:
            for i, case in enumerate(data["similar_cases"], 1):
                st.markdown(f"**Case {i}:** {case}")
        else:
            st.info("No similar cases found.")

        # ---- Raw JSON (Debug / Recruiter Friendly) ----
        with st.expander("🔍 View Raw JSON Output"):
            st.code(json.dumps(data, indent=2))
