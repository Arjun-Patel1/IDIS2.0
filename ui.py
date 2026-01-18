import streamlit as st
import requests

st.set_page_config(page_title="IDIS 2.0", layout="centered")
st.title("🧠 Intelligent Document Intelligence System")

# Text input
text = st.text_area("Enter customer issue:")

if st.button("Analyze") and text.strip():
    with st.spinner("Analyzing..."):
        try:
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text},
                timeout=60
            )
            data = res.json()
            st.session_state.data = data
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            st.session_state.data = None

# Display results if prediction succeeded
if "data" in st.session_state and st.session_state.data:
    data = st.session_state.data
    if "predicted_category" in data:
        st.success("✅ Analysis Complete")
        st.metric("🏷️ Category", data.get("predicted_category", "N/A"))
        st.metric("📊 Confidence", data.get("confidence", 0))
        st.metric("🔐 Trust Level", data.get("trust_level", "N/A"))

        st.subheader("🧠 Explanation")
        st.write(data.get("rag_explanation", "No explanation available."))

        st.subheader("📚 Similar Historical Cases")
        for case in data.get("similar_cases", []):
            st.markdown(f"- **{case.get('label', 'N/A')}** → {case.get('text', '')}")

        st.subheader("🔍 Raw JSON Output")
        st.json(data)

        st.subheader("📝 Feedback")
        col1, col2 = st.columns(2)

        if col1.button("✅ Prediction is Correct"):
            try:
                requests.post(
                    "http://127.0.0.1:8000/feedback",
                    json={
                        "query": text,
                        "predicted_category": data.get("predicted_category", ""),
                        "confidence": data.get("confidence", 0),
                        "trust_level": data.get("trust_level", ""),
                        "user_feedback": "correct"
                    }
                )
                st.success("Thanks! Feedback saved.")
            except Exception as e:
                st.error(f"❌ Failed to save feedback: {e}")

        if col2.button("❌ Prediction is Incorrect"):
            correct = st.selectbox(
                "Select correct category",
                ["atm_issue", "card_issue", "net_banking", "upi_issue", "other"],
                key="correct_category"
            )
            if st.button("Submit Correction"):
                try:
                    requests.post(
                        "http://127.0.0.1:8000/feedback",
                        json={
                            "query": text,
                            "predicted_category": data.get("predicted_category", ""),
                            "confidence": data.get("confidence", 0),
                            "trust_level": data.get("trust_level", ""),
                            "user_feedback": "incorrect",
                            "correct_category": correct
                        }
                    )
                    st.success("Correction saved for retraining.")
                except Exception as e:
                    st.error(f"❌ Failed to save correction: {e}")

