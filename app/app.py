
import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔍",
    layout="centered"
)

# Load trained model and TF-IDF vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("models/best_model.joblib")
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    return model, tfidf

model, tfidf = load_models()

# Website heading
st.title("🔍 TruthLens AI")
st.subheader("AI-Powered Fake News Detection")

st.write(
    "Enter a news headline or article below. "
    "Our machine-learning model will analyse its text."
)

st.divider()

# News input
news = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste your news headline or article here..."
)

# Prediction button
if st.button("Analyse News", type="primary"):

    if not news.strip():
        st.warning("Please enter a news article first.")

    else:
        # Clean input
        cleaned_news = news.lower()
        cleaned_news = re.sub(r"http\S+|www\S+", "", cleaned_news)
        cleaned_news = re.sub(r"[^a-z\s]", " ", cleaned_news)
        cleaned_news = re.sub(r"\s+", " ", cleaned_news).strip()

        if not cleaned_news:
            st.warning("Please enter valid English text.")
        else:
            # Convert text into TF-IDF
            news_vector = tfidf.transform([cleaned_news])

            # Predict
            prediction = model.predict(news_vector)[0]

            # Get important words
            feature_names = tfidf.get_feature_names_out()
            coefficients = model.coef_[0]

            contributions = (
                news_vector.multiply(coefficients).toarray()[0]
            )

            if prediction == 0:
                st.error("🚨 Prediction: FAKE-LIKE NEWS")
                direction = -1
            else:
                st.success("✅ Prediction: REAL-LIKE NEWS")
                direction = 1

            # Display influential words
            important_indices = [
                i for i, value in enumerate(contributions)
                if value * direction > 0
            ]

            important_indices.sort(
                key=lambda i: abs(contributions[i]),
                reverse=True
            )

            st.subheader("Why did the model predict this?")

            if important_indices:
                for i in important_indices[:10]:
                    st.write(
                        f"• **{feature_names[i]}** "
                        f"({abs(contributions[i]):.4f})"
                    )
            else:
                st.info("No influential words found.")

            st.caption(
                "Important words show which text features influenced "
                "the model's prediction."
            )

st.divider()

st.info(
    "Disclaimer: This AI prediction is based on patterns learned "
    "from a dataset. It does not verify whether a news article "
    "is factually true or false. Always check reliable sources."
)

st.caption("Developed as an NLP and Machine Learning project.")
