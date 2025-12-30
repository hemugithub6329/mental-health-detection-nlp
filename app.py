import streamlit as st
import pickle
import numpy as np
import re
import nltk
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# NLTK downloads (run once)
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Mental Health Detection from Text",
    layout="centered"
)

# -----------------------------
# Load ML Artifacts
# -----------------------------
with open("models/mental_health_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("models/emotion_encoder.pkl", "rb") as f:
    emotion_encoder = pickle.load(f)

# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# Semantic Emotion Detection (VADER)
# -----------------------------
sia = SentimentIntensityAnalyzer()

def detect_emotion_vader(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.5:
        return "joy"
    elif compound <= -0.5:
        return "sadness"
    elif scores["neg"] > scores["pos"]:
        return "fear"
    else:
        return "neutral"

# -----------------------------
# Safety Filter (Violence / Crime)
# -----------------------------
violence_keywords = [
    "killed", "kill", "murder", "stab", "shot", "rape",
    "assault", "beat", "abuse", "dead body", "homicide"
]

# -----------------------------
# Severity Logic
# -----------------------------
def assign_severity(predicted_label, confidence):
    if predicted_label == "Suicidal":
        return "High"
    elif predicted_label in ["Depression", "Bipolar"]:
        return "High" if confidence >= 0.7 else "Medium"
    elif predicted_label in ["Anxiety", "Stress"]:
        return "Medium" if confidence >= 0.6 else "Low"
    else:
        return "Low"

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Mental Health Detection from Social Media Text")
st.info("⚠️ This tool is for educational purposes only. Not a medical diagnosis.")

user_input = st.text_area(
    "Enter a social media post or personal text:",
    height=150
)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")

    elif any(word in user_input.lower() for word in violence_keywords):
        st.error("⚠️ Violent or criminal content detected.")
        st.info(
            "This system is designed only for mental health analysis "
            "and cannot process content involving harm to others or yourself."
        )

    else:
        # -----------------------------
        # Preprocessing
        # -----------------------------
        cleaned_text = clean_text(user_input)

        # -----------------------------
        # Emotion Detection (Semantic & Stable)
        # -----------------------------
        emotion = detect_emotion_vader(user_input)

        # -----------------------------
        # Feature Creation
        # -----------------------------
        X_text = tfidf.transform([cleaned_text])
        emotion_encoded = emotion_encoder.transform([emotion]).reshape(-1, 1)
        X_final = hstack([X_text, emotion_encoded])

        # -----------------------------
        # Prediction
        # -----------------------------
        pred_class = model.predict(X_final)[0]
        confidence = np.max(model.predict_proba(X_final))
        predicted_label = label_encoder.inverse_transform([pred_class])[0]

        # -----------------------------
        # Severity
        # -----------------------------
        severity = assign_severity(predicted_label, confidence)

        # -----------------------------
        # Explainable AI
        # -----------------------------
        feature_names = tfidf.get_feature_names_out()
        coefficients = model.coef_[pred_class]
        word_weights = list(zip(feature_names, coefficients))

        top_words = sorted(
            word_weights,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]

        # -----------------------------
        # Display Results
        # -----------------------------
        st.subheader("🔍 Results")
        st.write(f"**Mental Health Condition:** {predicted_label}")
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Severity Level:** {severity}")
        st.write(f"**Confidence:** {round(confidence, 3)}")

        st.subheader("🧾 Explanation (Why this result?)")
        st.write(
            "The model identified emotionally significant words in the input text. "
            "These words influenced the prediction based on patterns learned during training."
        )

        for word, weight in top_words:
            importance = abs(weight)
            if importance > 3:
                level = "Very strong influence"
            elif importance > 1.5:
                level = "Strong influence"
            else:
                level = "Moderate influence"
            st.write(f"• **{word}** → {level}")
