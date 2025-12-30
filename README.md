# Mental Health Detection from Social Media Text using NLP

This project detects mental health conditions from social media text using
Natural Language Processing (NLP) and Machine Learning.

## Features
- Mental health classification (Normal, Depression, Anxiety, Stress, etc.)
- Emotion detection
- Severity assessment
- Explainable AI (important words influencing prediction)
- Ethical safety layer for violent content
- Streamlit-based interactive web app

## Tech Stack
- Python
- NLP (TF-IDF, NLTK)
- Machine Learning (Logistic Regression)
- Streamlit
- VADER Sentiment Analysis

## 📁 Project Structure
mental-health-nlp/
├── app.py
├── models/
├── data/
├── notebooks/
├── requirements.txt
└── README.md


## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py