# Multilingual-sentiment-analysis

This project is a multilingual sentiment analysis web app supporting English, Hindi, and Telugu, developed using Streamlit and HuggingFace Transformers.

Accepts both typed and spoken inputs
Automatically detects input language using langdetect
Uses xlm-roberta-base or bert-base-multilingual-uncased for prediction
Categorizes sentiment into Positive, Neutral, or Negative with confidence scores
Supports CSV upload for batch sentiment analysis
Displays sentiment distribution in simple charts
Allows downloading of sentiment analysis results
Voice input supports English, Hindi, and Telugu
Limitations: Voice input requires additional dependencies like SpeechRecognition and PyAudio
Limitations: Accuracy may vary depending on accent and device microphone
