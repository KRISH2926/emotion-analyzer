# AI Emotion Analyzer

This project detects emotions from text using Natural Language Processing and Machine Learning.

The system analyzes emotional tone in a paragraph and visualizes emotional transitions across sentences.

## Features

- Emotion Prediction
- Emotion Journey Analysis
- Emotion Distribution Visualization
- Emotion Heatmap Visualization
- Interactive Web Interface (Gradio)

## Technologies Used

- Python
- Scikit-learn
- TF-IDF
- Support Vector Machine (SVM)
- Gradio
- Matplotlib
- Seaborn

## How It Works

1. Text is preprocessed and converted into TF-IDF vectors.
2. A trained SVM model predicts emotions.
3. The system analyzes emotions sentence by sentence.
4. Visualizations such as distribution charts and heatmaps are generated.

## Model Accuracy

Accuracy achieved: **~90%**

## Project Structure
emotion-analyzer
│
├── app.py
├── emotion_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── AI_Emotion_Recognition.ipynb
└── README.md

## Demo

The application allows users to enter a paragraph and analyze emotional patterns across the text.
