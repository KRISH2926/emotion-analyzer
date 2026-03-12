# AI Emotion Analyzer

## Name
Krish

## Project Title
AI Emotion Analyzer – Emotion Recognition from Text using Natural Language Processing

## Project Description
The AI Emotion Analyzer is a machine learning application designed to detect emotions from textual input. The system analyzes user-provided text and predicts the underlying emotional tone, such as sadness, joy, love, anger, fear, or surprise.

The application also provides a deeper analysis by examining emotional patterns across multiple sentences. This is achieved through visualizations such as emotion distribution graphs and heatmaps.

An interactive web interface built using Gradio allows users to enter text and instantly view the predicted emotions and visual analytics.

## Methodology

### 1. Data Collection
A labeled dataset containing sentences with associated emotions was used for training the model. Each text sample was categorized into one of six emotion classes:
- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

### 2. Data Preprocessing
Text preprocessing steps included:
- Text cleaning
- Removal of stopwords
- Tokenization
- Feature extraction using TF-IDF (Term Frequency – Inverse Document Frequency)

### 3. Feature Extraction
TF-IDF vectorization was used to convert textual data into numerical feature vectors that can be processed by machine learning models.

### 4. Model Training
Two machine learning models were evaluated:

- Logistic Regression
- Support Vector Machine (SVM)

The Support Vector Machine model produced the best performance and was selected as the final model.

### 5. Emotion Analysis System
The final system includes several analytical features:

- Emotion Prediction
- Sentence-wise Emotion Journey
- Emotion Distribution Visualization
- Emotion Heatmap Visualization

### 6. User Interface
An interactive interface was developed using **Gradio**, which allows users to input text and analyze emotional patterns in real time.

## Results
The trained SVM model achieved an approximate accuracy of **~90%** on the test dataset.

The application successfully performs:
- Emotion prediction from text
- Emotion analysis across sentences
- Visualization of emotional distribution
- Heatmap representation of emotional patterns

## Technologies Used
- Python
- Scikit-learn
- TF-IDF Vectorization
- Support Vector Machine (SVM)
- Pandas and NumPy
- Matplotlib and Seaborn
- Gradio

## Live Application
You can try the live application here:

https://huggingface.co/spaces/Krish2926/emotion-analyzer

## Project Structure
emotion-analyzer
│
├── app.py
├── emotion_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── AI_Emotion_Recognition.ipynb
└── README.md

## Future Improvements
Future improvements could include:

- Using deep learning models such as LSTM or BERT
- Expanding the dataset for improved accuracy
- Real-time emotion monitoring
- Integration with chatbots or social media analytics systems

