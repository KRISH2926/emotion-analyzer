import gradio as gr
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

# Load model and vectorizer
model = pickle.load(open("emotion_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

emotion_map = {
0:"Sadness",
1:"Joy",
2:"Love",
3:"Anger",
4:"Fear",
5:"Surprise"
}

emotion_to_number = {
"Sadness":0,
"Joy":1,
"Love":2,
"Anger":3,
"Fear":4,
"Surprise":5
}

def predict_emotion(text):

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    return emotion_map[prediction]


def analyze_emotion(paragraph):

    overall_emotion = predict_emotion(paragraph)

    sentences = re.split(r'[.!?]', paragraph)

    emotions = []
    journey_output = []

    for s in sentences:
        s = s.strip()
        if s:
            emotion = predict_emotion(s)
            emotions.append(emotion)
            journey_output.append(f"{s} → {emotion}")

    journey_text = "\n".join(journey_output)

    # Emotion Distribution
    fig1 = plt.figure(figsize=(6,4))
    count = Counter(emotions)

    plt.bar(count.keys(), count.values())
    plt.title("Emotion Distribution")

    # Emotion Heatmap
    numbers = [emotion_to_number[e] for e in emotions]

    df = pd.DataFrame([numbers], columns=[f"S{i+1}" for i in range(len(numbers))])

    fig2 = plt.figure(figsize=(8,2))
    sns.heatmap(df, annot=[emotions], fmt="", cmap="coolwarm", cbar=False)

    plt.title("Emotion Heatmap")

    return overall_emotion, journey_text, fig1, fig2


with gr.Blocks() as demo:

    gr.Markdown("# AI Emotion Analyzer")

    text_input = gr.Textbox(lines=6,label="Enter Paragraph")

    analyze_btn = gr.Button("Analyze Emotion")

    emotion_output = gr.Text(label="Overall Emotion")

    journey_output = gr.Textbox(label="Emotion Journey")

    dist_plot = gr.Plot(label="Emotion Distribution")

    heatmap_plot = gr.Plot(label="Emotion Heatmap")

    analyze_btn.click(
        analyze_emotion,
        inputs=text_input,
        outputs=[emotion_output, journey_output, dist_plot, heatmap_plot]
    )

demo.launch()