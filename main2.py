import streamlit as st
from transformers import pipeline

st.title("Spam vs Ham Classifier 🤖")

# Load the classifier
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# User input
user_input = st.text_area("Enter a message to classify:")

# Mapping from LABEL_x to custom tags
label_map = {
    "LABEL_0": "Positive (Ham)",   # Not spam
    "LABEL_1": "Negative (Spam)"   # Spam
}

# Prediction
if user_input:
    result = classifier(user_input)[0]
    label = label_map.get(result["label"], result["label"])
    confidence = result["score"] * 100
    st.success(f"**Prediction:** {label} ({confidence:.2f}%)")

