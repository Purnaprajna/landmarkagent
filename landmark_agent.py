import streamlit as st
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline

# ------------------------------
# 1. Load Vision Model (BLIP)
# ------------------------------
@st.cache_resource
def load_vision_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


processor, vision_model = load_vision_model()


# ------------------------------
# 2. Load Text Model (TinyLlama)
# ------------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto"
    )


llm = load_llm()


# ------------------------------
# 3. Vision Function (Image → Landmark)
# ------------------------------
def get_landmark_name(image_file):
    image = Image.open(image_file).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    out = vision_model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


# ------------------------------
# 4. Answer Function (LLM reasoning)
# ------------------------------
def answer_question(landmark, question):
    prompt = f"""
You are a helpful assistant.

Landmark: {landmark}
Question: {question}

Give a short, correct, and simple answer.
"""

    result = llm(prompt, max_new_tokens=200)

    return result[0]["generated_text"]


# ------------------------------
# 5. Streamlit UI
# ------------------------------
st.title("🗺️ Landmark Helper (Free AI Version)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
question = st.text_input("Ask a question about the landmark")

if uploaded_file and question:

    # Step 1: Vision
    with st.spinner("Detecting landmark..."):
        landmark = get_landmark_name(uploaded_file)
        st.success(f"Detected: {landmark}")

    # Step 2: Answer
    with st.spinner("Thinking..."):
        answer = answer_question(landmark, question)
        st.write(answer)