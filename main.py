import streamlit as st
import json
import requests
from openai import OpenAI
from groq import Groq
import base64
import os

# Set up the page layout
st.set_page_config(page_title="Model Match", layout="wide")

# Sidebar for navigation
with st.sidebar:
    st.title("Model Match")
    page = st.selectbox("Select Mode", ["Text Comparison", "Image Comparison (Coming Soon)", "Audio Comparison (Coming Soon)"])

# Define model options organized by group
model_options = {
    "OpenAI": [
        "GPT-4o (gpt-4o-2024-05-13)",
        "GPT-4 Turbo (gpt-4-turbo-2024-04-09)",
        "GPT-4 Turbo (gpt-4-turbo-preview)",
        "GPT-4 Turbo (gpt-4-0125-preview)",
        "GPT-4 Turbo (gpt-4-1106-preview)",
        "GPT-4 Turbo (gpt-4-vision-preview)",
        "GPT-4 Turbo (gpt-4-1106-vision-preview)",
        "GPT-3.5 Turbo (gpt-3.5-turbo-0125)",
        "GPT-3.5 Turbo (gpt-3.5-turbo)",
        "GPT-3.5 Turbo (gpt-3.5-turbo-1106)",
        "GPT-3.5 Turbo (gpt-3.5-turbo-instruct)",
        "GPT-3.5 Turbo (gpt-3.5-turbo-16k)",
        "GPT-3.5 Turbo (gpt-3.5-turbo-0613)",
        "GPT-3.5 Turbo (gpt-3.5-turbo-16k-0613)"
    ],
    "Gemini": [
        "Gemini 1.5 Pro (gemini-1.5-pro-latest)",
        "Gemini 1.5 Flash (gemini-1.5-flash-latest)",
        "Gemini 1.0 Pro (gemini-1.0-pro-latest)"
    ],
    "Groq": [
        "gemma-7b-it",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ]
}

# Function for making API calls to Gemini
def gemini(system_prompt, user_prompt, expected_format, url):
    payload = json.dumps({
        "contents": [
            {
                "parts": [
                    {"text": f"system_prompt : {system_prompt}"},
                    {"text": f"user_prompt : {user_prompt}"},
                    {"text": f"expected_format : {expected_format}"}
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    })

    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()
    text_value = response_data["candidates"][0]["content"]["parts"][0]["text"]
    return text_value

# Function for making API calls to OpenAI
def gpt(system_prompt, user_prompt, expected_format, gptkey, model):
    client = OpenAI(api_key=gptkey)
    chat_completion, *_ = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"system_prompt : {system_prompt}"},
            {"role": "user", "content": f"user_prompt : {user_prompt}"},
            {"role": "user", "content": f"expected_JSON_format : {expected_format}"}
        ],
        model=f"{model}",
        response_format={"type": "json_object"},
    ).choices
    content = chat_completion.message.content
    return content

# Function for making API calls to Groq
def groq(system_prompt, user_prompt, expected_format, groqkey, model="llama3-70b-8192"):
    client = Groq(api_key=groqkey)
    completion = client.chat.completions.create(
        model=f"{model}",
        messages=[
            {
                "role": "system",
                "content": f"output only JSON object. {system_prompt}"
            },
            {
                "role": "user",
                "content": f"{expected_format}"
            },
            {
                "role": "user",
                "content": f"{user_prompt}"
            }
        ],
        response_format={"type": "json_object"},
        stop=None,
    )
    content = completion.choices[0].message.content
    return json.loads(content)

# Text Comparison Page
if page == "Text Comparison":
    st.header("Text Comparison")

    # API keys input
    st.subheader("Enter API Keys")
    gemini_api_key_text = st.text_input("Gemini API Key (Text)", type="password", key="gemini_text")
    openai_api_key_text = st.text_input("OpenAI API Key (Text)", type="password", key="openai_text")
    groq_api_key_text = st.text_input("Groq API Key (Text)", type="password", key="groq_text")

    # Dropdown to select models for comparison
    selected_models = []
    for group, models in model_options.items():
        st.subheader(group)
        selected_models += st.multiselect(f"Select {group} models", models, key=f"{group}_models")

    selected_models = selected_models[:5]  # Limit to 5 models

    # Text input for user prompt
    user_prompt = st.text_area("Enter your prompt here")

    if st.button("Compare"):
        if len(selected_models) == 0:
            st.warning("Please select at least one model for comparison.")
        elif not user_prompt:
            st.warning("Please enter a prompt.")
        else:
            st.write("Comparing outputs for the selected models...")

            # Placeholder for the honeycomb structure of model outputs
            cols = st.columns(5)
            for i, model in enumerate(selected_models):
                with cols[i % 5]:
                    if "Gemini" in model:
                        output = gemini("system_prompt", user_prompt, "expected_format", gemini_api_key_text)
                    elif "OpenAI" in model:
                        output = gpt("system_prompt", user_prompt, "expected_format", openai_api_key_text, model)
                    elif "Groq" in model:
                        output = groq("system_prompt", user_prompt, "expected_format", groq_api_key_text)
                    else:
                        output = "Model not supported"

                    st.write(f"Output from {model}:")
                    st.write(output)

# Image Comparison Page
elif page == "Image Comparison (Coming Soon)":
    st.header("Image Comparison")
    st.write("Coming Soon...")

# Audio Comparison Page
elif page == "Audio Comparison (Coming Soon)":
    st.header("Audio Comparison")
    st.write("Coming Soon...")
