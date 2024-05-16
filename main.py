import streamlit as st
import json
import requests
from openai import OpenAI
from groq import Groq
import google.generativeai as genai

# Set up the page layout
st.set_page_config(page_title="Model Match", layout="wide")

# Sidebar for navigation
with st.sidebar:
    st.title("Model Match")
    page = st.selectbox("Select Mode", ["Text Comparison", "Image Comparison (Coming Soon)", "Audio Comparison (Coming Soon)"])

# Define model options organized by group
model_options = {
    "OpenAI": [
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4-1106-vision-preview",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613"
    ],
    "Gemini": [
        "models/gemini-1.5-pro-latest",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.0-pro-latest"
    ],
    "Groq": [
        "gemma-7b-it",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ]
}

# Function for making API calls to Gemini using google.generativeai
def gemini(system_prompt, user_prompt, expected_format, gemini_api_key, model):
    genai.configure(api_key=gemini_api_key)
    try:
        response = genai.chat(
            model=model,
            messages=[
                {"content": system_prompt},
                {"content": user_prompt}
            ]
        )
        return response["candidates"][0]["content"]
    except Exception as e:
        return f"Error: {e}"

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
def groq(system_prompt, user_prompt, expected_format, groqkey, model):
    client = Groq(api_key=groqkey)
    completion = client.chat.completions.create(
        model=f"{model}",
        messages=[
            {"role": "system", "content": f"output only JSON object. {system_prompt}"},
            {"role": "user", "content": f"{expected_format}"},
            {"role": "user", "content": f"{user_prompt}"}
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    reply = json.loads(content)
    return reply

# Helper function to map model names to appropriate API calls
def call_model_api(model, system_prompt, user_prompt, expected_format, keys):
    if "models/" in model:
        return gemini(system_prompt, user_prompt, expected_format, keys['gemini'], model)
    elif "gpt" in model:
        return gpt(system_prompt, user_prompt, expected_format, keys['openai'], model)
    elif "llama" in model or "gemma" in model or "mixtral" in model:
        return groq(system_prompt, user_prompt, expected_format, keys['groq'], model)
    else:
        return "Model not supported"

# Streamlit app UI
st.title("Model Match")
st.write("Compare outputs from different AI models.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text Comparison Page
if page == "Text Comparison":
    st.header("Text Comparison")

    # API keys input
    st.subheader("Enter API Keys")
    gemini_api_key_text = st.text_input("Gemini API Key (Text)", type="password")
    openai_api_key_text = st.text_input("OpenAI API Key (Text)", type="password")
    groq_api_key_text = st.text_input("Groq API Key (Text)", type="password")

    # Store keys in a dictionary
    api_keys = {
        "gemini": gemini_api_key_text,
        "openai": openai_api_key_text,
        "groq": groq_api_key_text
    }

    # Dropdown to select models for comparison
    selected_models = []
    for group, models in model_options.items():
        st.subheader(group)
        selected_models += st.multiselect(f"Select {group} models", models, key=f"{group}_models")

    selected_models = selected_models[:5]  # Limit to 5 models

    # Text input for user prompt
    user_prompt = st.text_area("Enter your prompt here")

    # System prompt and expected format
    system_prompt = "output only JSON object"
    expected_format = "{}"

    # Button to compare outputs
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
                    output = call_model_api(model, system_prompt, user_prompt, expected_format, api_keys)
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
