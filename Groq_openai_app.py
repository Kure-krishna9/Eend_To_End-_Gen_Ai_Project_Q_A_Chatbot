from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Optional LangSmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with OpenAI"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user questions."),
    ("user", "Question: {question}")
])

# Response Generator
def generate_response(question, api_key, model_name, temperature, max_tokens):
    if model_name in ["gpt-4", "gpt-4-turbo", "gpt-4o"]:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
    else:
        # Assume Groq models like "Mixtral-8x7b-32768", "Gemma-7b-it", "LLaMA3-8b"
        llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key  # This will be GROQ_API_KEY
        )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

# Streamlit UI
st.title("🧠 Q&A ChatBot with OpenAI & Groq")
st.sidebar.title("🔧 Settings")

model_name = st.sidebar.selectbox(
    "Select Model:",
    ["gpt-4o", "gpt-4-turbo", "gpt-4", "Mixtral-8x7b-32768", "Gemma2-9b-It", "LLaMA3-8b"]
)

# Set correct placeholder depending on model
if model_name.startswith("gpt"):
    api_label = "Enter your OpenAI API Key:"
    env_var_name = "OPENAI_API_KEY"
else:
    api_label = "Enter your Groq API Key:"
    env_var_name = "GROQ_API_KEY"

api_key = st.sidebar.text_input(api_label, type="password", value=os.getenv(env_var_name))

temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens:", 50, 300, 150)

st.write("💬 Go ahead and ask any question.")
user_input = st.text_input("You:")

if user_input:
    if api_key:
        try:
            response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
            st.write(f"🤖: {response}")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter your API key.")
else:
    st.info("ℹ️ Waiting for your question.")
