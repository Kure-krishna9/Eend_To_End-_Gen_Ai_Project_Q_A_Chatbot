
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_openai import ChatOpenAI   # Open ai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set Langsmith environment variables (if used)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with OpenAI"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # Optional: if you're using LangSmith

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user questions."),
    ("user", "Question: {question}")
])

# Response Generator
def generate_response(question, api_key, model_name, temperature, max_tokens):
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key  # ✅ pass it here
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

# Streamlit UI
st.title("Q & A ChatBot Application")
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
model_name = st.sidebar.selectbox("Select OpenAI Model:", ["gpt-4o", "gpt-4-turbo", "gpt-4","Gemma2-9b-It"])
temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens:", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any question.")
user_input = st.text_input("You:")

if user_input:
    if api_key:
        try:
            response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
            st.write(f"🤖: {response}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter your OpenAI API key.")
else:
    st.info("Please provide a query.")
