# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import OllamaLLM
# import streamlit as st
# import os
# from dotenv import load_dotenv
# load_dotenv()


# os.environ["Langchain_api_key"]=os.getenv("Langchain_api_key")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["Langchain_Project"]="Simple Q&A Chatbot OLLAMA"

# # Promt templatet

# prompt=ChatPromptTemplate.from_messages([
#     ("system","you are the helpfull assitent.give anser of every question "),
#     ("user","Question:{question}")
# ])


# # Response Generator
# def generate_response(question, api_key, model_name, temperature, max_tokens):
#     llm = OllamaLLM(
#         model=model_name
#         # ✅ pass it here
#     )
#     output_parser = StrOutputParser()
#     chain = prompt | llm | output_parser
#     return chain.invoke({'question': question})


# app_ollama_chatbot.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional: LangSmith (skip if not using)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot OLLAMA"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("Langchain_api_key")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer user questions."),
    ("user", "Question: {question}")
])

# Response Generator
def generate_response(question, model_name, temperature, max_tokens):
    llm = OllamaLLM(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

# Streamlit UI
st.set_page_config(page_title="Ollama Chatbot")
st.title("🧠 Local ChatBot with Ollama (LangChain)")
st.sidebar.title("Settings")

model_name = st.sidebar.selectbox("Select Ollama Model", ["llama3", "mistral", "gemma", "codellama", "phi3","gemma:2b"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 256)

st.write("💬 Ask me anything:")
user_input = st.text_input("You:")

if user_input:
    try:
        response = generate_response(user_input, model_name, temperature, max_tokens)
        st.write(f"🤖: {response}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("ℹ️ Please enter a question to start.")
