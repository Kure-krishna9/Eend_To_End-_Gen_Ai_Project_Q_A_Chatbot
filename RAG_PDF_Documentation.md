# 📘 Full Documentation for Conversational RAG with PDF (LangChain + Groq + Streamlit)

## 🔹 1. Imports & Their Purpose

``` python
import os
import streamlit as st
from dotenv import load_dotenv
```

-   **os** → For environment variables & file operations.\
-   **streamlit** → Builds the web UI.\
-   **dotenv.load_dotenv** → Loads API keys & secrets from `.env` file.

``` python
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
```

-   **ConversationBufferMemory** → Stores chat history across turns.\
-   **Chroma** → Vector database for storing/retrieving embeddings.\
-   **PyMuPDFLoader** → Extracts text from PDFs.\
-   **HuggingFaceEmbeddings** → Converts text chunks into numerical
    vectors.\
-   **RecursiveCharacterTextSplitter** → Splits large text into chunks
    for embedding.\
-   **ConversationalRetrievalChain** → Core RAG pipeline (retrieval +
    LLM + memory).\
-   **ChatGroq** → Wrapper for Groq-hosted LLMs.

------------------------------------------------------------------------

## 🔹 2. Environment Variables Setup

``` python
load_dotenv()
hf_token = os.getenv("HF_token")
os.environ["HK_Token"] = hf_token
```

-   Loads **HuggingFace token** from `.env`.\
-   Sets it as `HK_Token` for embedding models.

------------------------------------------------------------------------

## 🔹 3. Embedding Model Initialization

``` python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

-   Uses a **sentence transformer** to convert text into vector
    embeddings.\
-   These embeddings are stored in Chroma DB for semantic search.

------------------------------------------------------------------------

## 🔹 4. Streamlit User Interface

``` python
st.title("Conversational RAG with PDF")
st.write("Upload PDFs and chat with their content")
```

-   Sets **title** and **description** in Streamlit app.

``` python
api_key = st.text_input("Enter your Groq API key:", type="password")
```

-   Prompts user to enter **Groq API key** securely.

------------------------------------------------------------------------

## 🔹 5. LLM Setup (Groq)

``` python
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
```

-   Initializes **Groq-powered LLM** (`gemma2-9b-it`).\
-   Requires valid **Groq API key**.

------------------------------------------------------------------------

## 🔹 6. Session Management

``` python
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}
```

-   Uses `session_state` to persist memory across chat sessions.\
-   Allows multiple sessions via custom `session_id`.

------------------------------------------------------------------------

## 🔹 7. PDF Upload & Processing

``` python
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
```

-   Lets user upload a PDF.\
-   Saves it temporarily as `temp.pdf`.

``` python
loader = PyMuPDFLoader("temp.pdf")
documents = loader.load()
```

-   Loads PDF into `documents` with **metadata + text**.

``` python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)
```

-   Splits documents into chunks of **1000 characters** with **100
    overlap**.\
-   Ensures smooth retrieval.

------------------------------------------------------------------------

## 🔹 8. Vector Store Setup

``` python
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

-   Stores chunk embeddings in **Chroma DB**.\
-   Creates a retriever that fetches relevant chunks during queries.

------------------------------------------------------------------------

## 🔹 9. Memory Setup

``` python
if session_id not in st.session_state.store:
    st.session_state.store[session_id] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
memory = st.session_state.store[session_id]
```

-   Creates **conversation memory** for each session.\
-   Stores user + assistant messages for contextual answers.

------------------------------------------------------------------------

## 🔹 10. Conversational RAG Chain

``` python
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    output_key="answer"
)
```

-   Combines:
    -   **LLM (ChatGroq)**\
    -   **Retriever (Chroma)**\
    -   **Memory (ConversationBufferMemory)**\
-   `output_key="answer"` ensures proper output format.

------------------------------------------------------------------------

## 🔹 11. Chat Interaction

``` python
user_input = st.text_input("Your question:")
if user_input:
    response = qa_chain.invoke({"question": user_input})
```

-   Takes user input and queries the **RAG chain**.

``` python
st.markdown("### Assistant:")
st.write(response["answer"])
```

-   Displays assistant's answer.

``` python
if "source_documents" in response:
    st.write("### Source Documents:")
    for doc in response["source_documents"]:
        st.write(doc.page_content[:300] + "...")
```

-   Optionally shows **retrieved source documents**.

``` python
st.markdown("### Chat History")
for msg in memory.chat_memory.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")
```

-   Prints **full chat history** (user + assistant messages).

------------------------------------------------------------------------

## 🔹 12. Requirements

``` bash
pip install streamlit langchain langchain-groq langchain-chroma             langchain-community langchain-text-splitters             python-dotenv pymupdf
```

------------------------------------------------------------------------

## 🔹 13. Run App

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🔹 14. Example Run

-   Upload **Research_Paper.pdf**.\
-   Ask: *"Summarize section 3."*\
-   Assistant responds with summary from that section.\
-   Chat history keeps track of previous Q&A.
