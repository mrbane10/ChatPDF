import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv('groq_key')

st.title("Chat with PDF 😤 \n"
         "**Kaisar Imtiyaz**")

# Initializing the language model
llm = ChatGroq(groq_api_key=groq_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Initializing the conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""
def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()

    st.session_state.loader = PyPDFLoader(file_path)
    st.session_state.docs = st.session_state.loader.load()

    if not st.session_state.docs:
        st.error("No documents were loaded. Please check the document source.")
        return

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    if not st.session_state.final_documents:
        st.error("No document chunks were created. Please check the splitting logic.")
        return

    embeddings = st.session_state.embeddings.embed_documents(
        [doc.page_content for doc in st.session_state.final_documents]
    )

    if not embeddings:
        st.error("Embeddings could not be generated. Please check the embedding model and documents.")
        return

    # Reinitialize the Chroma index with the new document vectors
    st.session_state.vectors = Chroma.from_documents(
        documents=st.session_state.final_documents,
        embedding=st.session_state.embeddings
    )

    st.success("Chroma index has been updated with the new document vectors.")

# Function to handle user input
def handle_user_input():
    user_input = st.session_state.user_input
    if user_input.lower() == "clear":
        st.session_state.conversation_history = []
        st.success("Conversation history cleared.")
    elif user_input:
        # Add the user's question to the conversation history
        st.session_state.conversation_history.append(f"You: {user_input}")

        # Processing the user's question
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_input})

        # Add the AI's response to the conversation history
        st.session_state.conversation_history.append(f"AI: {response['answer']}")

    st.session_state.user_input = ""


uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Process Document"):
        os.makedirs("./pdfs", exist_ok=True)
        file_path = os.path.join("./pdfs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        vector_embedding(file_path)
        st.write(f"Vector Store DB Is Ready for file: {uploaded_file.name}")

st.subheader("Chat with the PDF")

st.text_input("You: ", value=st.session_state.user_input, key="user_input", on_change=handle_user_input)
for message in reversed(st.session_state.conversation_history):
    st.write(message)

