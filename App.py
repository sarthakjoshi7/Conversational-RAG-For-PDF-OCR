# Conversational RAG For PDF + OCR Project
import os
import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from OCR_Utils import extract_text_from_pdf
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"]=st.secrets["HF_TOKEN"]
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Set Up Streamlit
st.title("Conversational RAG With Chat History For PDF + OCR")
st.write("Upload PDF's and chat with their content")
# Input The API Key
api_key=st.text_input("Enter the API key:",type="password")
# Check If API Key Is Provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")
    # Chat Interface
    session_id=st.text_input("Session ID",value="Default_Session")
    # Statefully Manage Chat History
    if 'store' not in st.session_state:
        st.session_state.store={}
    if "retriever" not in st.session_state:
        st.session_state.retriever=None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed=False
    uploaded_files=st.file_uploader("Choose A PDF File",type="pdf",accept_multiple_files=True)
    # Detect New PDF Upload
    current_files=sorted([file.name for file in uploaded_files]) if uploaded_files else []
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files=[]
    if current_files!=st.session_state.last_uploaded_files:
        st.session_state.retriever=None
        st.session_state.pdf_processed=False
        st.session_state.last_uploaded_files=current_files
    # Process Uploaded PDF's
    if uploaded_files and not st.session_state.pdf_processed:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./{uploaded_file.name}"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
            # Extract Text (Normal PDF Or OCR Automatically)
            with st.spinner(f"Processing {uploaded_file.name}..."):
                all_text=extract_text_from_pdf(temppdf)
            if not all_text.strip():
                st.error(f"No text found in {uploaded_file.name}")
                if os.path.exists(temppdf):
                    os.remove(temppdf)
                continue
            docs=[Document(page_content=all_text,metadata={"source":uploaded_file.name})]
            documents.extend(docs)
        # Split And Create Embeddings For The Documents
        if not documents:
            st.error("No text could be extracted from uploaded PDF.")
            st.stop()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory="./chroma_db")
        retriever=vectorstore.as_retriever()
        st.session_state.retriever=retriever
        st.session_state.pdf_processed=True
    if st.session_state.retriever is None:
        st.stop()
    retriever=st.session_state.retriever
    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "form a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reform it if needed and otherwise return it as is."
    )
    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    # Answer Question
    system_prompt=(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    user_input=st.text_input("Your Question:")
    if user_input:
        session_history=get_session_history(session_id)
        response=conversational_rag_chain.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}
            }, # Constructs A Key "Abc1" In `Store`
        )
        st.write(st.session_state.store)
        st.write("Assistant:",response['answer'])
        st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter the API Key.")
