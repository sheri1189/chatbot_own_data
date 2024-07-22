import streamlit as st
import os
import docx
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import mysql.connector
from mysql.connector import Error
openapi_key = st.secrets['OPENAI_API_KEY']


def main():
    st.set_page_config(page_title="ibexstack Chatbot (own data)")
    st.markdown("<div style='display:flex'><h4 style='font-size:30px'>ibexstack Chatbot</h4><h5 style='font-size: 15px;font-weight:bold;margin-left: -22px;margin-top: 24px;color:#ff4b4b'>(own data)</h5></div>",
                unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    hide_streamlit_style = """
    <style>
    button[title="View fullscreen"] {display: none;}
    .css-1r0j92e {
        display: none;
    }
    </style>"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<img src='http://dev.ibexstack.com/stagging/assets/images/creative/logo_white.png' alt='Img not found' style='width: 182px;margin-top: -117px;'/>",unsafe_allow_html=True)
        # st.image('http://localhost:8501/media/455914e9848e16e37e3ca59caee1289961d39b820b5aae9bde5a55ab.png', width=250)
        hide_fullscreen_icon = """<style>button[title="View fullscreen"] {display: none;}</style>"""
        st.markdown(hide_fullscreen_icon, unsafe_allow_html=True)
        uploaded_files = st.file_uploader("", type=[
            'pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = openapi_key
        if uploaded_files:
            upload_process = st.button(
                "Upload Files", type="primary", use_container_width=True)
        else:
            upload_process = ""
    if upload_process:
        if not openai_api_key:
            st.error("Please Enter OpenAPI Key First")
            st.stop()
        if not uploaded_files:
            st.error("Please Upload the One File Atleast")
            st.stop()
        with st.spinner("Processing..."):
            files_text = get_files_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(
                vectorstore, openai_api_key)
            st.session_state.processComplete = True
    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)


def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1]
            if file_extension == ".pdf":
                text += get_pdf_text(uploaded_file)
            elif file_extension == ".docx":
                text += get_docx_text(uploaded_file)
    return text


def get_text_chunks(text):
    chunks = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    ).split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_docx_text(docx_file):
    doc = docx.Document(docx_file)
    allText = []
    for para in doc.paragraphs:
        allText.append(para.text)
    text = ' '.join(allText)
    return text


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     model_name='gpt-3.5-turbo', temperature=0, max_tokens=40)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
                user_question = messages.content
            else:
                message(messages.content, key=str(i))
                user_answer = messages.content
                insert_into_db(user_question, user_answer)


def insert_into_db(user_question, user_answer):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='python_chatbot'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = """
            INSERT INTO chatbot (user_question, user_answer)
            VALUES (%s, %s)
            """
            cursor.execute(insert_query, (user_question, user_answer))
            connection.commit()
    except Error as e:
        st.write(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == "__main__":
    main()
