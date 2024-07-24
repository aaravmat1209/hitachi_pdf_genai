import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.summarizers import Summarizer
from langchain.text_preprocessing import TextPreprocessor
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
from io import StringIO

# Function to categorize text chunks
def categorize_text_chunks(text_chunks):
    categorized_chunks = {'Introduction': [], 'Methods': [], 'Results': [], 'Discussion': []}
    for chunk in text_chunks:
        # Simple categorization based on keywords. This can be enhanced with more sophisticated methods.
        if 'introduction' in chunk.lower():
            categorized_chunks['Introduction'].append(chunk)
        elif 'method' in chunk.lower():
            categorized_chunks['Methods'].append(chunk)
        elif 'result' in chunk.lower():
            categorized_chunks['Results'].append(chunk)
        elif 'discussion' in chunk.lower():
            categorized_chunks['Discussion'].append(chunk)
        else:
            categorized_chunks.setdefault('Other', []).append(chunk)
    return categorized_chunks

# Function to handle categorized memory
def get_categorized_vectorstore(categorized_chunks):
    vectorstores = {}
    embeddings = OpenAIEmbeddings()
    for category, chunks in categorized_chunks.items():
        if chunks:
            vectorstores[category] = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstores

# Function to handle user input with categorized memory
def handle_user_input_with_categories(user_question, vectorstores):
    responses = {}
    for category, vectorstore in vectorstores.items():
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        response = conversation_chain({'question': user_question})
        responses[category] = response['chat_history']
    return responses

def main():
    st.set_page_config(page_title="Advanced PDF Chatbot with Categorized Memory", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Advanced Chat with Multiple PDFs and Categorized Memory :books:")

    with st.sidebar:
        st.subheader("Upload PDFs for Analysis and Categorization")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Extract text from PDFs
                    raw_text = ""
                    for pdf in pdf_docs:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            raw_text += page.extract_text()

                    # Text preprocessing options
                    remove_stopwords = st.checkbox("Remove stop words", value=False)
                    lemmatize = st.checkbox("Lemmatize text", value=False)
                    processed_text = TextPreprocessor(remove_stopwords=remove_stopwords, lemmatize=lemmatize).process(raw_text)

                    # Customizable text chunking
                    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000)
                    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200)
                    text_chunks = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len
                    ).split_text(processed_text)

                    # Categorize text chunks
                    categorized_chunks = categorize_text_chunks(text_chunks)

                    # Create vector store for each category
                    vectorstores = get_categorized_vectorstore(categorized_chunks)
                    st.session_state.vectorstores = vectorstores

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.vectorstores:
        responses = handle_user_input_with_categories(user_question, st.session_state.vectorstores)
        for category, chat_history in responses.items():
            st.write(f"**Responses from Category: {category}**")
            for i, message in enumerate(chat_history):
                if i % 2 == 0:
                    st.write(f"**User:** {message['content']}")
                else:
                    st.write(f"**Bot:** {message['content']}")

if __name__ == '__main__':
    main()
