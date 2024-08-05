import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

pc = Pinecone()
index_name = "youtube-index"
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone = PineconeVectorStore(index_name=index_name, embedding=embeddings)

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

chain = (
        {"context": pinecone.as_retriever(), "question": lambda x: x}
        | prompt
        | model
        | parser
)

# Streamlit app
st.title("YouTube Video Q&A Chatbot")
st.header("Deep Learning (INFS-778-DT1) - RAG Project")
st.caption("Muhammad Abdur Rahman")

user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer = chain.invoke(user_question)

        st.write("Answer:")
        st.write(answer)

st.write("---")

st.write("This chatbot answers questions about the following YouTube video:")
st.video('https://www.youtube.com/watch?v=cdiD-9MMpb0')
