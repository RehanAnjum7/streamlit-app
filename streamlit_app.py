import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

OPENAI_API_KEY = "sk-proj-wKSMQWB2nfZJ33gIbbjbT3BlbkFJQeXXwU8rjp5pPGK4DpVB"
os.environ['PINECONE_API_KEY'] = "672ff728-f3b7-45a6-a3ed-f9d54a410988"

# Set up Pinecone
pc = Pinecone()
index_name = "youtube-index"
index = pc.Index(index_name)

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Set up ChatGPT model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Set up prompt template
template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

# Set up chain
chain = (
        {"context": pinecone.as_retriever(), "question": lambda x: x}
        | prompt
        | model
        | parser
)

# Streamlit app
st.title("YouTube Video Q&A Chatbot")

st.write("Ask questions about the YouTube video:")

# Input for user question
user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Thinking..."):
        # Get the answer
        answer = chain.invoke(user_question)

        # Display the answer
        st.write("Answer:")
        st.write(answer)

st.write("---")
st.write("This chatbot answers questions based on the transcript of a specific YouTube video.")
