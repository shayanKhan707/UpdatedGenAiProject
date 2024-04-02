import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import tempfile

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def welcome_message(customer_name):
    return f"Hi, {customer_name}! Welcome to our Customer Support ChatBot for payments and banking! 🏦💳\n\nHow can we assist you today? Whether you have questions about transactions, account balances, or any other banking inquiries, feel free to ask. We're here to help you with any payment-related or banking-related concerns you may have.\n\nSimply type your query in the chatbox below, and we'll provide you with the assistance you need.\n\nLet's get started! 🚀"

def conversational_chat(query, customer_id):
  # Search the CSV data for matching keywords from the user query
    
    prompt_template = f"{query} Customer ID ={customer_id}) {query}.Give me answer in English."
    filled_prompt = prompt_template.format(user_query=query)
    result = chain({"question": filled_prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]
    


def fallback_response():
    # Define your fallback response here
    return "I'm sorry, I didn't understand that. Could you please rephrase your question?"

# Hardcoded file path for the CSV file
csv_file_path = "bankingpaymentsFAQ.csv"

loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")
data = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectors = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,
                                                                       convert_system_message_to_human=True),
                                              retriever=vectors.as_retriever())
st.header("Customer Support ChatBot")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your CSV data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    # Add a button to close the chatbot session and redirect

    if submit_button and user_input:
        # Retrieve customerId from query parameters
        query_params = st.query_params
        customer_id = query_params.get("customerId", "")

        if "thank you" in user_input.lower():
            response = "You're welcome!"
        elif "how many records" in user_input.lower():
            response = f"There are {len(data)} records in the file."
        else:
            response = conversational_chat(user_input, customer_id)
            if response == "0":
                response = fallback_response()

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(response)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# Retrieve customerId from query parameters
query_params = st.query_params
customer_id = query_params.get("customerId", "")
print(customer_id)

# Define a dictionary to map customer IDs to customer names
customer_names = {
    "C3452": "Shayan",
    "C3454": "Abhishek",
    "C3456": "Sharan"
}

# Check if the customerId exists in the dictionary
if customer_id in customer_names:
    # If exists, get the customer name
    customer_name = customer_names[customer_id]
    # Display welcome message with the customer name
    st.write(welcome_message(customer_name))
else:
    # If customerId doesn't exist, display a default welcome message
    st.write("Welcome! Please login to access the chatbot.")
