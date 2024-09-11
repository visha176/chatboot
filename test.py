from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import logging
import requests
from zeep import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the SOAP service URL
wsdl_url = 'http://retailapp.eastgateindustries.com:8082/RetailAppServices/RetailAppClass.svc?wsdl'

def call_ra_report_class_sellthru(sbs, logged_user_id):
    """
    Call the RA_REPORT_CLASS_SELLTHRU method with the given parameters using SOAP.
    """
    try:
        client = Client(wsdl_url)
        response = client.service.RA_REPORT_CLASS_SELLTHRU(sbs, logged_user_id)
        return response
    except Exception as e:
        logger.error(f"Error calling SOAP service: {e}")
        return None

def format_data_for_response(data):
    """
    Convert data into a format suitable for response generation.
    """
    # Adjust this according to the expected format for the prompt
    return {
        'formatted_data': data
    }

def generate_natural_language_response(user_query: str, data: dict):
    """
    Generate a natural language response based on the user's query and provided data.
    """
    template = """
    You are a data analyst. Based on the data provided, answer the user's query in a natural language.
    <DATA>{formatted_data}</DATA>
    User question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    # Format data properly for RunnablePassthrough
    formatted_data = format_data_for_response(data)['formatted_data']
    
    # Create a Runnable that can be passed to RunnablePassthrough
    def format_data_for_runnables(data):
        return {
            'formatted_data': formatted_data,
            'question': user_query
        }
    
    chain = (
        RunnablePassthrough(assign=format_data_for_runnables)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "formatted_data": formatted_data
    })
# Initialize Streamlit app
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a data assistant. Ask me anything about your data."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with Retail Service", page_icon=":speech_balloon:")

st.title("Chat with Retail Service")

# Sidebar for service settings
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using SOAP service. Connect to the service and start chatting.")
    
    st.text_input("SBS", value="2", key="SBS")
    st.text_input("Logged User ID", value="141", key="LoggedUserID")
    
    if st.button("Fetch Data"):
        with st.spinner("Fetching data from service..."):
            sbs = st.session_state["SBS"]
            logged_user_id = st.session_state["LoggedUserID"]
            data = call_ra_report_class_sellthru(sbs, logged_user_id)
            if data:
                st.session_state.data = data
                st.success("Data fetched successfully!")
            else:
                st.error("Failed to fetch data. Check the logs for details.")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Handle user input and display responses
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        if "data" in st.session_state:
            response = generate_natural_language_response(user_query, st.session_state.data)
        else:
            response = "No data available. Please fetch data from the service first."
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
