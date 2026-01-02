import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pyngrok import ngrok



#public_url = ngrok.connect(8501) #put your port provided by streamlit
#st.write(f"Public URL: {public_url}")

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="ML Chatbot", layout="wide")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        SystemMessage(content="You are an expert on machine learning. You will give precise answers on every query asked by the user. Make sure to use the context given to you from a book on machine learning to support your answer.")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []







# ---- Setup ----




@st.cache_resource
def setup_vector_store():
    persist_dir = os.path.join(os.path.dirname(__file__), "myChromaDB")
    return Chroma(
        
        collection_name="chunks",
        persist_directory=persist_dir
    )



@st.cache_resource
def setup_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=google_api_key
    )




# Load resources
vector_store = setup_vector_store()
llm = setup_model()








# ---- UI ----
st.title("Hands-on Machine Learning Chatbot")



# chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_input = st.chat_input("Enter your query about machine learning...")






if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # retrieve from ChromaDB
    dbresult = vector_store.similarity_search(query=user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in dbresult])
    
    messages_with_context = [
        st.session_state.conversation_history[0],
        HumanMessage(content=f"User query: {user_input}\n\nRelevant context from book:\n{context}")
    ]
    

    with st.spinner("Thinking..."):
        try:
            answer = llm.invoke(messages_with_context)
            assistant_response = answer.content
        except Exception as e:
            assistant_response = f"Error: {str(e)}"
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response
    })
    
    st.rerun()

