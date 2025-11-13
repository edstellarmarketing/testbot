import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

# Set page config
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖", layout="centered")

# Initialize the LLM
@st.cache_resource
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set it in Streamlit Cloud secrets.")
        st.stop()
    return ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", api_key=api_key)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.title("🤖 Simple Chatbot")
st.caption("Powered by LangChain and OpenAI")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = get_llm()
            
            # Convert chat history to LangChain format
            langchain_messages = []
            for msg in st.session_state.messages[:-1]:  # Exclude the last user message
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Add the current user message
            langchain_messages.append(HumanMessage(content=prompt))
            
            # Get response
            response = llm.invoke(langchain_messages)
            st.markdown(response.content)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.content})

# Add a clear chat button in the sidebar
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
