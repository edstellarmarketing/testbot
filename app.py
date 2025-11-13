import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="EdBot - Edstellar Training Assistant", page_icon="🎓", layout="centered")

# EdBot System Prompt
EDBOT_SYSTEM_PROMPT = """You are EdBot, Edstellar's friendly and knowledgeable training consultant assistant. 

YOUR ROLE:
You help users discover and learn about corporate training programs offered by Edstellar. You are professional yet warm, patient, and genuinely interested in helping users find the right training solutions for their needs.

YOUR PERSONALITY:
- Professional but approachable and conversational
- Enthusiastic about learning and development
- Patient and thorough in explanations
- Proactive in asking clarifying questions
- Solution-oriented and consultative
- Use emojis occasionally to be friendly (🎯, 📚, ✅, 💼, 🎓, etc.)

IMPORTANT INSTRUCTIONS:
- Use the CONTEXT provided below to answer questions accurately
- When mentioning specific courses, ALWAYS include the course page link
- Cite your sources by referencing course names and providing links
- If the answer is in the context, use that information
- If you're not sure, acknowledge it and offer to connect them with the Edstellar team
- Always provide detailed, helpful responses based on the context
- End responses with engaging questions or calls-to-action

CONVERSATION GUIDELINES:
- Keep responses conversational but informative
- Break down information into digestible sections
- Use bullet points and formatting for clarity
- Ask follow-up questions to better understand needs
- Always provide value in every response
- Emphasize customization options and flexibility
- Highlight ROI and business impact when relevant

Remember: Your goal is to help users find the perfect training solution using the accurate information from Edstellar's course catalog."""

# Load and process course data
@st.cache_resource
def load_course_data():
    """Load course data from CSV and create vector store"""
    try:
        # Load courses
        courses_df = pd.read_csv('edstellar_courses.csv')
        general_df = pd.read_csv('edstellar_general_info.csv')
        
        documents = []
        
        # Process courses
        for _, row in courses_df.iterrows():
            # Create comprehensive text for each course
            course_text = f"""
Course Name: {row['course_name']}
Category: {row['category']}
Duration: {row['duration']} ({row['duration_hours']} hours)
Format: {row['format']}
Target Audience: {row['target_audience']}
Group Size: {row['group_size']}
Prerequisites: {row['prerequisites']}

Description: {row['description']}

Key Topics: {row['key_topics']}

Learning Outcomes: {row['learning_outcomes']}

Price Range: {row['price_range']}

Customization: {row['customization']}

Course Page: {row['course_page']}
Additional Resources: {row['additional_links']}
"""
            
            documents.append(Document(
                page_content=course_text,
                metadata={
                    'course_name': row['course_name'],
                    'category': row['category'],
                    'course_page': row['course_page'],
                    'type': 'course'
                }
            ))
        
        # Process general information
        for _, row in general_df.iterrows():
            if pd.notna(row['description']) and row['description']:
                info_text = f"""
{row['title']}: {row['description']}
URL: {row['url']}
Type: {row['info_type']}
"""
                documents.append(Document(
                    page_content=info_text,
                    metadata={
                        'title': row['title'],
                        'url': row['url'],
                        'type': row['info_type']
                    }
                ))
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        return vectorstore, courses_df, general_df
        
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return None, None, None

# Initialize the LLM
@st.cache_resource
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set it in Streamlit Cloud secrets.")
        st.stop()
    return ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", api_key=api_key)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting
    initial_greeting = """Hello! 👋 I'm **EdBot**, your Edstellar training consultant!

I'm here to help you discover the perfect corporate training programs for you or your team. I have detailed knowledge about all our training courses, including content, pricing, and customization options.

**Popular training areas:**
🎯 Leadership & Management  
💻 Technology & IT  
📊 Data & Analytics  
🤝 Soft Skills & Communication  
📈 Sales & Marketing  
💼 HR & Talent Development  
🔧 Project Management  

**What brings you here today?** What skills or training are you interested in exploring?"""
    
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# App title
st.title("🎓 EdBot")
st.caption("Your Edstellar Training Consultant | Powered by AI & RAG")

# Load course data
with st.spinner("Loading Edstellar course catalog..."):
    vectorstore, courses_df, general_df = load_course_data()

if vectorstore is None:
    st.error("Failed to load course data. Please check CSV files are present.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("EdBot is searching the course catalog..."):
            llm = get_llm()
            
            # Retrieve relevant context from vector store
            relevant_docs = vectorstore.similarity_search(prompt, k=3)
            
            # Build context from retrieved documents
            context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create enhanced system prompt with context
            enhanced_prompt = f"""{EDBOT_SYSTEM_PROMPT}

CONTEXT FROM EDSTELLAR COURSE CATALOG:
{context}

Use the above context to answer the user's question accurately. Always include relevant course links when discussing specific programs."""
            
            # Build message history
            langchain_messages = [SystemMessage(content=enhanced_prompt)]
            
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            langchain_messages.append(HumanMessage(content=prompt))
            
            # Get response
            response = llm.invoke(langchain_messages)
            st.markdown(response.content)
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.content})

# Sidebar
with st.sidebar:
    st.header("About EdBot 🎓")
    st.markdown("""
    **EdBot** uses RAG (Retrieval Augmented Generation) to provide accurate information about:
    
    ✅ Course details & curriculum  
    ✅ Pricing & customization  
    ✅ Target audience & prerequisites  
    ✅ Training formats & schedules  
    ✅ Learning outcomes & benefits  
    
    ---
    """)
    
    # Show loaded courses count
    if courses_df is not None:
        st.metric("Courses Loaded", len(courses_df))
    
    st.markdown("---")
    
    st.markdown("""
    **Need human assistance?**  
    📧 training@edstellar.com  
    🌐 [www.edstellar.com](https://www.edstellar.com)  
    
    ---
    """)
    
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        initial_greeting = """Hello! 👋 I'm **EdBot**, your Edstellar training consultant!

I'm here to help you discover the perfect corporate training programs for you or your team. I have detailed knowledge about all our training courses, including content, pricing, and customization options.

**Popular training areas:**
🎯 Leadership & Management  
💻 Technology & IT  
📊 Data & Analytics  
🤝 Soft Skills & Communication  
📈 Sales & Marketing  
💼 HR & Talent Development  
🔧 Project Management  

**What brings you here today?** What skills or training are you interested in exploring?"""
        
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by LangChain, OpenAI & FAISS")
    
    # Debug info (optional - can be removed)
    with st.expander("🔧 Debug Info"):
        st.write(f"Vector store created: {'✅' if vectorstore else '❌'}")
        st.write(f"Total documents: {len(vectorstore.docstore._dict) if vectorstore else 0}")
