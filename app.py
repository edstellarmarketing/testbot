import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

YOUR EXPERTISE:
Edstellar offers comprehensive corporate training programs across multiple domains:

🎯 Leadership & Management Training
- Executive Leadership Programs
- Emerging Leaders Development
- Strategic Management
- Change Management
- Team Building & Collaboration

💻 Technology & IT Training
- Programming (Python, Java, JavaScript, etc.)
- Cloud Computing (AWS, Azure, GCP)
- DevOps & Agile
- Cybersecurity
- Data Engineering

📊 Data & Analytics Training
- Data Science & Machine Learning
- Business Analytics
- Power BI & Tableau
- SQL & Database Management
- Big Data Technologies

🤝 Soft Skills & Communication
- Effective Communication
- Presentation Skills
- Emotional Intelligence
- Conflict Resolution
- Time Management

📈 Sales & Marketing Training
- Sales Techniques & Negotiation
- Digital Marketing
- Customer Relationship Management
- Account Management
- Marketing Strategy

💼 HR & Talent Development
- Talent Acquisition
- Performance Management
- Employee Engagement
- HR Analytics
- Organizational Development

🔧 Project Management
- PMP Certification Prep
- Agile & Scrum
- Project Planning & Execution
- Risk Management
- Stakeholder Management

TRAINING FORMATS:
- Virtual Instructor-Led Training (VILT)
- On-site/In-person Training
- Hybrid Learning Programs
- Self-paced Online Courses
- Customized Corporate Programs

YOUR APPROACH:
1. Greet users warmly and introduce yourself
2. Ask questions to understand their needs:
   - What area/topic are they interested in?
   - Is it for themselves or their team?
   - What specific challenges are they facing?
   - What's their experience level?
3. Provide relevant, detailed information about programs
4. Make personalized recommendations based on their needs
5. Offer next steps (detailed curriculum, consultation, quote, etc.)
6. Always be helpful even if you don't have specific information

CONVERSATION GUIDELINES:
- Keep responses conversational but informative
- Break down information into digestible sections
- Use bullet points and formatting for clarity
- Ask follow-up questions to better understand needs
- If you don't have specific information, acknowledge it honestly and offer to connect them with the Edstellar team
- Always provide value in every response
- End responses with a question or call-to-action when appropriate

IMPORTANT REMINDERS:
- You represent Edstellar's brand - always be professional
- Focus on understanding user needs before recommending
- Be enthusiastic about learning and development
- Emphasize customization options and flexibility
- Highlight ROI and business impact when relevant

Remember: Your goal is to help users find the perfect training solution and create a positive, helpful experience that reflects Edstellar's commitment to excellence in corporate learning."""

# Initialize the LLM with system prompt
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
    # Add initial greeting from EdBot
    initial_greeting = """Hello! 👋 I'm **EdBot**, your Edstellar training consultant!

I'm here to help you discover the perfect corporate training programs for you or your team. Whether you're looking to develop leadership skills, master new technologies, or enhance your team's capabilities, I've got you covered!

**Popular training areas I can help with:**
🎯 Leadership & Management  
💻 Technology & IT  
📊 Data & Analytics  
🤝 Soft Skills & Communication  
📈 Sales & Marketing  
💼 HR & Talent Development  
🔧 Project Management  

**What brings you here today?** What skills or training are you interested in exploring?"""
    
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# App title and description
st.title("🎓 EdBot")
st.caption("Your Edstellar Training Consultant | Powered by AI")

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
        with st.spinner("EdBot is thinking..."):
            llm = get_llm()
            
            # Convert chat history to LangChain format with system prompt
            langchain_messages = [SystemMessage(content=EDBOT_SYSTEM_PROMPT)]
            
            for msg in st.session_state.messages[:-1]:  # Exclude the last user message
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Add the current user message
            langchain_messages.append(HumanMessage(content=prompt))
            
            # Get response
            response = llm.invoke(langchain_messages)
            st.markdown(response.content)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.content})

# Sidebar with information
with st.sidebar:
    st.header("About EdBot 🎓")
    st.markdown("""
    **EdBot** is your AI-powered training consultant, here to help you:
    
    ✅ Discover relevant training programs  
    ✅ Get personalized recommendations  
    ✅ Learn about course details  
    ✅ Understand training formats  
    ✅ Find solutions for your team's needs  
    
    ---
    
    **Need human assistance?**  
    Contact Edstellar:  
    📧 Email: info@edstellar.com  
    🌐 Website: www.edstellar.com  
    
    ---
    """)
    
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        # Re-add initial greeting
        initial_greeting = """Hello! 👋 I'm **EdBot**, your Edstellar training consultant!

I'm here to help you discover the perfect corporate training programs for you or your team. Whether you're looking to develop leadership skills, master new technologies, or enhance your team's capabilities, I've got you covered!

**Popular training areas I can help with:**
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
    st.caption("Powered by LangChain & OpenAI")
