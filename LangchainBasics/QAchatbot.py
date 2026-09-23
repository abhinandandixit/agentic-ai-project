import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

# Page Config
st.set_page_config(page_title="Simple Langchain Chatbot With Groq", page_icon="🚀")

# Title
st.title("🚀 Simple Langchain Chat With Groq")
st.markdown("Learn langchain basics with Groq's Ultra Fast inference!")

with st.sidebar:
    st.header("Settings")

    # API Key
    api_key=st.text_input("GROQ API Key", type="password",help="Get free API Key at console.groq.com")

    # Model Selection
    model_name=st.selectbox(
        "Model",
        ["openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"],
        index=0
    )

    # Clear Button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages=[]


# Initialize LLM
@st.cache_resource
def get_chain(api_key,model_name):
    if not api_key:
        return None


    # Initialize the GROQ Model
    llm = ChatGroq(groq_api_key=api_key,
                   model_name=model_name,
                   temperature=0.7,
                   streaming=True)

    # Create Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assisstant powered by Groq. Answer questions clearly and concisely."),
        ("user", "{question}")
    ])


    # Create Chain 

    chain=prompt| llm | StrOutputParser()

    return chain


# Get Chain
chain=get_chain(api_key,model_name)

if not chain:
    st.warning("Please enter your Groq API key in the sidebar to start chatting!")
    st.markdown("[Get your free API key here](https://console.groq.com)")

else:
    # Display the chat message

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # Chat Input

    if question:= st.chat_input("Ask me anything"):

        # Add user maessage to session state
        st.session_state.messages.append({"role":"user", "content":question})
        with st.chat_message("user"):
            st.write(question)

        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Stream Response From Groq
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + " ")

                message_placeholder.markdown(full_response)

                # Add to history
                st.session_state.messages.append({"role": "assistant", "content":full_response})


            except Exception as e:
                st.error(f"Error: {str(e)}")


## Examples

st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Built with LangChain & Groq | Experience the speed! ⚡")





