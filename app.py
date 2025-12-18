import streamlit as st
from langchain_core.messages import HumanMessage

from llm import get_llm
from rag import create_vector_store, get_retriever, get_rag_chain
from tools import get_tools
from agent import create_agent_runnable
from dotenv import load_dotenv

load_dotenv()

# Supported languages
LANGUAGES = ["English", "French", "Spanish", "Sinhala", "Hindi", "Tamil"]

@st.cache_resource
def load_components(source_lang: str, target_lang: str):
    llm = get_llm()
    vector_store = create_vector_store()
    retriever = get_retriever(vector_store)
    rag_chain = get_rag_chain(llm, retriever, source_lang, target_lang)
    tools = get_tools(rag_chain, source_lang, target_lang)
    agent = create_agent_runnable(llm, tools, source_lang, target_lang)
    return agent


st.title("Translator Bot POC")
st.markdown("A LangChain-powered translator with RAG, agents, and tools — powered by **OpenAI**")

# Language selection
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("Source Language", LANGUAGES, index=0)
with col2:
    target_lang = st.selectbox("Target Language", LANGUAGES, index=1)

if source_lang == target_lang:
    st.warning("Please select different source and target languages.")
    st.stop()

# Load agent (cached per language pair)
agent = load_components(source_lang, target_lang)

# User input
user_input = st.text_area(
    "Enter text to translate:",
    placeholder="e.g., It's raining cats and dogs, and I walked 5 miles to the bank.",
    height=120
)

if st.button("Translate", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner(f"Translating to {target_lang}..."):
            try:
                response = agent.invoke({
                    "messages": [
                        HumanMessage(content=f"Translate the following text to natural, idiomatic {target_lang}:\n\n{user_input.strip()}")
                    ]
                })

                if isinstance(response, dict) and "messages" in response:
                    messages = response["messages"]
                    if messages and hasattr(messages[-1], "content"):
                        translation = messages[-1].content.strip()
                    else:
                        translation = str(response)
                else:
                    translation = str(response)

                st.success(f"Translation: **{translation}**")

            except Exception as e:
                st.error("An error occurred:")
                st.exception(e)

# Info note
st.info(
    "Note: RAG examples and dictionary tool are optimized for English ↔ French/Spanish. "
    "Other language pairs rely on the LLM's strong multilingual capabilities. "
    "Agent reasoning and tool calls appear in your terminal."
)