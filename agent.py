from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

def create_agent_runnable(llm, tools, source_lang="English", target_lang="French"):
    """
    Creates a modern agent using the unified create_agent.
    Works great with any model (including OpenAI, Hugging Face, etc.).
    """
    system_prompt = f"""You are an intelligent Translator Bot that translates from {source_lang} to {target_lang}.
    Use tools when necessary:
    - Use rag_context_retriever for idioms or common expressions (if relevant for the language pair).
    - Use dictionary_lookup for ambiguous words.
    - Use unit_converter for measurements or units in the text.
    If no tool is needed, translate directly.
    Always output only the final translation in {target_lang}.
    Be natural and idiomatic."""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SystemMessage(content=system_prompt),
    )

    return agent