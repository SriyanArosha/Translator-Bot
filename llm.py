from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.7):
    """Return configured ChatOpenAI instance."""
    return ChatOpenAI(model=model, temperature=temperature)

def get_embeddings():
    """Return OpenAI embeddings."""
    return OpenAIEmbeddings()