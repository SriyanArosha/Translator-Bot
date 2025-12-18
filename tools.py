from langchain_core.tools import tool
from rag import get_rag_chain, create_vector_store, get_retriever
from llm import get_llm

@tool
def dictionary_lookup(word: str, source_lang: str = "English", target_lang: str = "French") -> str:
    """Look up a word for context in translation (handles ambiguity). Supports basic pairs."""
    # Expand mock dict for more languages
    mock_dict = {
        ("english", "french"): {
            "bank": "banque (financial) or rive (river)",
            "apple": "pomme (fruit) or Apple (company)",
        },
        ("english", "spanish"): {
            "bank": "banco (financial) or orilla (river)",
            "apple": "manzana (fruit) or Apple (company)",
        },
        ("english", "sinhala"): {
            "bank": "බැංකුව (financial) or ගඟ (river)",
            "apple": "ඇපල් (fruit) or Apple (company)",
        },
        ("english", "hindi"): {
            "bank": "बैंक (financial) or नदी (river)",
            "apple": "सेब (fruit) or Apple (company)",
        },
        ("english", "tamil"): {
            "bank": "வங்கி (financial) or ஆறு (river)",
            "apple": "ஆப்பிள் (fruit) or Apple (company)",
        },
    }
    key = (source_lang.lower(), target_lang.lower())
    lang_dict = mock_dict.get(key, {})
    return lang_dict.get(word.lower(), f"No special context found for '{word}' in {source_lang} to {target_lang}.")

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert common units (language-agnostic)."""
    conversions = {
        ("miles", "km"): value * 1.60934,
        ("feet", "meters"): value * 0.3048,
        ("pounds", "kg"): value * 0.453592,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key]
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    return "Unsupported conversion."

# RAG tool will be initialized later
rag_retrieval_tool = None

def get_tools(rag_chain, source_lang="English", target_lang="French"):
    global rag_retrieval_tool
    
    @tool
    def rag_context_retriever(query: str) -> str:
        """Retrieve relevant translation examples or idioms for better accuracy."""
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    
    # Update dictionary_lookup to take langs (but since it's a tool, args are passed at call time)
    dictionary_lookup.args_schema = None  # If needed, but for POC, agent will pass args
    
    rag_retrieval_tool = rag_context_retriever
    return [dictionary_lookup, unit_converter, rag_retrieval_tool]