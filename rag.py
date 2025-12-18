from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm, get_embeddings

# Sample translation examples (expand to more language pairs for POC)
SAMPLE_DOCS = [
    # English-French
    Document(page_content="English: Good Morning. French: Bonjour."),
    Document(page_content="English: This is Test Tool. French: Ceci est un outil de test."),
    Document(page_content="English: Hi, How are you? French: Salut, comment vas-tu?"),
    Document(page_content="English: How can I help you? French: Comment puis-je t'aider?"),
    # English-Spanish
    Document(page_content="English: Good Morning. Spanish: Buen día."),
    Document(page_content="English: This is Test Tool. Spanish: Esta es una herramienta de prueba."),
    Document(page_content="English: Hi, How are you? Spanish: ¿Hola, cómo estás?"),
    Document(page_content="English: How can I help you? Spanish: ¿Le puedo ayudar en algo?"),
    # English-Sinhala
    Document(page_content="English: Good Morning. Sinhala: සුභ උදෑසනක්."),
    Document(page_content="English: This is Test Tool. Sinhala: මෙය පරීක්ෂණ මෙවලමකි."),
    Document(page_content="English: Hi, How are you? Sinhala: ආයුබෝවන්, ඔයාට කොහොම ද?"),
    Document(page_content="English: How can I help you? Sinhala: මම ඔයාට උදව් කරන්නේ කෙසේ ද?"),
    # English-Hindi
    Document(page_content="English: Good Morning. Hindi: शुभ प्रभात."),
    Document(page_content="English: This is Test Tool. Hindi: यह टेस्ट टूल है."),
    Document(page_content="English: Hi, How are you? Hindi: हैलो, क्या हाल हैं?"),
    Document(page_content="English: How can I help you? Hindi: मैं आपकी कैसे मदद कर सकता हूँ?"),
    # English-Tamil
    Document(page_content="English: Good Morning. Tamil: காலை வணக்கம்."),
    Document(page_content="English: This is Test Tool. Tamil: இது சோதனைக் கருவி."),
    Document(page_content="English: Hi, How are you? Tamil: வணக்கம், எப்படி இருக்கீங்க?"),
    Document(page_content="English: How can I help you? Tamil: நான் உங்களுக்கு எப்படி உதவ முடியும்?"),
]

def create_vector_store():
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(SAMPLE_DOCS, embeddings)
    return vector_store

def get_retriever(vector_store, k: int = 3):
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_rag_chain(llm, retriever, source_lang="English", target_lang="French"):
    rag_prompt = ChatPromptTemplate.from_template(
        f"""Use the following retrieved context to improve the translation from {source_lang} to {target_lang}.
        Only use the context if it is relevant and matches the language pair.

        Context:
        {{context}}

        Text to translate: {{input}}

        Translation:"""
    )

    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain