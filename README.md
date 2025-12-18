# Translator-Bot

This project implements the *Translator Bot* with Python and uses interaction with a Large Language Model (LLM) to translate text between languages with ease. Constructed using LangChain, the system facilitates modular component integration, context management, and prompt coordination. Agents are used to make dynamic decisions, like choosing the best translation approach or deftly managing user input. When necessary, the bot can use tool-calling capabilities to launch custom preprocessing tools, language detection services, or external translation tools. Furthermore, by obtaining pertinent linguistic rules, domain-specific terminology, or previous examples from a knowledge base, a Retrieval-Augmented Generation (RAG) approach improves translation accuracy and ensures more context-aware and trustworthy translations.


## Prerequisites

Python 3.8+

## Running the Demo

Open a terminal in the project directory.

Run the following commands:

```bash
pip install -r requirements.txt
```
(Above command is one time execution command. No need to use it everytime you run the application)

```bash
streamlit run app.py
```


## Key Concepts Covered

### Interaction with LLM

Used an LLM (OpenAI's GPT) for core translation and reasoning.

### LangChain usage

LangChain will handle chains, prompts, and integrations.

### Agents

An agent will decide when to use tools based on user input (if the translation needs external lookup or context).

### Tool calling capabilities

Custom tools for tasks like fetching dictionary definitions or currency conversions (to make translations context-aware, handling idioms or units).

### RAG implementation

Retrieval-Augmented Generation to pull relevant examples or contexts from a vector store (pre-stored translation pairs for domain-specific accuracy).


## Module Design

### app.py

This is the main file that runs the Streamlit web app you see in your browser.

### llm.py

Connects to OpenAI’s powerful AI model (like GPT-3.5 or GPT-4).

### rag.py

Helps the bot remember good example translations (especially useful for common phrases).

### tools.py

Gives the agent extra powers beyond just translating.

### agent.py

Creates an "agent" — a smart system that decides whether to translate directly or use tools.