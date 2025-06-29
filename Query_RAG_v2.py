import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from collections import defaultdict
import json
import os
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import time


# Load environment variables
load_dotenv()

# Initialize embedder and Chroma client
embedder = OllamaEmbeddings(model="mxbai-embed-large")
client = chromadb.PersistentClient(path="chromadb")
namespaces = [col.name for col in client.list_collections()]
llm_local=OllamaLLM(model="llama3.2", temperature=0)


# Function to select LLM
def get_llm(model_choice):
    if model_choice.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    elif model_choice.lower() == "ollama":
        return OllamaLLM(model="llama3.2", temperature=0)
    else:
        raise ValueError("Invalid model choice. Use 'ollama' or 'gemini'.")

# Prompt for sub-question generation
custom_prompt_str = """
You are a Claims Analyst working for a life insurer. Your task is to break down complex user questions into smaller, focused sub-questions.

You have access to the following tools (each tool corresponds to a specific Product Disclosure Statement (PDS) document):

{tools}

Each tool name includes a product code and version date. The user query may mention one or more of these tools explicitly. Your job is to:
- Identify which tools are explicitly mentioned in the query. 
- Generate sub-questions only for those tools.
- Do not include tools that are not mentioned in the query.

Given the user query: "{query}", generate a list of sub-questions in the following JSON format:

[
  {{
    "sub_question": "Sub-question text here",
    "tool_name": "name_of_the_tool"
  }},
  ...
]

- Use the tool_name exactly as provided in the list.
- Do not include any explanations or extra text outside the JSON.
"""

# Retrieve contexts
def retrieve_contexts_for_subquestions(llm_output, client, embedder, k=5):
    content = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
    if content.startswith('```json'):
        content = content.replace('```json\n', '').replace('\n```', '')
    elif content.startswith('```'):
        content = content.replace('```\n', '').replace('\n```', '')

    sub_questions = json.loads(content)
    all_docs = []

    for item in sub_questions:
        namespace = item["tool_name"]
        question = item["sub_question"]

        vectorstore = Chroma(
            client=client,
            collection_name=namespace,
            embedding_function=embedder
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "lambda_mult": 0.6}
        )

        results = retriever.invoke(question)

        for doc in results:
            metadata = doc.metadata
            all_docs.append({
                "namespace": namespace,
                "sub_question": question,
                "source_file": metadata.get("source_file", "N/A"),
                "document_name": metadata.get("document_name", metadata.get("title", "N/A")),
                "section_title": metadata.get("section_title", "N/A"),
                "subheading": metadata.get("subheading", "N/A"),
                "content": doc.page_content,
                "insurer": metadata.get("insurer", "N/A"),
                "product_type": metadata.get("product_type", "N/A"),
                "page_no": metadata.get("page_no", "N/A")
            })
    return all_docs

# Format documents
def group_and_format_docs(docs):
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc['namespace']].append(doc)

    formatted = []
    for namespace, entries in grouped.items():
        formatted.append(f"=== Namespace: {namespace} ===")
        for doc in entries:
            formatted.append(
                f"--- START OF DOCUMENT ---\n"
                f"Sub-question: {doc['sub_question']}\n"
                f"Source File: {doc['source_file']}\n"
                f"Insurer: {doc.get('insurer', 'N/A')}\n"
                f"Product Type: {doc.get('product_type', 'N/A')}\n"
                f"Page No: {doc.get('page_no', 'N/A')}\n"
                f"Section Title: {doc.get('section_title', 'N/A')}\n"
                f"Subheading: {doc.get('subheading', 'N/A')}\n"
                f"Content:\n{doc['content']}\n"
                f"--- END OF DOCUMENT ---"
            )
    return "\n\n".join(formatted)
    
    
def display_subquestions_table(result):
    st.subheader("Generated Sub-questions")

    # Extract and clean JSON
    subquestions = result.content if hasattr(result, 'content') else str(result)
    if isinstance(subquestions, str):
        if subquestions.startswith("```json"):
            subquestions = subquestions.replace("```json\n", "").replace("\n```", "")
        elif subquestions.startswith("```"):
            subquestions = subquestions.replace("```\n", "").replace("\n```", "")
        subquestions = json.loads(subquestions)

    # Rename columns
    df = pd.DataFrame(subquestions)
    df = df.rename(columns={"sub_question": "Sub-question", "tool_name": "Tool Name"})

    # Display with word wrap using HTML
    st.markdown(
        df.to_html(escape=False, index=False, justify="left", classes="styled-table"),
        unsafe_allow_html=True
    )

    # Add custom CSS for word wrapping
    st.markdown("""
        <style>
        .styled-table td {
            white-space: normal !important;
            word-wrap: break-word;
            max-width: 400px;
        }
        </style>
    """, unsafe_allow_html=True)





# Final answer generation
def generate_final_answer(original_query, context_str, llm):
    prompt = ChatPromptTemplate.from_template("""
You are an expert in Life Insurance product disclosure statements, working in the claims department.

You are provided with multiple document excerpts. Each excerpt is clearly marked with:
- '--- START OF DOCUMENT ---'
- Namespace
- Sub-question
- Source File
- Document Name
- Section Title
- Subheading
- Page No
- Content
- '--- END OF DOCUMENT ---'

Your task is to:
1. Carefully read and extract relevant information from each document based on the user's question.
2. Compare and contrast the information across the different documents.
3. Clearly identify which document each piece of information comes from.
4. When referencing information, include a numbered footnote in square brackets (e.g., [1], [2], etc.) that corresponds to the source listed in step 6.
5. Ensure your answer is structured, accurate, and easy to follow.
6. At the end, MUST include a "Sources" section in a neat numbered list format. Each number must match the footnote used in the answer. For each source, include:
   - Document Name
    - Section Title: Section Title
    - Subheading: Subheading 
    - Page Number: Page No

If the answer cannot be found in the provided documents, say "Not found in the provided sources." Do not guess or hallucinate or use any external infomation apart from what is provided.

Context:
{context}

Question:
{question}
""")
    chain = prompt | llm
    return chain.invoke({"context": context_str, "question": original_query})

# === Streamlit UI ===
st.set_page_config(page_title="Life Insurance RAG Assistant", layout="wide")
st.title("Life Insurance RAG Assistant")

# Sidebar: logo + model selector + formatted namespace table
with st.sidebar:
    # Remove top padding from sidebar and image container
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem;
        }
        img {
            margin-top: -70px;
            margin-left: -30px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display logo
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=380)
    except FileNotFoundError:
        st.warning("Logo not found. Please place 'logo.png' in the app directory.")

    # Settings and namespaces
    st.header("🔧 Settings")
    model_choice = st.selectbox("Choose LLM model:", ["ollama", "gemini"])

    st.markdown("### 📂 Available Files")
    namespace_df = pd.DataFrame(namespaces, columns=["Namespace"])
    st.markdown(
        namespace_df.to_html(escape=False, index=False, justify="left", classes="styled-table"),
        unsafe_allow_html=True
    )
    st.markdown("""
        <style>
        .styled-table td {
            white-space: normal !important;
            word-wrap: break-word;
            max-width: 250px;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)



# Main area: query input
query = st.text_area("Enter your query:", height=100, key="main_query_input")

# Load LLM after model selection
llm = get_llm(model_choice)

# Submit button and processing logic
if st.button("Submit Query"):
    with st.spinner("Processing..."):
        start_time=time.time()
        # Run the chain to get sub-questions
        prompt = ChatPromptTemplate.from_template(custom_prompt_str)
        chain = prompt | llm
        result = chain.invoke({"query": query, "tools": namespaces})

        # Retrieve and format contexts
        formatted_docs = retrieve_contexts_for_subquestions(result, client, embedder)
        context_str = group_and_format_docs(formatted_docs)

        # Generate final answer
        final_answer = generate_final_answer(query, context_str, llm)
        
        end_time=time.time()
        elapsed_minutes = (end_time - start_time) / 60
        elapsed_seconds = (end_time - start_time)
        
        st.markdown(
        f"<p style='font-size:12px;'>Script executed in {elapsed_minutes:.2f} minutes ({elapsed_seconds:.0f} seconds).</p>",
        unsafe_allow_html=True)


        
        # Create tabs
        tab1, tab2 = st.tabs(["Sub-questions & Generated Answer", "Retrieved Contexts"])

        with tab1:
            display_subquestions_table(result)
            st.subheader("Generated Answer")
            st.markdown(final_answer.content if hasattr(final_answer, 'content') else str(final_answer))

        with tab2:
            st.subheader("Retrieved Contexts")
            st.text(context_str)



