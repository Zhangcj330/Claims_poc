import chromadb
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from collections import defaultdict
import json
import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# Initialize embedder and persistent ChromaDB client
query = "give me separately the Female CI medical conditions for both CS_TALR7983-0923-accelerated-protection-pds-8-sep-2023 and CS_TAL_AcceleratedProtection_2022-08-05"
# embedder = OllamaEmbeddings(model="mxbai-embed-large")
embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
client = chromadb.PersistentClient(path="chromadb")
namespaces = [col.name for col in client.list_collections()]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

# === Custom Prompt ===
custom_prompt_str = """
You are a Claims Analyst working for a life insurer. Your task is to break down complex user questions into smaller, focused sub-questions.

You have access to the following tools (each tool corresponds to a specific Product Disclosure Statement document):

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
prompt = ChatPromptTemplate.from_template(custom_prompt_str)
chain = prompt | llm
result = chain.invoke({"query": query, "tools": namespaces})


print(result)

# === Retrieve and Format Contexts ===
def retrieve_contexts_for_subquestions(llm_output, client, embedder, k=5):
    # Extract content from AIMessage and remove markdown formatting
    content = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
    # Remove markdown code block formatting if present
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
            search_kwargs={"k": k, "lambda_mult": 0.8}
        )

        results = retriever.get_relevant_documents(question)

        for doc in results:
            metadata = doc.metadata
            all_docs.append({
                "namespace": namespace,
                "sub_question": question,
                "source_file": metadata.get("source_file", "N/A"),
                "title": metadata.get("title", "N/A"),
                "subheader": metadata.get("subheader", "N/A"),
                "content": metadata.get("raw_content", "")
            })
        

    return all_docs

# === Group and Format Contexts ===
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
                f"Title: {doc['title']}\n"
                f"Subheader: {doc['subheader']}\n"
                f"Content:\n{doc['content']}\n"
                f"--- END OF DOCUMENT ---"
              
            )
    return "\n\n".join(formatted)

# === Final Answer Generator ===
def generate_final_answer(original_query, context_str, llm):
    prompt = ChatPromptTemplate.from_template("""
You are an expert in Life Insurance product disclosure statements, working in the claims department.

You are provided with multiple document excerpts. Each excerpt is clearly marked with:
- '--- START OF DOCUMENT ---'
- Namespace
- Sub-question
- Source File
- Title
- Subheader
- Content
- '--- END OF DOCUMENT ---'

Your task is to:
1. Carefully read and extract relevant information from each document based on the user's question.
2. Compare and contrast the information across the different documents.
3. Clearly identify which document each piece of information comes from.
4. When referencing information, always include the [source_file, subheader] in brackets.
5. Ensure your answer is structured, accurate, and easy to follow.
6. At the end, include a "Sources" section in bullet format, listing:
   - Subheader
   - Title
   - Source File

You are not limited to any specific topic — the question may relate to definitions, exclusions, benefits, waiting periods, claim processes, or any other aspect of the PDS.
If the answer cannot be found in the provided documents, say "Not found in the provided sources." Do not guess or hallucinate.

Context:
{context}

Question:
{question}

Do not forget to include a "Sources" section in bullet format (task 6) which includes the Subheader, Title, and Source File for all the documnets used
""")

    chain = prompt | llm
    return chain.invoke({"context": context_str, "question": original_query})

# === Run Pipeline ===
formatted_docs = retrieve_contexts_for_subquestions(result, client, embedder)

context_str = group_and_format_docs(formatted_docs)
print(context_str)

print("==========================================================")
final_answer = generate_final_answer(query, context_str, llm)
print("\nFinal Answer:\n", final_answer)
