from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Read and preprocess the PDF
reader = PdfReader("data/real-estate-report.pdf")  # Update with your PDF path
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]

# Split text into chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0,
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Set up ChromaDB and add documents
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "real-estate-collection", embedding_function=embedding_function
)
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

# Define function to generate augmented queries
def generate_multiple_queries(query, model="gpt-3.5-turbo"):
    system_prompt = """You are a knowledgeable real estate analyst. 
    For the given question about a real estate investment, propose up to five related questions to assist in finding relevant information. 
    Provide concise, single-topic questions that cover various aspects of the topic (e.g., market trends, financials, property condition). 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content.split("\n")

# Original query and augmented queries
org_query = """Analyze this real estate investment and provide a detailed recommendation:
Recommend whether the owner should:
- Sell (if market is peaking)
- Rent (if rental demand is high)
- Hold (if future appreciation is likely; suggest years)
Provide a step-by-step action plan to maximize ROI, including:
- Pricing strategy (if selling)
- Rental pricing tips (if renting)
- Renovations or improvements needed
- Risks to consider
Write in a professional, easy-to-understand tone."""
aug_queries = generate_multiple_queries(org_query)

# Combine queries
joint_query = [org_query] + aug_queries

# Query ChromaDB
results = chroma_collection.query(
    query_texts=joint_query,
    n_results=5,
    include=["documents"]
)
retrieved_docs = results["documents"]

# Combine unique documents
unique_documents = set()
for docs in retrieved_docs:
    for document in docs:
        unique_documents.add(document)
context = "\n\n".join(unique_documents)

# Generate recommendation using GPT-3.5
system_prompt = """You are an expert real estate analyst with access to a real estate investment document. Your task is to analyze the provided document and extract relevant information such as property type, location, market trends, financial metrics (e.g., purchase price, rental income, expenses, ROI), property condition, and any other pertinent details. If specific data is missing, make reasonable assumptions based on general real estate trends and note these assumptions clearly. Provide concise, actionable advice in the following format:

**Recommendation:** [SELL/RENT/HOLD X years]  
**Action Plan:**  
- Step 1: [Specific action, e.g., set a competitive sale price or improve property condition]  
- Step 2: [Specific action, e.g., target specific tenant groups or monitor market trends]  
**Risks:** [Key risks, e.g., market downturn, high vacancy rates, or unexpected maintenance costs]

Ensure recommendations are data-driven, professional, and easy to understand. If data is insufficient, prioritize conservative assumptions and highlight them."""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Document excerpts:\n{context}\n\n{org_query}"}
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
)
recommendation = response.choices[0].message.content

# Output the recommendation
print("Real Estate Investment Recommendation:")
print(recommendation)