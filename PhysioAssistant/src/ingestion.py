import os
import re
from typing import List, Dict
from dotenv import load_dotenv
import pdfplumber
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

class ExerciseMetadata(BaseModel):
    condition_name: str = Field(description="Name of the medical condition (e.g., Adhesive Capsulitis, ACL Tear)")
    body_part: str = Field(description="Body part affected (e.g., Shoulder, Knee, Back)")
    exercise_type: str = Field(description="Type of exercise (e.g., Stretching, Strengthening, ROM)")
    contraindications: str = Field(description="Any contraindications or warnings mentioned")

def clean_text(text: str) -> str:
    """Removes common PDF artifacts like page numbers, headers, and footers."""
    if not text:
        return ""
    # Remove lines that look like page headers/footers (e.g., "9865.ch09 2/14/02 2:29 PM Page 331")
    text = re.sub(r'\d+\.\w+\s+\d+/\d+/\d+\s+\d+:\d+\s+[AP]M\s+Page\s+\d+', '', text)
    # Remove "CHAPTER X" or "PART X" patterns
    text = re.sub(r'CHAPTER\s+\d+', '', text, flags=re.IGNORECASE)
    # Remove redundant whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_metadata_from_chunk(chunk_text: str, llm: ChatOpenAI) -> Dict:
    """Uses LLM to extract semantic metadata for filtering."""
    prompt = f"""
    Analyze the following physiotherapy exercise instruction and extract structured metadata.
    If a specific detail is missing, infer it from the context or use "General".
    
    Instruction:
    {chunk_text}
    
    Return ONLY JSON matching the schema:
    {{
      "condition_name": "...",
      "body_part": "...",
      "exercise_type": "...",
      "contraindications": "..."
    }}
    """
    try:
        response = llm.with_structured_output(ExerciseMetadata).invoke(prompt)
        return response.dict()
    except Exception as e:
        return {
            "condition_name": "General",
            "body_part": "General",
            "exercise_type": "Exercise",
            "contraindications": "None specified"
        }

def process_pdf(pdf_path: str, db_path: str):
    print(f"--- Phase 1: Data Ingestion ---")
    print(f"Loading PDF: {pdf_path}")
    
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Extracting text"):
            page_text = page.extract_text()
            full_text += clean_text(page_text) + "\n\n"
    
    print("Initializing Semantic Chunking (using OpenRouter LLM for breakpoints)...")
    
    # Configure OpenRouter for LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    
    # For embeddings, we'll try to use the OpenAI API key provided (which is OpenRouter)
    # OpenRouter now supports embeddings for some models, or we can use the OpenAI compatible endpoint.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    
    # We create raw chunks first
    raw_chunks = text_splitter.create_documents([full_text])
    print(f"Created {len(raw_chunks)} semantic chunks.")
    
    final_docs = []
    
    print("Enriching chunks with semantic metadata...")
    for chunk in tqdm(raw_chunks, desc="Processing metadata"):
        # Skip chunks that are too small to contain meaningful instructions
        if len(chunk.page_content.split()) < 20:
            continue
            
        metadata = extract_metadata_from_chunk(chunk.page_content, llm)
        enriched_doc = Document(
            page_content=chunk.page_content,
            metadata=metadata
        )
        final_docs.append(enriched_doc)

    print(f"Saving {len(final_docs)} chunks to FAISS index at {db_path}...")
    vector_store = FAISS.from_documents(final_docs, embeddings)
    vector_store.save_local(db_path)
    print("--- Phase 1 Complete ---")

if __name__ == "__main__":
    PDF_FILE = r"C:\Users\PARTH PACHORI\Downloads\therapeutic-exercise.pdf"
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "physio_index")
    
    if not os.path.exists(PDF_FILE):
        print(f"CRITICAL ERROR: PDF not found at {PDF_FILE}")
    else:
        # Create DB directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        process_pdf(PDF_FILE, DB_PATH)
