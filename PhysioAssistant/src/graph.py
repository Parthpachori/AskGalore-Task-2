import os
import sqlite3
from typing import Annotated, List, Dict, TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

# --- State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]
    intent: str
    body_part: str
    condition: str
    retrieved_docs: List[str]
    safety_breach: bool
    red_flags: List[str]
    response: str
    patient_context: Dict[str, str] # Track injury, pain location, etc.

# --- Structured Output Schemas ---

class IntentClassifierOutput(BaseModel):
    intent: str = Field(description="One of: exercise_guidance, pain_explanation, rehab_plan, safety_warning")
    body_part: str = Field(description="Affected body part")
    condition: str = Field(description="Possible condition name mentioned by user")
    recovery_stage: str = Field(description="Current stage: Acute, Sub-acute, or Chronic")

class SafetyCheckOutput(BaseModel):
    has_red_flags: bool = Field(description="True if emergency symptoms detected")
    detected_flags: List[str] = Field(description="List of red flags like numbness, severe swelling, etc.")
    should_escalate: bool = Field(description="True if user should see a doctor immediately")

# --- Nodes ---

def get_llm(model="nvidia/nemotron-nano-9b-v2:free", temperature=0):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

import json

def analyze_query(state: AgentState):
    llm = get_llm()
    last_message = state['messages'][-1].content
    
    prompt = ChatPromptTemplate.from_template("""
    Analyze the user query for intent, physical context, and clinical safety.
    Intents: exercise_guidance, pain_explanation, rehab_plan, safety_warning.
    Stages: Acute, Sub-acute, Chronic.
    MUST-REFER Red flags: Loss of bladder/bowel control, Numbness in saddle/limbs, Severe swelling/deformity, Fever with joint pain, Night pain preventing sleep.
    
    You MUST respond with ONLY valid raw JSON structure. No markdown blocks, no text outside the JSON.
    Example clean:
    {{"intent": "exercise_guidance", "body_part": "knee", "condition": "pain", "recovery_stage": "Acute", "should_escalate": false, "detected_flags": []}}
    
    Example with red flags:
    {{"intent": "safety_warning", "body_part": "lower back", "condition": "numbness", "recovery_stage": "Acute", "should_escalate": true, "detected_flags": ["Numbness in limbs"]}}
    
    Query: {query}
    """)
    
    res = llm.invoke(prompt.format(query=last_message))
    
    try:
        content = res.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        parsed = json.loads(content.strip())
    except Exception as e:
        print("Fallback analyzer failed", e)
        parsed = {
            "intent": "exercise_guidance",
            "body_part": "general",
            "condition": "general",
            "recovery_stage": "Sub-acute",
            "should_escalate": False,
            "detected_flags": []
        }
    
    return {
        "intent": parsed.get("intent", "exercise_guidance"),
        "body_part": parsed.get("body_part", "general"),
        "condition": parsed.get("condition", "general"),
        "patient_context": {
            "body_part": parsed.get("body_part", "general"),
            "condition": parsed.get("condition", "general"),
            "recovery_stage": parsed.get("recovery_stage", "Sub-acute")
        },
        "safety_breach": parsed.get("should_escalate", False),
        "red_flags": parsed.get("detected_flags", [])
    }

def retriever_node(state: AgentState):
    if state.get("safety_breach"):
        return {"retrieved_docs": []}
        
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
    db_path = os.path.join(os.path.dirname(__file__), "..", "db", "physio_index")
    
    try:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        # Filter docs by body part for higher precision
        query = f"{state['body_part']} {state['condition']} {state['messages'][-1].content}"
        docs = vector_store.similarity_search(query, k=4)
        return {"retrieved_docs": [d.page_content for d in docs]}
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return {"retrieved_docs": ["Failed to retrieve documents due to API limitations (e.g. Embedding costs). Provide general advice."]}

def response_generator(state: AgentState):
    llm = get_llm(temperature=0.7)
    
    if state.get("safety_breach"):
        flags = ", ".join(state['red_flags'])
        response = (
            f"⚠️ **EMERGENCY NOTICE**: I have detected potential red-flag symptoms: {flags}. \n\n"
            "Please stop all activity and consult a licensed medical professional or visit an emergency room immediately. "
            "These symptoms can indicate serious underlying issues that require an in-person clinical exam. "
            "\n\n*General Advice: Do not attempt any new exercises until cleared by a doctor.*"
        )
        return {"response": response, "messages": [AIMessage(content=response)]}

    context = "\n\n".join(state['retrieved_docs'])
    prompt = ChatPromptTemplate.from_template("""
    You are a professional Healthcare Physiotherapy Assistant.
    Answer based ONLY on the provided guidelines and medical safety rules.
    
    Medical Safety Rules:
    - NEVER provide a definitive diagnosis.
    - ALWAYS include: "Consult a licensed physiotherapist for a personalized assessment."
    - If pain increases, tell them to STOP.
    
    Response Format:
    1. **Condition Overview**: Briefly explain what might be happening (non-diagnostic).
    2. **Exercise Guidance**: Step-by-step instructions.
    3. **Precautions**: Specific warnings for this movement.
    4. **Next Steps**: When to seek help.
    
    Context:
    {context}
    
    User Query: {query}
    """)
    
    chain_input = {
        "context": context,
        "query": state['messages'][-1].content
    }
    
    res = llm.invoke(prompt.format(**chain_input))
    
    # Append the standard escalation
    final_response = res.content + "\n\n---\n*Disclaimer: This is for informational purposes only. Consult a licensed physiotherapist for a personalized assessment.*"
    
    return {"response": final_response, "messages": [AIMessage(content=final_response)]}

# --- Router ---

def route_after_safety(state: AgentState) -> Literal["generate", "retrieve"]:
    if state.get("safety_breach"):
        return "generate"
    return "retrieve"

# --- Graph Construction ---

def create_assistant_graph():
    # Use SQLite for memory persistence
    conn = sqlite3.connect("physio_memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate", response_generator)
    
    workflow.set_entry_point("analyze_query")
    
    workflow.add_conditional_edges(
        "analyze_query",
        route_after_safety,
        {
            "generate": "generate",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    app = create_assistant_graph()
    # To run: app.invoke({"messages": [HumanMessage(content="My knee is swollen and numb")]}, config={"thread_id": "1"})
