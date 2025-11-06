# --- Filename: api.py ---

from dotenv import load_dotenv
import os
from typing import TypedDict, List
from flask import Flask, request, jsonify  
from flask_cors import CORS            

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

# -------------------------
# 1️⃣ Groq Configuration
# -------------------------
load_dotenv(dotenv_path=".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("🚨 Missing GROQ_API_KEY.")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# -------------------------
# 2️⃣ Load FAISS Index
# -------------------------
print("🔍 Loading FAISS index...")
INDEX_PATH = "index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 3️⃣ Define Graph State (with intent)
# -------------------------
class ChatState(TypedDict):
    messages: List
    context: str
    intent: str

# -------------------------
# 4️⃣ Define Graph Nodes
# -------------------------
# ... (Paste ALL your node functions here) ...
# (classify_intent_node, handle_greeting_node, handle_services_node, 
#  retrieve_node, generate_node, handle_contact_node, etc.)

### --- Classification Node (UPDATED) --- ###
def classify_intent_node(state: ChatState):
    """Classify the user's intent to route to the correct node."""
    query = state["messages"][-1].content
    
    classification_prompt = (
        f"You are an intent classifier. Classify the user's query into one of the "
        f"following categories: [greeting, services, contact, appointment, hours, "
        f"location, founder, motto, staffing, industries, experience, general_qa]\n\n"
        f"Query: {query}\n\n"
        f"Category:"
    )
    
    response = llm.invoke(classification_prompt)
    intent = response.content.strip().lower()
    
    if "greeting" in intent:
        state["intent"] = "greeting"
    elif "services" in intent:
        state["intent"] = "services"
    elif "contact" in intent:
        state["intent"] = "contact"
    elif "appointment" in intent:
        state["intent"] = "appointment"
    elif "hours" in intent:
        state["intent"] = "hours"
    elif "location" in intent:
        state["intent"] = "location"
    elif "founder" in intent:
        state["intent"] = "founder"
    elif "motto" in intent:
        state["intent"] = "motto"
    elif "staffing" in intent:
        state["intent"] = "staffing"
    elif "industries" in intent:
        state["intent"] = "industries"
    elif "experience" in intent:
        state["intent"] = "experience"
    elif "experience" in intent:
        state["policy"] = "policy"
    else:
        state["intent"] = "general_qa"
        
    # print(f"DEBUG: Intent classified as: {state['intent']}") # For debugging
    return state

### --- Quick Answer Nodes (EXISTING) --- ###
def handle_greeting_node(state: ChatState):
    reply = "Hello! I'm the SLCI assistant. How can I help you today?"
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_services_node(state: ChatState):
    reply = ("SLCI offers ESI & EPF compliance, Labour Law Compliances, HR Solutions, and Payroll. ")
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_contact_node(state: ChatState):
    reply = ("You can call us at +91 9999329153 or 011-41609501, or email us at contact@slci.in.")
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_appointment_node(state: ChatState):
    reply = "You can book an appointment by calling us at +91 9999329153 or emailing contact@slci.in."
    state["messages"].append(AIMessage(content=reply))
    return state

### --- Quick Answer Nodes (NEW) --- ###
def handle_hours_node(state: ChatState):
    reply = "We are open Monday to Saturday, from 9:30 AM to 6:30 PM."
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_location_node(state: ChatState):
    reply = "Our office is located at 83, DSIDC Complex, Okhla Industrial Area Phase 1, New Delhi - 110020."
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_founder_node(state: ChatState):
    reply = "SLCI was founded by Mr. S.K. Sharma, who has over 38 years of experience."
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_motto_node(state: ChatState):
    reply = 'Our motto is "DO BUSINESS, NOT HR".'
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_staffing_node(state: ChatState):
    reply = "Yes, our staffing solutions include recruitment, background verification, and third-party manpower services."
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_industries_node(state: ChatState):
    reply = "We serve a wide range of industries, including manufacturing, logistics, IT, healthcare, and retail."
    state["messages"].append(AIMessage(content=reply))
    return state

def handle_experience_node(state: ChatState):
    reply = "We have over 38 years of experience in the field of law and compliance."
    state["messages"].append(AIMessage(content=reply))
    return state
def handle_privicy_poilicy(state: ChatState):
    reply = """Data We Collect: We collect personal and confidential information solely for the purpose of providing legal services and meeting professional obligations.
    How We Use It: Your data is used exclusively for client representation, case management, billing, and communication related to our engagement.
    Sharing & Security: We do not sell your information and protect it with reasonable security measures; data is only shared as required by law or with your explicit consent."""
    state["messages"].append(AIMessage(content=reply))
    return state

### --- Original RAG Nodes --- ###
def retrieve_node(state: ChatState):
    print("DEBUG: Retrieving documents...")
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    state["context"] = context
    return state

def generate_node(state: ChatState):
    print("DEBUG: Generating answer with RAG...")
    query = state["messages"][-1].content
    context = state.get("context", "")

    prompt = (
        f"You are an AI assistant answering questions about SLCI.\n"
        f"Use the following document context to give concise and accurate responses.\n\n"
        f"Context:\n{context}\n\n"
        f"User: {query}\n\n"
        f"Answer:"
    )

    response = llm.invoke(prompt)
    reply = response.content if hasattr(response, "content") else str(response)
    state["messages"].append(AIMessage(content=reply))
    return state


# -------------------------
# 5️⃣ Build the LangGraph
# -------------------------
print("🔧 Building graph with expanded conditional routing...")
graph = StateGraph(ChatState)

# --- Add ALL nodes ---
graph.add_node("classify_intent", classify_intent_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("greeting", handle_greeting_node)
graph.add_node("services", handle_services_node)
graph.add_node("contact", handle_contact_node)
graph.add_node("appointment", handle_appointment_node)
graph.add_node("hours", handle_hours_node)
graph.add_node("location", handle_location_node)
graph.add_node("founder", handle_founder_node)
graph.add_node("motto", handle_motto_node)
graph.add_node("staffing", handle_staffing_node)
graph.add_node("industries", handle_industries_node)
graph.add_node("experience", handle_experience_node)
graph.add_node("policy", handle_privicy_poilicy)

graph.set_entry_point("classify_intent")

def route_after_classification(state: ChatState):
    intent = state.get("intent")
    if intent == "general_qa":
        return "retrieve"
    else:
        return intent

# --- UPDATED conditional edge mapping ---
graph.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "general_qa": "retrieve", "greeting": "greeting", "services": "services",
        "contact": "contact", "appointment": "appointment", "hours": "hours",
        "location": "location", "founder": "founder", "motto": "motto",
        "staffing": "staffing", "industries": "industries", "experience": "experience","policy":"policy"
    }
)

graph.add_edge("retrieve", "generate")

# --- Connect ALL paths to the END ---
graph.add_edge("generate", END)
graph.add_edge("greeting", END)
graph.add_edge("services", END)
graph.add_edge("contact", END)
graph.add_edge("appointment", END)
graph.add_edge("hours", END)
graph.add_edge("location", END)
graph.add_edge("founder", END)
graph.add_edge("motto", END)
graph.add_edge("staffing", END)
graph.add_edge("industries", END)
graph.add_edge("experience", END)

# -------------------------
# 6️⃣ Memory Checkpointer & Compile
# -------------------------
print("✅ Compiling graph...")
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# -----------------------------------
# 7️⃣ NEW: Flask API Server
# -----------------------------------
print("🚀 Starting Flask server...")
flask_app = Flask(__name__)
CORS(flask_app)  # Enable CORS for all routes

@flask_app.route("/chat", methods=["POST"])
def chat():
    # Get message and session_id from the frontend
    data = request.json
    user_message = data.get("message")
    session_id = data.get("session_id")

    if not user_message or not session_id:
        return jsonify({"error": "Missing 'message' or 'session_id'"}), 400

    # This config tells LangGraph which conversation thread to use
    config = {"configurable": {"thread_id": session_id}}

    # The input state for the graph
    current_chat_state = {
        "messages": [HumanMessage(content=user_message)],
        "context": "", 
        "intent": ""
    }

    # Invoke the graph
    response_state = app.invoke(current_chat_state, config=config)

    # Get the *last* message (the bot's reply)
    ai_response = response_state["messages"][-1].content
    
    # Send the reply back to the frontend
    return jsonify({"reply": ai_response})

if __name__ == "__main__":
    # Run the Flask server
    # Note: `debug=True` is great for development,
    # but change it to `debug=False` for production.
    flask_app.run(port=5000, debug=True)