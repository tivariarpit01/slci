import os
from typing import TypedDict, List
from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq  # GROQ LLM

# -------------------------
# 0️⃣ Load .env file
# -------------------------
load_dotenv()  # loads variables from .env into os.environ

# -------------------------
# 1️⃣ Groq Configuration
# -------------------------
load_dotenv()  # load .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("🚨 Missing GROQ_API_KEY. Make sure it is in your .env file.")


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",  # or another supported model
    temperature=0.3
)

# -------------------------
# 2️⃣ Load FAISS Index
# -------------------------
print("🔍 Loading FAISS index...")

INDEX_PATH = "modules/slci_faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 3️⃣ Define Graph State
# -------------------------
class ChatState(TypedDict):
    messages: List
    context: str

# -------------------------
# 4️⃣ Define Graph Nodes
# -------------------------
def retrieve_node(state: ChatState):
    """Retrieve relevant chunks from FAISS index."""
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    state["context"] = context
    return state

def generate_node(state: ChatState):
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
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# -------------------------
# 6️⃣ Memory Checkpointer
# -------------------------
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# -------------------------
# 7️⃣ Run Interactive Chat
# -------------------------
print("✅ LangGraph + GROQ Chatbot ready! Ask anything about SLCI.\n")

chat_state = {"messages": [], "context": ""}
session_config = {"configurable": {"thread_id": "slci_chat_session"}}

while True:
    user_input = input("🧩 You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Bye! Chat session ended.")
        break

    chat_state["messages"].append(HumanMessage(content=user_input))
    chat_state = app.invoke(chat_state, config=session_config)

    ai_response = chat_state["messages"][-1].content
    print("\n🤖 Bot:", ai_response)
    print("-" * 60)
