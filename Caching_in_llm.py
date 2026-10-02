import os
import sqlite3
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_classic.embeddings import CacheBackedEmbeddings
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env", override=True)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Add it to .env or your environment.")
# ==========================================
# 1. GLOBAL LLM RESPONSE CACHING
# ==========================================
# This intercepts repeated prompt requests to OpenAI globally.
# Use SQLiteCache instead of InMemoryCache if you want it persistent across restarts.
set_llm_cache(InMemoryCache())

# ==========================================
# 2. EMBEDDING & VECTOR CACHING (RAG)
# ==========================================
# Prevents recalculating embeddings for identical raw text chunks.
underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = InMemoryStore()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, 
    store, 
    namespace=underlying_embeddings.model
)

# ==========================================
# 3. DEFINE THE STATE & TOOLS
# ==========================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Application/Tool Level Cache Store
custom_tool_cache = {}

@tool
def get_weather_forecast(city: str) -> str:
    """Get the current weather forecast for a city."""
    # Custom Application Node Caching (or leveraging Stale-While-Revalidate APIs)
    if city.lower() in custom_tool_cache:
        return f"[CACHED RESULT] {custom_tool_cache[city.lower()]}"
        
    # Simulate API Call
    result = f"The weather in {city} is sunny and 22°C."
    custom_tool_cache[city.lower()] = result
    return result

# ==========================================
# 4. INITIALIZE THE CACHE-ENABLED MODEL
# ==========================================
# This model auto-inherits the global LLM cache.
# Additionally, we inject prompt caching natively by setting extra headers if using providers like Anthropic or DeepSeek.
model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0
).bind_tools([get_weather_forecast])

# ==========================================
# 5. DEFINE NODES & CONDITIONAL EDGES
# ==========================================
def chatbot_node(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def route_tools(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def execute_tools_node(state: State):
    last_message = state["messages"][-1]
    tool_outputs = []
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "get_weather_forecast":
            # Run the tool which has built-in node-level caching logic
            res = get_weather_forecast.invoke(tool_call["args"])
            tool_outputs.append({
                "role": "tool",
                "content": str(res),
                "tool_call_id": tool_call["id"]
            })
            
    return {"messages": tool_outputs}

# ==========================================
# 6. ASSEMBLE GRAPH WITH THREAD CHECKPOINTING
# ==========================================
workflow = StateGraph(State)

# Add elements
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", execute_tools_node)

workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
workflow.add_edge("tools", "chatbot")

# State & Conversation Caching Layer (Checkpointer)
# In production, replace SqliteSaver with langgraph-checkpoint-redis or MongoDB equivalents.
conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
memory_checkpointer = SqliteSaver(conn)

# Compile graph with persistence
app = workflow.compile(checkpointer=memory_checkpointer)

# Create a unique thread configuration session
config = {"configurable": {"thread_id": "user_session_101"}}

# --- RUN 1 (Fresh Execution) ---
input_message = {"messages": [{"role": "user", "content": "What's the weather like in Paris?"}]}
print("--- RUN 1 ---")
for event in app.stream(input_message, config):
    print(event)

# --- RUN 2 (State Retention & Node Caching) ---
# Asking the same question within the same thread_id will leverage node-level logic and LLM caching.
print("\n--- RUN 2 (Cached Response & Tool) ---")
for event in app.stream(input_message, config):
    print(event)
