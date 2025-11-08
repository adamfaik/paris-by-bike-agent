import os
import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

# Load .env variables
load_dotenv()

# --- 1. The "Switcher" LLM ---
# This intelligently chooses the LLM based on available API keys.
# For deployment on HF Spaces, it will find the GROQ_API_KEY.
# For local use, it will find no key and default to Ollama.
if os.getenv("GROQ_API_KEY"):
    print("LOADED: Using Groq LLM (Cloud)")
    # Using llama-3.3-70b-versatile (current production model as of Nov 2024)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
else:
    print("LOADED: Groq key not found. Using Ollama LLM (Local)")
    llm = ChatOllama(model="llama3", temperature=0)


# --- 2. Define The Tools ---

# Tool 1: The RAG Tool (from the other file)
from rag_pipeline import get_rag_chain
rag_chain = get_rag_chain() # Initialize it once

@tool
def paris_bike_guide(query: str) -> str:
    """
    Use this tool ONLY for specific questions about Paris bike routes, 
    scenic paths, rental shops, repair shops, or local biking rules 
    that are likely found in the curated .md guides.
    """
    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        return f"Error running RAG tool: {e}"

# Tool 2: Web Search (Tavily)
try:
    from langchain_tavily import TavilySearchResults as TavilySearch
    web_search_tool = TavilySearch(
        description="Use this for real-time info like events, opening hours, news, or general web searches."
    )
except ImportError:
    # Fallback to old import if langchain-tavily not installed
    web_search_tool = TavilySearchResults(
        description="Use this for real-time info like events, opening hours, news, or general web searches."
    )

# Tool 3: Weather (Demo)
@tool
def get_weather(location: str) -> str:
    """
    Get the current weather forecast for a given location. 
    Always default to 'Paris, France' if no location is specified.
    """
    if "paris" in location.lower():
        # You can upgrade this later with a real API like OpenWeatherMap
        return "Demo: The forecast for Paris is Sunny with a high of 22°C."
    else:
        return f"Demo: The weather in {location} is clear."

# Tool 4: Biking Time Calculator
@tool
def calculate_biking_time(distance_km: float) -> str:
    """
    Calculates the approximate biking time for a given distance in kilometers.
    Assumes an average casual biking speed of 15 km/h.
    """
    speed_kmh = 15.0
    time_hours = distance_km / speed_kmh
    time_minutes = int(time_hours * 60)
    return f"At a casual pace, a {distance_km} km ride will take approximately {time_minutes} minutes."

# --- 3. Create the Agent ---
tools = [paris_bike_guide, web_search_tool, get_weather, calculate_biking_time]

# This system prompt is the agent's "soul"
SYSTEM_PROMPT = """
You are "Paris by Bike", a friendly and expert assistant for tourists.
You have access to several tools.
1.  `paris_bike_guide`: Your internal knowledge base of curated routes, rental shops, and rules.
2.  `web_search`: For live information (events, hours, news).
3.  `get_weather`: For the latest weather forecast.
4.  `calculate_biking_time`: To estimate travel time for a route.

**Your Logic:**
1.  For questions about specific routes, rentals, or rules, ALWAYS prefer `paris_bike_guide`.
2.  For real-time info (events, hours), use `web_search`.
3.  If a user asks for a recommendation for "today" or "tomorrow", ALWAYS use `get_weather` first.
4.  If you get a distance, use `calculate_biking_time` to add value.
5.  For a simple "Hello", just have a normal conversation.
"""

# Create the agent using langgraph's create_react_agent
# We'll add the system prompt in the message preprocessing instead
agent_executor = create_react_agent(llm, tools)

# Wrapper to inject system prompt into messages
class AgentWithSystemPrompt:
    def __init__(self, agent, system_prompt):
        self.agent = agent
        self.system_prompt = system_prompt
    
    def invoke(self, inputs):
        # Prepend system message if not already present
        messages = inputs.get("messages", [])
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        return self.agent.invoke({"messages": messages})
    
    async def ainvoke(self, inputs):
        # Prepend system message if not already present
        messages = inputs.get("messages", [])
        if not messages or (hasattr(messages[0], 'type') and messages[0].type != "system"):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=self.system_prompt)] + messages
        return await self.agent.ainvoke({"messages": messages})

agent_executor = AgentWithSystemPrompt(agent_executor, SYSTEM_PROMPT)

# --- Main function to test this file directly ---
if __name__ == "__main__":
    print("\n--- Agent Logic Test (Local) ---")
    
    def get_agent_response(user_input, chat_history=[]):
        response = agent_executor.invoke({
            "messages": chat_history + [HumanMessage(content=user_input)]
        })
        # Extract the last AI message
        return response["messages"][-1].content

    chat_history = []
    
    # Test 1
    response1 = get_agent_response("Hello")
    print("Agent:", response1)
    chat_history.append(HumanMessage(content="Hello"))
    chat_history.append(AIMessage(content=response1))
    
    # Test 2
    response2 = get_agent_response("What's a good scenic route for beginners?", chat_history)
    print("Agent:", response2)
    chat_history.append(HumanMessage(content="What's a good scenic route for beginners?"))
    chat_history.append(AIMessage(content=response2))
    
    # Test 3
    response3 = get_agent_response("How long would a 20km ride take me?", chat_history)
    print("Agent:", response3)
    chat_history.append(HumanMessage(content="How long would a 20km ride take me?"))
    chat_history.append(AIMessage(content=response3))
    
    # Test 4
    response4 = get_agent_response("Are there any bike enthusiast meetups in Paris today?", chat_history)
    print("Agent:", response4)