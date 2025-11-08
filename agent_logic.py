import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool, StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta

# Load .env variables
load_dotenv()

# --- Environment Validation ---
def validate_environment():
    """Check and report on environment configuration."""
    issues = []
    warnings = []
    
    if os.getenv("GROQ_API_KEY"):
        print("✅ Groq API key found (Cloud LLM)")
    else:
        warnings.append("⚠️  No Groq key - using Ollama (ensure it's running: `ollama serve`)")
    
    if not os.getenv("OPENWEATHERMAP_API_KEY"):
        warnings.append("⚠️  No OpenWeatherMap key - weather will use demo data")
    else:
        print("✅ OpenWeatherMap API key found")
    
    if not os.getenv("TAVILY_API_KEY"):
        warnings.append("⚠️  No Tavily key - web search may be limited")
    else:
        print("✅ Tavily API key found")
    
    if not os.path.exists("data"):
        issues.append("❌ 'data' folder not found - RAG will fail. Create it and add .md/.pdf files")
    elif not any(Path("data").rglob("*.md")) and not any(Path("data").rglob("*.pdf")):
        issues.append("❌ No .md or .pdf files in 'data' folder - add guide documents")
    else:
        print("✅ Data folder found with documents")
    
    if warnings:
        print("\n" + "\n".join(warnings))
    
    if issues:
        print("\n⚠️  CRITICAL ISSUES:")
        print("\n".join(issues))
        print("\nSome features may not work correctly.\n")
    else:
        print("✅ All systems ready!\n")

validate_environment()

# --- 1. The "Switcher" LLM ---
if os.getenv("GROQ_API_KEY"):
    print("LOADED: Using Groq LLM (Cloud)")
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
else:
    print("LOADED: Groq key not found. Using Ollama LLM (Local)")
    llm = ChatOllama(model="llama3.1", temperature=0.3)


# --- 2. Define The Tools ---

# Tool 1: The RAG Tool
from rag_pipeline import get_rag_chain
rag_chain = get_rag_chain()

@tool
def paris_bike_guide(query: str) -> str:
    """Search curated Paris bike guides for routes, monuments, rental shops, repair shops, and biking rules.
    Use this to answer specific questions about routes, locations, or biking information in Paris."""
    try:
        response = rag_chain(query)
        return response
    except Exception as e:
        return f"Error running RAG tool: {e}"

# Tool 2: Web Search with links
class WebSearchInput(BaseModel):
    """Input for web search."""
    query: str = Field(description="Search query for events, shops, or current information")

def web_search_function(query: str) -> str:
    """Search the web for current events, shops, and information. Returns results with clickable links."""
    try:
        # Use the Tavily search
        from langchain_tavily import TavilySearchResults as TavilySearch
        search_tool = TavilySearch(max_results=3)
        
        results = search_tool.invoke(query)
        
        if not results:
            return "No results found for this search."
        
        # Format results with links
        formatted_results = []
        for i, result in enumerate(results[:3], 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            content = result.get('content', '')
            
            # Truncate content for readability
            if len(content) > 200:
                content = content[:200] + "..."
            
            formatted = f"{i}. **{title}**\n"
            formatted += f"   {content}\n"
            if url:
                formatted += f"   🔗 Link: {url}\n"
            
            formatted_results.append(formatted)
        
        return "\n".join(formatted_results)
        
    except ImportError:
        # Fallback to old search
        from langchain_community.tools.tavily_search import TavilySearchResults
        search_tool = TavilySearchResults(max_results=3)
        results = search_tool.invoke(query)
        
        formatted_results = []
        for i, result in enumerate(results[:3], 1):
            if isinstance(result, dict):
                title = result.get('title', 'Result')
                url = result.get('url', '')
                content = result.get('content', str(result))
                
                if len(content) > 200:
                    content = content[:200] + "..."
                
                formatted = f"{i}. **{title}**\n   {content}\n"
                if url:
                    formatted += f"   🔗 {url}\n"
                formatted_results.append(formatted)
            else:
                formatted_results.append(f"{i}. {str(result)[:200]}\n")
        
        return "\n".join(formatted_results) if formatted_results else "No results found."
    
    except Exception as e:
        return f"Search error: {str(e)}"

web_search_tool = StructuredTool.from_function(
    func=web_search_function,
    name="web_search",
    description="Search for current events, cycling meetups, bike shop information with URLs. Use ONLY when user asks about events, current happenings, or specific businesses. Always include the URLs from results in your response.",
    args_schema=WebSearchInput,
    return_direct=False
)

# Tool 3: Weather with hourly forecast
class WeatherInput(BaseModel):
    """Input for weather tool."""
    location: str = Field(
        description="City name to get weather for. Examples: 'Paris', 'Lyon', 'Marseille'"
    )

def get_weather_function(location: str) -> str:
    """Get current weather and upcoming hourly forecast to help plan bike rides and avoid rain."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    
    if not location or not location.strip():
        location = "Paris"
    
    location = location.strip()
    
    if not api_key:
        current_hour = datetime.now().hour
        return (
            f"Weather for {location} (Demo data):\n\n"
            f"Current: Sunny, 22°C\n"
            f"{current_hour+1}:00: Partly cloudy, 21°C\n"
            f"{current_hour+2}:00: Cloudy, 20°C\n"
            f"{current_hour+3}:00: Light rain expected, 19°C\n\n"
            f"Best time to ride: Now to {current_hour+2}:00 before the rain.\n"
            f"(Add OPENWEATHERMAP_API_KEY to .env for real weather data)"
        )
    
    try:
        if "," not in location:
            location = f"{location}, France"
        
        current_url = "http://api.openweathermap.org/data/2.5/weather"
        current_params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
            "lang": "en"
        }
        
        current_response = requests.get(current_url, params=current_params, timeout=10)
        
        if current_response.status_code == 404:
            return f"Couldn't find weather for '{location}'"
        
        current_response.raise_for_status()
        current_data = current_response.json()
        
        forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
            "cnt": 6
        }
        
        forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        city = current_data["name"]
        temp = round(current_data["main"]["temp"], 1)
        feels_like = round(current_data["main"]["feels_like"], 1)
        description = current_data["weather"][0]["description"].capitalize()
        wind_speed = round(current_data["wind"]["speed"] * 3.6, 1)
        
        result = f"Weather for {city}:\n\n"
        result += f"Current conditions:\n"
        result += f"- Temperature: {temp}°C (feels like {feels_like}°C)\n"
        result += f"- Conditions: {description}\n"
        result += f"- Wind: {wind_speed} km/h\n\n"
        
        result += "Next few hours:\n"
        rain_times = []
        
        for i, item in enumerate(forecast_data['list'][:4]):
            dt = datetime.fromtimestamp(item['dt'])
            hour_temp = round(item['main']['temp'], 1)
            hour_desc = item['weather'][0]['description']
            hour_time = dt.strftime("%H:%M")
            
            if "rain" in hour_desc.lower():
                rain_times.append(hour_time)
            
            result += f"- {hour_time}: {hour_temp}°C, {hour_desc}\n"
        
        if rain_times:
            result += f"\nRain expected around {', '.join(rain_times[:2])}. Plan accordingly."
        elif temp > 30:
            result += f"\nHot conditions. Bring extra water."
        elif temp < 5:
            result += f"\nCold conditions. Dress warmly."
        else:
            result += f"\nGood conditions for biking."
        
        return result
        
    except Exception as e:
        return f"Weather error: {str(e)}"

get_weather = StructuredTool.from_function(
    func=get_weather_function,
    name="get_weather",
    description="Get current weather and hourly forecast. Use only when the user asks about weather or mentions timing (today, this afternoon, tomorrow).",
    args_schema=WeatherInput,
    return_direct=False
)

# Tool 4: Biking Time & Distance Calculator
class BikingPlanInput(BaseModel):
    """Input for biking planner."""
    duration_minutes: Optional[int] = Field(
        default=None,
        description="Desired ride duration in minutes (e.g., 60 for 1 hour)"
    )
    distance_km: Optional[float] = Field(
        default=None,
        description="Distance in kilometers if known"
    )

def calculate_biking_plan_function(duration_minutes: Optional[int] = None, distance_km: Optional[float] = None) -> str:
    """Calculate distance from duration or time from distance. Helps plan rides based on available time."""
    speed_kmh = 15.0
    
    if duration_minutes and duration_minutes > 0:
        hours = duration_minutes / 60
        distance = round(hours * speed_kmh, 1)
        
        result = f"For {duration_minutes} minutes ({duration_minutes//60}h {duration_minutes%60}min) at casual pace:\n"
        result += f"- Total distance: approximately {distance} km\n"
        result += f"- One-way distance (with return): {distance/2} km\n"
        return result
    
    elif distance_km and distance_km > 0:
        if distance_km > 200:
            return "That's a very long ride (200+ km). Consider breaking it into multiple days."
        
        time_hours = distance_km / speed_kmh
        time_minutes = int(time_hours * 60)
        hours = time_minutes // 60
        minutes = time_minutes % 60
        
        result = f"For a {distance_km} km ride at casual pace:\n"
        if hours > 0:
            result += f"- Time needed: approximately {hours}h {minutes}min\n"
        else:
            result += f"- Time needed: approximately {minutes} minutes\n"
        
        if distance_km < 5:
            result += f"- Difficulty: Easy ride\n"
        elif distance_km < 15:
            result += f"- Difficulty: Moderate ride\n"
        elif distance_km < 30:
            result += f"- Difficulty: Good workout\n"
        else:
            result += f"- Difficulty: Long ride, bring snacks\n"
        
        return result
    else:
        return "Please specify either duration (e.g., 60 minutes) or distance (e.g., 10 km)."

calculate_biking_plan = StructuredTool.from_function(
    func=calculate_biking_plan_function,
    name="calculate_biking_plan",
    description="Calculate ride distance from duration or time from distance. Use when user mentions specific time or distance.",
    args_schema=BikingPlanInput,
    return_direct=False
)

# Tool 5: Generate Google Maps cycling route URL
class RouteMapInput(BaseModel):
    """Input for route map generator."""
    start_location: str = Field(description="Starting point (e.g., 'Paris 17e', 'Eiffel Tower')")
    end_location: str = Field(description="Destination (e.g., 'Montmartre', 'Notre-Dame')")
    waypoints: Optional[str] = Field(default=None, description="Optional intermediate stops separated by |")

def generate_route_map_function(start_location: str, end_location: str, waypoints: Optional[str] = None) -> str:
    """Generate a Google Maps cycling route URL for the user to visualize and navigate the route."""
    try:
        start = start_location.strip()
        end = end_location.strip()
        
        if "paris" not in start.lower() and not any(x in start.lower() for x in ["tour eiffel", "louvre", "montmartre", "notre"]):
            start = f"{start}, Paris"
        if "paris" not in end.lower() and not any(x in end.lower() for x in ["tour eiffel", "louvre", "montmartre", "notre"]):
            end = f"{end}, Paris"
        
        maps_url = f"https://www.google.com/maps/dir/?api=1&origin={start.replace(' ', '+')}&destination={end.replace(' ', '+')}&travelmode=bicycling"
        
        if waypoints:
            maps_url += f"&waypoints={waypoints.replace(' ', '+')}"
        
        result = f"Route map from {start} to {end}:\n"
        result += f"View cycling route: {maps_url}\n\n"
        result += f"The map includes:\n"
        result += f"- Bike-friendly routes and bike lanes\n"
        result += f"- Elevation changes\n"
        result += f"- Estimated distance and time\n"
        result += f"- Turn-by-turn GPS navigation (on mobile)\n"
        
        return result
    except Exception as e:
        return f"Error generating map: {str(e)}"

generate_route_map = StructuredTool.from_function(
    func=generate_route_map_function,
    name="generate_route_map",
    description="Generate a Google Maps cycling route. Use when user asks for a route between specific locations or after suggesting a route from paris_bike_guide.",
    args_schema=RouteMapInput,
    return_direct=False
)

# --- 3. Create the Agent ---
tools = [paris_bike_guide, web_search_tool, get_weather, calculate_biking_plan, generate_route_map]

SYSTEM_PROMPT = """You are "Paris by Bike" - a knowledgeable and friendly biking assistant for Paris.

CRITICAL RULES FOR TOOL USAGE:

1. MATCH YOUR RESPONSE TO THE QUESTION COMPLEXITY:
   - Simple greeting ("hello", "hi") → Respond naturally, NO TOOLS
   - Specific question ("what's a good route for beginners?") → Use ONLY paris_bike_guide, answer the question directly
   - Complex planning ("I have 1h from Paris 17e, what should I do?") → Use multiple tools for complete plan

2. WHEN TO USE EACH TOOL:
   - paris_bike_guide: For routes, shops, monuments, biking rules - PRIMARY tool for most questions
   - web_search: ONLY when user asks about events, current happenings, or specific business info. Always include URLs from results
   - get_weather: ONLY when user mentions timing (today, tomorrow, this afternoon) or explicitly asks
   - calculate_biking_plan: ONLY when user mentions duration or distance
   - generate_route_map: ONLY after suggesting a route OR when user explicitly requests directions

3. RESPONSE STRUCTURE (for complex queries):
   - Start with brief friendly intro (1-2 sentences)
   - Use clear section headers with ONE emoji each: "🗺️ Route Suggestion:", "🌤️ Weather:", etc.
   - Present information in bullet points for readability
   - End with "What would you like to know next?" and 1-2 suggested follow-up questions

4. CONVERSATION FLOW:
   - Focus on the CURRENT user message, not previous conversation
   - Answer what was just asked, then suggest related actions
   - Don't repeat information from previous responses

5. WEB SEARCH RESULTS:
   - Always include clickable URLs for events, shops, or businesses
   - Format: "[Event Name](URL)" or "Visit: URL"

6. ROUTE MAPS:
   - When generating maps, consider the user's stated duration/distance
   - For 1h ride = ~15km total or 7.5km one-way
   - For 2h ride = ~30km total or 15km one-way
   - Suggest waypoints to match the desired duration

7. EMOJI USAGE:
   - One emoji per section header only
   - No emojis in regular text or bullet points
   - Keep it professional and readable

EXAMPLE RESPONSES:

User: "Hello"
You: "Hello! I'm here to help you explore Paris by bike. What are you interested in today?"

User: "What's a good scenic route for beginners?"
You: [Use paris_bike_guide ONLY]
"Here's a great option for beginners:

🗺️ Recommended Route:
- [Route details from RAG]
- Distance: X km
- Highlights: [landmarks]

Would you like me to create a map for this route, or check the weather for today?"

User: "I'm at Paris 17e and have 1h to bike, what's your advice?"
You: [Use calculate_biking_plan + paris_bike_guide + get_weather + generate_route_map]
"Perfect! Let me plan a great 1-hour ride from Paris 17e.

⏱️ Time & Distance:
- In 1 hour, you can cover about 15 km total
- Or 7.5 km one-way if you want to return

🗺️ Route Suggestion:
[Route from RAG matching the distance]

🌤️ Weather:
[Current conditions and forecast]

🗺️ Interactive Map:
[Google Maps URL]

What would you like to know next?
- Need bike rental locations nearby?
- Want to know about upcoming cycling events?"

Remember: Match complexity to the question. Simple questions get simple answers."""

# Create agent
agent_executor = create_react_agent(llm, tools)

# Wrapper for system prompt
class AgentWithSystemPrompt:
    def __init__(self, agent, system_prompt):
        self.agent = agent
        self.system_prompt = system_prompt
    
    def invoke(self, inputs):
        messages = inputs.get("messages", [])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        return self.agent.invoke({"messages": messages})
    
    async def ainvoke(self, inputs):
        messages = inputs.get("messages", [])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        return await self.agent.ainvoke({"messages": messages})

agent_executor = AgentWithSystemPrompt(agent_executor, SYSTEM_PROMPT)

# --- Test function ---
if __name__ == "__main__":
    print("\n--- Testing Agent ---")
    
    print("\n🧪 TEST 1: Simple greeting")
    response = agent_executor.invoke({
        "messages": [HumanMessage(content="Hello")]
    })
    print("="*60)
    print(response["messages"][-1].content)
    print("="*60)
    
    print("\n🧪 TEST 2: Specific route question")
    response = agent_executor.invoke({
        "messages": [HumanMessage(content="What's a good scenic route for beginners?")]
    })
    print("="*60)
    print(response["messages"][-1].content)
    print("="*60)