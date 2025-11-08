import chainlit as cl
from agent_logic import agent_executor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import asyncio
import re

@cl.on_chat_start
async def on_chat_start():
    """Initialize the agent and chat history for a new session."""
    
    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("chat_history", [])

@cl.set_starters
async def set_starters():
    """Return the list of starter prompts."""
    return [
        cl.Starter(
            label="🗺️ Route for Beginners",
            message="What's a good scenic route for beginners in Paris?",
        ),
        cl.Starter(
            label="⏱️ 1 Hour Ride Plan",
            message="I'm at Paris 17e and have 1h to bike, what's your advice?",
        ),
        cl.Starter(
            label="📍 Eiffel Tower to Montmartre",
            message="Show me the bike route from Eiffel Tower to Montmartre",
        ),
        cl.Starter(
            label="🚲 Find Bike Rentals",
            message="Where can I rent a bike near the Louvre?",
        ),
    ]

@cl.on_message
async def on_message(message: cl.Message):
    """Handle a new message from the user."""
    
    max_retries = 3
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            agent = cl.user_session.get("agent_executor")
            chat_history = cl.user_session.get("chat_history")
            
            if not agent:
                await cl.Message(content="⚠️ Session error. Please refresh the page.").send()
                return

            # Keep only system message + last 3 exchanges (6 messages) to focus on current context
            # This prevents the agent from re-processing the entire conversation
            relevant_history = []
            if len(chat_history) > 6:
                relevant_history = chat_history[-6:]  # Last 3 exchanges
            else:
                relevant_history = chat_history
            
            current_messages = relevant_history + [HumanMessage(content=message.content)]
            
            # Create a parent step for the entire agent process
            step_name = "🤔 Thinking..." if attempt == 0 else f"🔄 Retry {attempt}/{max_retries}..."
            async with cl.Step(name=step_name, type="llm") as agent_step:
                response = await agent.ainvoke({"messages": current_messages})
                all_messages = response["messages"]
                
                # Track tool usage
                tool_count = 0
                tools_used = []
                
                for msg in all_messages[len(current_messages):]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_count += 1
                            tool_name = tool_call.get('name', 'unknown')
                            tools_used.append(tool_name)
                            tool_args = tool_call.get('args', {})
                            
                            async with cl.Step(
                                name=f"🔧 {tool_name}",
                                type="tool",
                                parent_id=agent_step.id
                            ) as tool_step:
                                # Show shortened args for readability
                                args_str = str(tool_args)
                                tool_step.input = args_str[:200] + "..." if len(args_str) > 200 else args_str
                    
                    elif isinstance(msg, ToolMessage):
                        tool_output = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                        async with cl.Step(
                            name=f"✅ Result",
                            type="tool",
                            parent_id=agent_step.id
                        ) as result_step:
                            result_step.output = tool_output
                
                output = all_messages[-1].content
                
                # Add tool usage summary
                if tools_used:
                    tools_summary = ", ".join(set(tools_used))
                    agent_step.output = f"Used: {tools_summary}"
                else:
                    agent_step.output = "Direct response"

            # Update the chat history with full history
            chat_history.append(HumanMessage(content=message.content))
            chat_history.append(AIMessage(content=output))
            
            # Keep only last 20 messages (10 exchanges) to prevent memory issues
            MAX_HISTORY = 20
            if len(chat_history) > MAX_HISTORY:
                chat_history = chat_history[-MAX_HISTORY:]
            
            cl.user_session.set("chat_history", chat_history)
            
            # Send the response back to the user
            await cl.Message(content=output).send()
            
            # Success - exit retry loop
            return
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_str or "rate_limit" in error_str.lower() or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    
                    try:
                        match = re.search(r'try again in (\d+)m(\d+)', error_str)
                        if match:
                            minutes = int(match.group(1))
                            seconds = int(match.group(2))
                            wait_time = minutes * 60 + seconds
                    except:
                        pass
                    
                    await cl.Message(
                        content=f"⏳ Rate limit reached. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}..."
                    ).send()
                    
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_message = (
                        "🚦 **Rate Limit Reached**\n\n"
                        "The AI service needs a short break. Please:\n"
                        "- Wait 1-2 minutes and try again\n"
                        "- Refresh the page for a clean start\n"
                        "- Consider using Ollama locally for unlimited requests\n"
                    )
                    await cl.Message(content=error_message).send()
                    return
            else:
                error_message = (
                    f"❌ **Error occurred:**\n\n"
                    f"```\n{error_str}\n```\n\n"
                    "Please try rephrasing your question or refresh the page."
                )
                await cl.Message(content=error_message).send()
                print(f"Error in on_message: {e}")
                import traceback
                traceback.print_exc()
                return