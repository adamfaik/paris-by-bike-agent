import chainlit as cl
from agent_logic import agent_executor
from langchain_core.messages import HumanMessage, AIMessage

@cl.on_chat_start
async def on_chat_start():
    """Initialize the agent and chat history for a new session."""
    
    # Store the agent executor in the user session
    cl.user_session.set("agent_executor", agent_executor)
    
    # Store an empty chat history
    cl.user_session.set("chat_history", [])
    
    await cl.Message(content="Welcome to 'Paris by Bike'! How can I help you plan your trip?").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle a new message from the user."""
    
    # Retrieve the agent and history from the user session
    agent = cl.user_session.get("agent_executor")
    chat_history = cl.user_session.get("chat_history")

    # Add the current user message to history
    current_messages = chat_history + [HumanMessage(content=message.content)]
    
    # Run the agent with langgraph's API
    response = await agent.ainvoke({
        "messages": current_messages
    })
    
    # Extract the output from the last message
    output = response["messages"][-1].content

    # Update the chat history
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=output))
    
    # Send the response back to the user
    await cl.Message(content=output).send()