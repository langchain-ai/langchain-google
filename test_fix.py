import os
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
# Make sure to import the correct Chat Model you are using (VertexAI or GenAI)
# Based on your error message, we'll use ChatVertexAI:
from langchain_google_vertexai import ChatVertexAI

# --- 1. Define a Simple Tool ---
@tool
def calculate_sum(a: int, b: int) -> str:
    """Returns the sum of two integers."""
    return str(a + b)

# --- 2. Initialize the Model ---
# Ensure your authentication (e.g., gcloud auth application-default login) is set up.
# Use the correct model name (e.g., gemini-2.5-flash is good for testing).
try:
    llm = ChatVertexAI(model="gemini-2.5-flash",  project="langchain-gemini-development" ).bind_tools([calculate_sum])
except Exception as e:
    print("Model initialization failed. Check your Vertex AI connection/credentials.")
    print(f"Error: {e}")
    exit()

# --- 3. Construct the Tool-Use Conversation History ---
user_query = "Calculate 10 + 5."
print(f"-> User Query: {user_query}")

# A. Model asks for a tool call
print("-> Step A: Invoking LLM to get the tool call...")
ai_msg = llm.invoke([HumanMessage(user_query)])

# Check if a tool call was made (necessary for the next step)
if not ai_msg.tool_calls:
    print("! Model did not request a tool call. Stopping test.")
    print(f"Model response: {ai_msg.content}")
    exit()

tool_call = ai_msg.tool_calls[0]
tool_call_id = tool_call["id"]
tool_name = tool_call["name"]
tool_args = tool_call["args"]

# B. Application executes the tool
print(f"-> Step B: Executing Tool: {tool_name}({tool_args})")
tool_result_content = calculate_sum(**tool_args)

# C. Create the ToolMessage (this is the message that causes the bug when role='function')
tool_message = ToolMessage(
    content=tool_result_content,
    tool_call_id=tool_call_id,
    name=tool_name
)

# The full conversation history
fixed_history = [
    HumanMessage(user_query),
    ai_msg,
    tool_message  # The fixed message is inserted here
]

# --- 4. RUN THE TEST (Expect SUCCESS after the fix) ---
print("-> Step C: Sending Tool Result back to LLM to generate final response...")

try:
    final_response = llm.invoke(fixed_history)
    
    # If this line is reached, the API accepted the fixed message role!
    print("\n✅ SUCCESS: The ToolMessage role bug is fixed.")
    print(f"Final LLM Response: {final_response.content}")

except Exception as e:
    # If the BadRequest 400 error is still here, the fix was incomplete.
    print(f"\n❌ FAILURE: The error persists.")
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR MESSAGE (Check for 'valid role: user, model'): {e}")