#!/usr/bin/env python3
"""
Test script to verify the streaming callback fix for ChatGoogleGenerativeAI.

This script tests that the streaming callback now properly receives tokens
when using ChatGoogleGenerativeAI with streaming enabled.
"""

import os
import sys
from typing import List

# Add the libs directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'genai'))

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI


class TestStreamingCallback(BaseCallbackHandler):
    """Test callback handler to verify streaming tokens are received."""
    
    def __init__(self):
        self.tokens_received = []
        self.llm_start_called = False
        self.llm_new_token_called = False
    
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """Called when LLM starts."""
        self.llm_start_called = True
        print(f"✓ LLM started with {len(prompts)} prompt(s)")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.llm_new_token_called = True
        self.tokens_received.append(token)
        print(f"✓ Received token: '{token}'")
    
    def get_summary(self) -> dict:
        """Get a summary of the callback execution."""
        return {
            "llm_start_called": self.llm_start_called,
            "llm_new_token_called": self.llm_new_token_called,
            "total_tokens": len(self.tokens_received),
            "tokens": self.tokens_received
        }


def test_streaming_callback():
    """Test that streaming callbacks work correctly."""
    print("Testing streaming callback fix for ChatGoogleGenerativeAI...")
    
    # Check if we have the required API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set. Skipping test.")
        return False
    
    try:
        # Create the callback handler
        callback = TestStreamingCallback()
        
        # Create the LLM with streaming enabled
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            model_kwargs={"streaming": True},
            callbacks=[callback]
        )
        
        print("✓ Created ChatGoogleGenerativeAI with streaming enabled")
        
        # Test with a simple message
        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(content="Say 'Hello World' in exactly 3 words.")
        
        print("✓ Created test message")
        
        # Invoke the LLM
        print("Invoking LLM...")
        result = llm.invoke([message])
        
        print("✓ LLM invocation completed")
        
        # Check the results
        summary = callback.get_summary()
        
        print(f"\nTest Results:")
        print(f"  LLM start called: {summary['llm_start_called']}")
        print(f"  New token called: {summary['llm_new_token_called']}")
        print(f"  Total tokens received: {summary['total_tokens']}")
        print(f"  Response content: {result.content}")
        
        # Verify the fix worked
        if summary['llm_new_token_called'] and summary['total_tokens'] > 0:
            print("\n✅ SUCCESS: Streaming callback fix is working correctly!")
            print("   Tokens are being properly passed to the callback handler.")
            return True
        else:
            print("\n❌ FAILURE: Streaming callback fix is not working.")
            print("   No tokens were received by the callback handler.")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_toolmessage_support():
    """Test that ToolMessage support works correctly."""
    print("\nTesting ToolMessage support fix...")
    
    try:
        from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
        
        # Create a test conversation with ToolMessage
        messages = [
            HumanMessage(content="What's the weather like?"),
            AIMessage(content="I need to check the weather for you.", tool_calls=[{
                "name": "get_weather",
                "args": {"location": "New York"},
                "id": "call_123"
            }]),
            ToolMessage(content="It's sunny and 72°F", tool_call_id="call_123", name="get_weather"),
            HumanMessage(content="Thanks!")
        ]
        
        # This should not raise an error anymore
        from langchain_google_genai.chat_models import _parse_chat_history
        
        system_instruction, parsed_messages = _parse_chat_history(messages)
        
        print(f"✓ Successfully parsed {len(messages)} messages including ToolMessage")
        print(f"✓ Parsed into {len(parsed_messages)} Google Content objects")
        
        # Check that ToolMessage was processed
        tool_message_found = any(
            any(part.function_response for part in content.parts if hasattr(part, 'function_response'))
            for content in parsed_messages
        )
        
        if tool_message_found:
            print("✅ SUCCESS: ToolMessage support is working correctly!")
            return True
        else:
            print("❌ FAILURE: ToolMessage was not properly processed.")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: ToolMessage test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LangChain Google GenAI Fixes")
    print("=" * 60)
    
    # Test streaming callback fix
    streaming_success = test_streaming_callback()
    
    # Test ToolMessage support fix
    toolmessage_success = test_toolmessage_support()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Streaming Callback Fix: {'✅ PASS' if streaming_success else '❌ FAIL'}")
    print(f"  ToolMessage Support Fix: {'✅ PASS' if toolmessage_success else '❌ FAIL'}")
    
    if streaming_success and toolmessage_success:
        print("\n🎉 All tests passed! The fixes are working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        sys.exit(1)
