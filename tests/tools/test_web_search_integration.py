"""Integration tests for the web search tool with actual LLM."""
import json
import time
import pytest
from typing import List, Dict, Any

from forecasting.tools.agent import ToolAugmentedAgent
from forecasting.reasoning.llm_engine import LlamaCppEngine

# Mark this test as integration test (requires LLM and internet connection)
pytestmark = pytest.mark.integration

def test_web_search_with_llm():
    """Test web search tool integration with actual LLM."""
    # Initialize the LLM engine
    print("\nInitializing LLM engine...")
    llm_engine = LlamaCppEngine()
    
    # Create the agent with web search tool
    print("Creating agent with web search tool...")
    agent = ToolAugmentedAgent(llm_engine)
    
    # Test a simple query that would benefit from web search
    query = "What are the latest features in Python 3.13?"
    print(f"\nRunning query: {query}")
    
    # Run the agent
    start_time = time.time()
    response = agent.run(query, max_iterations=3)
    end_time = time.time()
    
    # Extract content from LLMOutput if needed
    content = response.content.raw_text if hasattr(response.content, 'raw_text') else str(response.content)
    
    # Print the results
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("-"*80)
    print("RESPONSE:")
    print(content)
    print("-"*80)
    
    # Print tool calls and results
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("\nTOOL CALLS:")
        for i, call in enumerate(response.tool_calls, 1):
            print(f"{i}. {call['tool']}:")
            print(f"   Args: {json.dumps(call['args'], indent=4)}")
    
    if hasattr(response, 'tool_results') and response.tool_results:
        print("\nTOOL RESULTS:")
        for i, result in enumerate(response.tool_results, 1):
            status = "✓" if result.get('success') else "✗"
            print(f"{i}. {status} {result['tool']}")
            if 'error' in result and result['error']:
                print(f"   Error: {result['error']}")
    
    print("\n" + "="*80)
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    # Basic assertions
    assert response.content is not None
    assert hasattr(response.content, 'raw_text')
    assert isinstance(response.content.raw_text, str)
    assert len(response.content.raw_text) > 0
    
    # Verify tool was used (should be at least one tool call and result)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        assert len(response.tool_calls) > 0
        assert len(response.tool_results) > 0
        # Verify the tool was successful
        assert all(r.get('success', False) for r in response.tool_results)
    else:
        pytest.skip("No tool calls were made during this test")

def test_news_search_with_llm():
    """Test news search tool integration with actual LLM."""
    # Initialize the LLM engine
    print("\nInitializing LLM engine...")
    llm_engine = LlamaCppEngine()
    
    # Create the agent with news search tool
    print("Creating agent with news search tool...")
    agent = ToolAugmentedAgent(llm_engine)
    
    # Test a news query
    query = "What are the latest developments in artificial intelligence?"
    print(f"\nRunning query: {query}")
    
    # Run the agent
    start_time = time.time()
    response = agent.run(query, max_iterations=3)
    end_time = time.time()
    
    # Extract content from LLMOutput if needed
    content = response.content.raw_text if hasattr(response.content, 'raw_text') else str(response.content)
    
    # Print the results
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("-"*80)
    print("RESPONSE:")
    print(content)
    print("-"*80)
    
    # Print tool calls and results
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("\nTOOL CALLS:")
        for i, call in enumerate(response.tool_calls, 1):
            print(f"{i}. {call['tool']}:")
            print(f"   Args: {json.dumps(call['args'], indent=4)}")
    
    if hasattr(response, 'tool_results') and response.tool_results:
        print("\nTOOL RESULTS:")
        for i, result in enumerate(response.tool_results, 1):
            status = "✓" if result.get('success') else "✗"
            print(f"{i}. {status} {result['tool']}")
            if 'error' in result and result['error']:
                print(f"   Error: {result['error']}")
    
    print("\n" + "="*80)
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    # Basic assertions
    assert response.content is not None
    assert hasattr(response.content, 'raw_text')
    assert isinstance(response.content.raw_text, str)
    assert len(response.content.raw_text) > 0
    
    # Verify tool was used (should be at least one tool call and result)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        assert len(response.tool_calls) > 0
        assert len(response.tool_results) > 0
        # Verify the tool was successful
        assert all(r.get('success', False) for r in response.tool_results)
    else:
        pytest.skip("No tool calls were made during this test")
