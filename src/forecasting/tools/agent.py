from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel
import json

from . import ToolRegistry
from .manager import ToolManager
from .web_search import WebSearchTool, NewsSearchTool
from ..reasoning.llm_engine import LlmEngineABC

@dataclass
class AgentResponse:
    """Structured response from the agent."""
    content: str
    tool_calls: List[Dict[str, Any]] = None
    tool_results: List[Dict[str, Any]] = None

class ToolAugmentedAgent:
    """An agent that can use tools to augment its capabilities."""
    
    def __init__(self, llm_engine: LlmEngineABC):
        """Initialize the agent with an LLM engine."""
        self.llm = llm_engine
        
        # Set up tools
        self.tool_registry = ToolRegistry()
        self.tool_manager = ToolManager(self.tool_registry)
        
        # Register default tools
        self.register_default_tools()
    
    def register_default_tools(self) -> None:
        """Register default tools with the agent."""
        self.tool_registry.register(WebSearchTool())
        self.tool_registry.register(NewsSearchTool())
        # Add more default tools here
    
    def register_tool(self, tool) -> None:
        """Register a new tool with the agent."""
        self.tool_registry.register(tool)
    
    def generate_system_prompt(self) -> str:
        """Generate the system prompt for the LLM."""
        return (
            "You are a helpful AI assistant with access to tools. "
            "Use the tools when needed to answer the user's questions.\n\n"
            + self.tool_manager.get_tools_system_prompt()
        )
    
    def run(self, user_input: str, max_iterations: int = 3) -> AgentResponse:
        """Run the agent with the given user input.
        
        Args:
            user_input: The user's input/query
            max_iterations: Maximum number of tool iterations to perform
            
        Returns:
            An AgentResponse with the final response and any tool usage
        """
        system_prompt = self.generate_system_prompt()
        
        # Initialize conversation history
        conversation = [
            f"""{system_prompt}
            
User: {user_input}

Assistant:"""
        ]
        
        tool_calls = []
        tool_results = []

        for _ in range(max_iterations):
            # Build the full prompt with conversation history
            full_prompt = "\n\n".join(conversation)
            
            # Get the LLM's response
            llm_response = self.llm.generate(prompt=full_prompt)
            
            # Try to parse a tool call from the response
            tool_call = self.tool_manager.parse_tool_call(llm_response)
            
            if not tool_call:
                # No tool call, we're done
                return AgentResponse(
                    content=llm_response,
                    tool_calls=tool_calls,
                    tool_results=tool_results
                )
            
            # Execute the tool
            tool_calls.append({
                "tool": tool_call.tool_name,
                "args": tool_call.args
            })
            
            try:
                result = self.tool_manager.execute_tool(tool_call)
                tool_results.append({
                    "tool": tool_call.tool_name,
                    "success": result.success,
                    "result": str(result.result)[:500] + "..." if result.result else None,
                    "error": result.error
                })
                
                # Format the result for the LLM
                result_text = self.tool_manager.format_tool_result(
                    tool_call.tool_name, result
                )
                
                # Add the tool call and result to the conversation
                conversation.extend([
                    f"I'll use the {tool_call.tool_name} tool with args: {tool_call.args}",
                    f"Tool result: {result_text}",
                    "Assistant:"
                ])
                
            except Exception as e:
                error_msg = f"Error executing tool {tool_call.tool_name}: {str(e)}"
                conversation.extend([
                    f"Error: {error_msg}",
                    "Assistant:"
                ])
                tool_results.append({
                    "tool": tool_call.tool_name,
                    "success": False,
                    "error": str(e)
                })
        
        # If we've reached max iterations, get a final response
        final_response = self.llm.generate(messages=messages)
        return AgentResponse(
            content=final_response,
            tool_calls=tool_calls,
            tool_results=tool_results
        )


# Example usage
def example_usage():
    """Example of how to use the ToolAugmentedAgent."""
    from forecasting.reasoning.llm_engine import LlamaCppEngine
    
    # Initialize the LLM engine
    llm_engine = LlamaCppEngine()  # Configure as needed
    
    # Create the agent
    agent = ToolAugmentedAgent(llm_engine)
    
    # Run a query
    response = agent.run(
        "What are the latest developments in AI?"
    )
    
    print("\nFinal Response:")
    print(response.content)
    
    if response.tool_calls:
        print("\nTool Calls:")
        for call in response.tool_calls:
            print(f"- {call['tool']}: {call['args']}")
    
    if response.tool_results:
        print("\nTool Results:")
        for result in response.tool_results:
            status = "✓" if result.get('success') else "✗"
            print(f"{status} {result['tool']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")


if __name__ == "__main__":
    example_usage()
