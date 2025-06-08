import json
from typing import Dict, Any, List, Optional, Type, TypeVar, Generic
from pydantic import BaseModel, ValidationError
from . import Tool, ToolRegistry, ToolCall, ToolResult

T = TypeVar('T')

class ToolInvocationError(Exception):
    """Raised when there's an error invoking a tool."""
    pass

class ToolManager:
    """Manages tool execution and maintains conversation state with the LLM."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
    
    def get_tools_system_prompt(self) -> str:
        """Generate the system prompt describing available tools."""
        tools = self.tool_registry.get_tools_descriptions()
        if not tools:
            return ""
            
        prompt = [
            "You have access to the following tools. Use them if needed to answer the user's question.",
            "When using a tool, respond with a JSON object containing the following fields:",
            "- 'tool': The name of the tool to use.",
            "- 'args': A dictionary of arguments for the tool.",
            "\nAvailable tools:"
        ]
        
        for tool in tools:
            tool_desc = [
                f"{tool['name']}: {tool['description']}",
                f"  Arguments:"
            ]
            
            if 'args' in tool and 'properties' in tool['args']:
                for arg_name, arg_schema in tool['args']['properties'].items():
                    arg_type = arg_schema.get('type', 'any')
                    arg_desc = arg_schema.get('description', 'No description')
                    default = f" (default: {arg_schema['default']})" if 'default' in arg_schema else ""
                    tool_desc.append(f"  - {arg_name} ({arg_type}){default}: {arg_desc}")
            
            prompt.append("\n".join(tool_desc))
        
        prompt.extend([
            "\nExample usage:",
            '```json',
            '{"tool": "web_search", "args": {"query": "latest news about AI", "max_results": 3}}',
            '```',
            '\nAfter using a tool, the results will be provided to you. You can then use this information to answer the user\'s question.'
        ])
        
        return "\n".join(prompt)
    
    def parse_tool_call(self, llm_output) -> Optional[ToolCall]:
        """Parse the LLM's output to extract a tool call.
        
        Args:
            llm_output: The output from the LLM, either a string or LLMOutput object
            
        Returns:
            A ToolCall if a valid tool call is found, None otherwise.
        """
        # Extract the raw text from LLMOutput if it's an object
        if hasattr(llm_output, 'raw_text'):
            output_text = llm_output.raw_text
        else:
            output_text = str(llm_output)
            
        # Try to find JSON in the output
        try:
            # Look for code blocks first
            if '```json' in output_text:
                json_str = output_text.split('```json')[1].split('```')[0].strip()
            elif '```' in output_text:
                # Try any code block
                json_str = output_text.split('```')[1].split('```')[0].strip()
            else:
                # Try to find JSON in the text
                json_str = output_text.strip()
                
            # Try to parse as JSON
            data = json.loads(json_str)
            
            if not isinstance(data, dict) or 'tool' not in data:
                return None
                
            return ToolCall(
                tool_name=data['tool'],
                args=data.get('args', {})
            )
            
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult[Any]:
        """Execute a tool call.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            The result of the tool execution
            
        Raises:
            ToolInvocationError: If the tool is not found or there's an error
        """
        tool = self.tool_registry.get_tool(tool_call.tool_name)
        if not tool:
            raise ToolInvocationError(f"Unknown tool: {tool_call.tool_name}")
        
        try:
            # Execute the tool with the provided arguments
            return tool.execute(**tool_call.args)
        except Exception as e:
            raise ToolInvocationError(f"Error executing tool {tool_call.tool_name}: {str(e)}")
    
    def format_tool_result(self, tool_name: str, result: ToolResult[Any]) -> str:
        """Format a tool result for the LLM."""
        if not result.success:
            return f"Error using {tool_name}: {result.error}"
        
        if not result.result:
            return f"No results from {tool_name}."
            
        if isinstance(result.result, list):
            if not result.result:
                return f"No results from {tool_name}."
                
            formatted = [f"Results from {tool_name}:"]
            for i, item in enumerate(result.result, 1):
                if isinstance(item, dict):
                    item_str = []
                    for k, v in item.items():
                        if v:  # Skip empty values
                            item_str.append(f"{k}: {v}")
                    formatted.append(f"{i}. " + " | ".join(item_str))
                else:
                    formatted.append(f"{i}. {item}")
            return "\n".join(formatted)
        elif isinstance(result.result, dict):
            formatted = [f"Results from {tool_name}:"]
            for k, v in result.result.items():
                formatted.append(f"{k}: {v}")
            return "\n".join(formatted)
        else:
            return f"Result from {tool_name}: {result.result}"
