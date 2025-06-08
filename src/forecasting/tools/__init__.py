from typing import Protocol, Dict, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass
from pydantic import BaseModel

T = TypeVar('T')

class Tool(Protocol[T]):
    """Base interface for all tools that can be used by the LLM."""
    
    @property
    def name(self) -> str:
        """Name of the tool."""
        ...
        
    @property
    def description(self) -> str:
        """Description of what the tool does."""
        ...
        
    @property
    def args_schema(self) -> type[BaseModel]:
        """Pydantic model for the tool's input arguments."""
        ...
        
    def execute(self, **kwargs) -> T:
        """Execute the tool with the given arguments."""
        ...


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    tool_name: str
    args: Dict[str, Any]


@dataclass
class ToolResult(Generic[T]):
    """Result of a tool execution."""
    success: bool
    result: Optional[T] = None
    error: Optional[str] = None


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def get_tools_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all tools for the LLM prompt."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args": tool.args_schema.model_json_schema() if hasattr(tool, 'args_schema') and hasattr(tool.args_schema, 'model_json_schema') else {}
            }
            for tool in self._tools.values()
        ]
