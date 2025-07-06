from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
import json
from typing import List, Dict, Any
import nest_asyncio

nest_asyncio.apply()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.ollama = ollama.Client()
        self.available_tools: List[dict] = []
        self._initialized = False
        self._stdio_client = None
        self._client_session = None
        # Add conversation memory
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10  # Keep last 10 exchanges

    async def initialize(self):
        """Initialize the MCP connection"""
        if self._initialized:
            return
            
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "server.py"],
            env=None,
        )
        
        # Store the connection for proper cleanup
        self._stdio_client = stdio_client(server_params)
        self.read, self.write = await self._stdio_client.__aenter__()
        
        self._client_session = ClientSession(self.read, self.write)
        self.session = await self._client_session.__aenter__()
        
        await self.session.initialize()
        response = await self.session.list_tools()
        
        self.available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        print(f"Available tools: {self.available_tools}")
        self._initialized = True

    async def cleanup(self):
        """Cleanup connections"""
        try:
            if self._client_session:
                await self._client_session.__aexit__(None, None, None)
            if self._stdio_client:
                await self._stdio_client.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self._initialized = False

    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep only the last max_history_length exchanges
        if len(self.conversation_history) > self.max_history_length * 2:  # *2 because each exchange has user + assistant
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]

    def get_conversation_messages(self, system_prompt: str, current_query: str) -> List[Dict[str, str]]:
        """Build the full conversation context for the LLM"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current query
        messages.append({"role": "user", "content": current_query})
        
        return messages

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    async def process_query(self, query: str) -> str:
        """Process a query and return the response"""
        if not self._initialized:
            await self.initialize()
            
        client = self.ollama
        
        system_prompt = """You are a helpful equipment booking assistant at Roche. 
    You have access to the following tools:

    1. search_equipment(site_name) - Search for available equipment at a specific site.
    2. book_equipment(equipment_ids, date, time_start, time_end, number_of_people, reason, timezone) - Create a booking for equipment.

    IMPORTANT RULES:
    - Use only ONE tool call per response
    - When you need to use a tool, respond ONLY with the JSON object, no additional text
    - If you need to use multiple tools, do them in separate responses

    Tool call format:
    {"tool_name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}

    Examples:
    - {"tool_name": "search_equipment", "arguments": {"site_name": "Basel pRED"}}
    - {"tool_name": "book_equipment", "arguments": {"equipment_ids": "45c5a1ee-2929-4b95-8bc9-d36b2b624a1c", "date": "2025-07-07", "time_start": "10:30", "time_end": "12:00", "number_of_people": 1, "reason": "Lab tests"}}

    For booking, use equipment_ids as a simple string (not in brackets): "id1" or "id1,id2" for multiple."""

        # Different system prompt for follow-up responses
        follow_up_system_prompt = """You are a helpful equipment booking assistant at Roche. 
    You have just executed a tool and received the results. Your job is to provide a clear, human-readable summary of the results to the user.

    IMPORTANT: Do NOT return JSON. Provide a natural language response that summarizes the tool results in a helpful way."""

        try:
            # Build conversation with history
            messages = self.get_conversation_messages(system_prompt, query)
            
            response = client.chat(
                model='llama3.2',
                messages=messages
            )
            
            response_content = response['message']['content']
            print(f"Model response: {response_content}")
            
            # Check if the response contains a tool call
            if self.is_tool_call(response_content):
                tool_call = self.parse_tool_call(response_content)
                if tool_call:
                    tool_name = tool_call['tool_name']
                    tool_args = tool_call['arguments']
                    
                    print(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    # Execute the tool
                    result = await self.execute_tool(tool_name, tool_args)
                    print(f"Tool result: {result}")
                    
                    # Build follow-up messages with different system prompt
                    follow_up_messages = [
                        {"role": "system", "content": follow_up_system_prompt},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": response_content},
                        {"role": "user", "content": f"Tool execution completed. Here are the results:\n\n{result}\n\nPlease provide a helpful summary of these results for the user."}
                    ]
                    
                    print("Sending follow-up request to LLM...")
                    final_response = client.chat(
                        model='llama3.2',
                        messages=follow_up_messages
                    )
                    
                    final_content = final_response['message']['content']
                    print(f"Final response: {final_content}")
                    
                    # Add to conversation history
                    self.add_to_history("user", query)
                    self.add_to_history("assistant", final_content)
                    
                    return final_content
                else:
                    error_msg = "Failed to parse tool call"
                    print(f"Error: {error_msg}")
                    self.add_to_history("user", query)
                    self.add_to_history("assistant", error_msg)
                    return error_msg
            else:
                print("No tool call detected, returning direct response")
                # Add to conversation history
                self.add_to_history("user", query)
                self.add_to_history("assistant", response_content)
                return response_content
                
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            print(f"Exception: {error_msg}")
            self.add_to_history("user", query)
            self.add_to_history("assistant", error_msg)
            return error_msg

    def is_tool_call(self, content: str) -> bool:
        """Check if the response contains a tool call"""
        try:
            # First try to parse the entire content as JSON
            parsed = json.loads(content.strip())
            if 'tool_name' in parsed and 'arguments' in parsed:
                tool_names = [tool['name'] for tool in self.available_tools]
                if parsed['tool_name'] in tool_names:
                    print("Tool call detected - full JSON")
                    return True
        except:
            # If that fails, look for JSON within the text
            import re
            # More flexible pattern that handles nested objects better
            json_pattern = r'\{[^{}]*"tool_name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}'
            matches = re.findall(json_pattern, content)
            if matches:
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        if 'tool_name' in parsed and 'arguments' in parsed:
                            tool_names = [tool['name'] for tool in self.available_tools]
                            if parsed['tool_name'] in tool_names:
                                print("Tool call detected - regex match")
                                return True
                    except:
                        continue
            
            # Try even simpler approach - look for the basic structure
            if '"tool_name"' in content and '"arguments"' in content:
                print("Tool call structure detected")
                return True
        
        print("No tool call detected")
        return False

    def parse_tool_call(self, content: str) -> Dict[str, Any]:
        """Parse tool call from model response"""
        try:
            # First try to parse the entire content as JSON
            content_stripped = content.strip()
            if content_stripped.startswith('```json'):
                content_stripped = content_stripped[7:-3]
            elif content_stripped.startswith('```'):
                content_stripped = content_stripped[3:-3]
            
            try:
                parsed = json.loads(content_stripped)
                if 'tool_name' in parsed and 'arguments' in parsed:
                    print(f"Parsed tool call: {parsed}")
                    return parsed
            except Exception as e:
                print(f"Failed to parse as full JSON: {e}")
            
            # If that fails, try to extract JSON using a more robust approach
            import re
            
            # Look for JSON objects that contain tool_name and arguments
            json_pattern = r'\{[^{}]*"tool_name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}'
            matches = re.findall(json_pattern, content)
            
            if matches:
                # Try to parse each match
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        if 'tool_name' in parsed and 'arguments' in parsed:
                            tool_names = [tool['name'] for tool in self.available_tools]
                            if parsed['tool_name'] in tool_names:
                                print(f"Parsed tool call from regex: {parsed}")
                                return parsed
                    except Exception as e:
                        print(f"Failed to parse regex match: {e}")
                        continue
            
            # Last resort - try to find and parse any JSON-like structure
            json_start = content.find('{"tool_name"')
            if json_start != -1:
                # Find the matching closing brace
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(content[json_start:], json_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = content[json_start:json_end]
                    try:
                        parsed = json.loads(json_str)
                        if 'tool_name' in parsed and 'arguments' in parsed:
                            print(f"Parsed tool call with brace matching: {parsed}")
                            return parsed
                    except Exception as e:
                        print(f"Failed to parse with brace matching: {e}")
            
            print("Failed to parse any tool call")
            return None
            
        except Exception as e:
            print(f"Error parsing tool call: {e}")
            return None
            
    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute MCP tool"""
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            return str(result.content[0].text) if result.content else "No result"
        except Exception as e:
            return f"Error executing tool: {e}"