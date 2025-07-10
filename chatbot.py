from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
import json
from typing import List, Dict, Any
import nest_asyncio
import os

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
        self.default_auth_token = os.getenv('API_KEY', 'supersecretdevtoken')

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

    async def process_query(self, query: str, auth_token: str = None) -> str:
        """Process a query and return the response"""
        if not self._initialized:
            await self.initialize()
        
        # Store the auth_token for use in tool execution
        self.current_auth_token = auth_token or self.default_auth_token
        
        client = self.ollama
        
        # DON'T mention the auth_token in the system prompt to prevent hallucination
        system_prompt = """You are a helpful equipment booking assistant at Roche. 

You have access to the following tools:

1. search_equipment(site_name) - Search for available equipment at a specific site.
2. book_equipment(equipment_ids, date, time_start, time_end, number_of_people, reason, timezone) - Create a booking for equipment.

CRITICAL RULES FOR EQUIPMENT BOOKING:
- Use the EXACT equipment ID for Booking from the search results
- DO NOT hallucinate equipment ID for Booking

IMPORTANT RULES:
- Use only ONE tool call per response
- When you need to use a tool, respond ONLY with the JSON object, no additional text
- If you need to use multiple tools, do them in separate responses
- When using the book_equipment tool, ensure to pass the EXACT ID for Booking field from the search results

WORKFLOW:
1. For equipment booking requests: First search for equipment, then book using the EXACT ID for Booking
2. Always use the full UUID format for equipment IDs
3. Never truncate or modify equipment IDs

Tool call format:
{"tool_name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}

Examples:
- {"tool_name": "search_equipment", "arguments": {"site_name": "Basel pRED"}}
- {"tool_name": "book_equipment", "arguments": {"equipment_ids": "09ed436d-7c04-4c74-84f2-54b213cfb0fd", "date": "2025-07-18", "time_start": "14:30", "time_end": "17:00", "number_of_people": 3, "reason": "Calibration tests", "timezone": "Europe/Zurich"}}

Remember: Equipment IDs are always in UUID format (8-4-4-4-12 characters) and must be copied exactly!"""

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
            # Always inject the auth_token - this prevents LLM hallucination
            current_token = getattr(self, 'current_auth_token', self.default_auth_token)
            
            # Add auth_token to the arguments
            tool_args_with_auth = tool_args.copy()
            tool_args_with_auth['auth_token'] = current_token
            
            print(f"DEBUG: Calling tool '{tool_name}' with args: {tool_args_with_auth}")
            result = await self.session.call_tool(tool_name, tool_args_with_auth)
            return str(result.content[0].text) if result.content else "No result"
        except Exception as e:
            print(f"DEBUG: Tool execution failed: {e}")
            return f"Error executing tool: {e}"

    async def process_forecast(self, data: dict) -> dict:
        """Process forecast data and return analysis in the required format"""
        if not self._initialized:
            await self.initialize()

        client = self.ollama

        # Extract key metrics from the data
        manufacturer = data.get('manufacturer', 'Unknown')
        equipment_model = data.get('equipment_model', 'Unknown')
        team_name = data.get('team_name', 'Unknown')
        current_utilization = data.get('utilization_rate', 0) * 100  # Convert to percentage
        usage_per_day = data.get('usage_per_day', [])
        avg_booking_duration = data.get('average_booking_duration', 30.0)
        
        # Calculate current metrics
        total_hours = sum(day.get('hours', 0) for day in usage_per_day)
        total_bookings = len(usage_per_day)
        
        system_prompt = """You are a data analyst specializing in equipment utilization forecasting. Your job is to analyze equipment usage data and generate realistic weekly forecasts with insights.

    IMPORTANT: You must respond with ONLY a valid JSON object in the exact format requested. Do not include any explanatory text, markdown formatting, or additional content.

    The response must be a JSON object with two fields:
    {
    "forecast": [...], 
    "insights": "detailed analysis text"
    }

    Where:
    - forecast: A JSON array with exactly 6 objects, each representing a week of forecast data
    - insights: A string containing detailed analysis, trends, and recommendations

    Forecast array structure:
    [
    { "week": "Week 1", "utilization": 67, "hours": 145, "bookings": 12 },
    { "week": "Week 2", "utilization": 72, "hours": 158, "bookings": 14 },
    ...
    ]

    Insights should include:
    - Analysis of current usage patterns
    - Predicted trends and their reasoning
    - Potential risks or opportunities
    - Recommendations for optimization"""

        query = f"""Based on the following equipment data, generate a 6-week forecast with insights:

    Equipment: {manufacturer} {equipment_model}
    Team: {team_name}
    Current Utilization Rate: {current_utilization:.2f}%
    Historical Usage: {len(usage_per_day)} bookings, {total_hours} total hours
    Average Booking Duration: {avg_booking_duration} minutes

    Historical usage pattern:
    {json.dumps(usage_per_day, indent=2)}

    Generate a realistic 6-week forecast considering:
    1. Current usage trends
    2. Typical equipment utilization patterns
    3. Potential seasonal variations
    4. Team workflow patterns

    Response format: JSON object with forecast array and insights string, no additional text."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = client.chat(
                model='llama3.2',
                messages=messages
            )
            
            response_content = response['message']['content'].strip()
            print(f"Forecast response: {response_content}")
            
            # Try to parse and validate the JSON response
            try:
                # Remove any markdown formatting if present
                if response_content.startswith('```json'):
                    response_content = response_content[7:-3]
                elif response_content.startswith('```'):
                    response_content = response_content[3:-3]
                
                # Parse the JSON to validate it
                parsed_response = json.loads(response_content)
                
                # Validate the structure
                if isinstance(parsed_response, dict) and 'forecast' in parsed_response and 'insights' in parsed_response:
                    forecast = parsed_response['forecast']
                    insights = parsed_response['insights']
                    
                    # Validate forecast structure
                    if isinstance(forecast, list) and len(forecast) == 6:
                        for i, week_data in enumerate(forecast):
                            if not all(key in week_data for key in ['week', 'utilization', 'hours', 'bookings']):
                                raise ValueError(f"Missing required keys in week {i+1}")
                        
                        # Return both forecast and insights
                        return {
                            "forecast": json.dumps(forecast),
                            "insights": insights
                        }
                    else:
                        raise ValueError("Invalid forecast structure")
                else:
                    raise ValueError("Missing forecast or insights in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse LLM response as valid JSON: {e}")
                # Generate a fallback forecast and insights
                return self._generate_fallback_forecast_with_insights(current_utilization, total_hours, total_bookings)
                
        except Exception as e:
            error_msg = f"Error processing forecast: {e}"
            print(f"Exception: {error_msg}")
            # Return fallback forecast with insights
            return self._generate_fallback_forecast_with_insights(current_utilization, total_hours, total_bookings)

    def _generate_fallback_forecast_with_insights(self, current_utilization: float, total_hours: float, total_bookings: int) -> dict:
        """Generate a fallback forecast with insights if LLM fails"""
        import json
        import random
        
        # Base predictions on current usage with some variation
        base_utilization = max(10, min(90, current_utilization + random.randint(-5, 15)))
        base_hours = max(10, int(total_hours * 1.2 + random.randint(-10, 20)))
        base_bookings = max(5, int(total_bookings * 1.1 + random.randint(-2, 5)))
        
        forecast = []
        for i in range(1, 7):
            # Add some week-to-week variation
            utilization_variance = random.randint(-8, 12)
            hours_variance = random.randint(-15, 25)
            bookings_variance = random.randint(-3, 5)
            
            forecast.append({
                "week": f"Week {i}",
                "utilization": max(5, min(95, base_utilization + utilization_variance)),
                "hours": max(5, base_hours + hours_variance),
                "bookings": max(1, base_bookings + bookings_variance)
            })
        
        insights = f"""Based on historical usage data analysis:

    **Current Status:**
    - Current utilization rate: {current_utilization:.1f}%
    - Historical usage: {total_bookings} bookings totaling {total_hours} hours
    - Average booking pattern shows {'consistent' if total_bookings > 5 else 'sporadic'} usage

    **Forecast Trends:**
    - Projected utilization range: {base_utilization-8}% to {base_utilization+12}%
    - Expected weekly hours: {base_hours-15} to {base_hours+25} hours
    - Anticipated bookings: {base_bookings-3} to {base_bookings+5} per week

    **Recommendations:**
    - {'Consider increasing equipment availability during peak times' if base_utilization > 70 else 'Equipment appears underutilized - consider promotion or reallocation'}
    - Monitor usage patterns for optimization opportunities
    - Track actual vs predicted usage for model improvement

    *Note: This forecast was generated using fallback analysis due to system limitations.*"""
        
        return {
            "forecast": json.dumps(forecast),
            "insights": insights
        }