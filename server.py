import requests
from mcp.server.fastmcp import FastMCP
import logging
import json
import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("bookings")

backend_url = 'http://127.0.0.1:8000'
auth_token = 'supersecretdevtoken'

@mcp.tool()
def search_equipment(site_name: str) -> str:
    """
    Search for available equipment at a specific site.
    
    Args:
        site_name: The name of the site to search for equipment (e.g., "Basel pRED")
    
    Returns:
        A formatted list of available equipment with details
    """
    try:
        response = requests.get(
            f"{backend_url}/tools/by-site/bookable?site_name={site_name}",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=10
        )
        
        if response.status_code == 200:
            equipment_list = response.json()
            
            if not equipment_list:
                return f"No equipment found at site: {site_name}"
            
            # Format the response for better readability
            result = f"Found {len(equipment_list)} equipment items at {site_name}:\n\n"
            
            for idx, equipment in enumerate(equipment_list, 1):
                location = equipment.get('location', {})
                responsible = equipment.get('responsible_person', {})
                
                result += f"{idx}. {equipment.get('manufacturer', 'Unknown')} {equipment.get('equipment_model', 'Unknown Model')}\n"
                result += f"   Category: {equipment.get('category', 'Unknown')}\n"
                result += f"   Material #: {equipment.get('material_number', 'N/A')}\n"
                result += f"   Location: Room {location.get('room', 'N/A')}, Floor {location.get('floor', 'N/A')}, Building {location.get('building', 'N/A')}\n"
                result += f"   Contact: {responsible.get('first_name', 'Unknown')} {responsible.get('last_name', '')} ({responsible.get('email', 'N/A')})\n"
                result += f"   Check-in required: {'Yes' if equipment.get('requires_check_in') else 'No'}\n"
                result += f"   ID: {equipment.get('id', 'N/A')}\n\n"
            
            return result
            
        else:
            return f"Error searching equipment: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error searching equipment: {str(e)}"

@mcp.tool()
def book_equipment(
    equipment_ids: str,
    date: str,
    time_start: str,
    time_end: str,
    number_of_people: int = 1,
    reason: str = "Equipment usage",
    timezone: str = "Europe/Zurich"
) -> str:
    """
    Create a booking for equipment.
    
    Args:
        equipment_ids: Comma-separated list of equipment IDs to book (e.g., "45c5a1ee-2929-4b95-8bc9-d36b2b624a1c" or "id1,id2")
        date: Date of booking in YYYY-MM-DD format (e.g., "2025-07-07")
        time_start: Start time in HH:MM format (e.g., "10:30")
        time_end: End time in HH:MM format (e.g., "12:00")
        number_of_people: Number of people using the equipment (default: 1)
        reason: Reason for booking (default: "Equipment usage")
        timezone: Timezone for the booking (default: "Europe/Zurich")
    
    Returns:
        Confirmation message with booking details
    """
    try:
        # Clean up equipment_ids - remove brackets if present and handle quotes
        equipment_ids_cleaned = equipment_ids.strip()
        if equipment_ids_cleaned.startswith('[') and equipment_ids_cleaned.endswith(']'):
            equipment_ids_cleaned = equipment_ids_cleaned[1:-1]
        
        # Parse equipment IDs into array of strings
        tool_ids = [id.strip().strip('"').strip("'") for id in equipment_ids_cleaned.split(',') if id.strip()]
        
        # Format time with timezone suffix
        time_start_formatted = f"{time_start}:00.000Z"
        time_end_formatted = f"{time_end}:00.000Z"
        
        payload = {
            "tool_ids": tool_ids,  # This is already an array of strings
            "date": date,
            "time_start": time_start_formatted,
            "time_end": time_end_formatted,
            "timezone": timezone,
            "number_of_people": number_of_people,
            "reason": reason
        }
        
        logger.debug(f"Booking payload: {json.dumps(payload, indent=2)}")
                
        response = requests.post(
            f"{backend_url}/bookings",
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200 or response.status_code == 201:
            booking_data = response.json()
            
            # Format success response
            result = f"✅ Booking created successfully!\n\n"
            result += f"Booking Details:\n"
            result += f"- Equipment IDs: {', '.join(tool_ids)}\n"
            result += f"- Date: {date}\n"
            result += f"- Time: {time_start} - {time_end} ({timezone})\n"
            result += f"- Number of people: {number_of_people}\n"
            result += f"- Reason: {reason}\n"
            
            # Add booking ID if available in response
            if isinstance(booking_data, dict) and 'id' in booking_data:
                result += f"- Booking ID: {booking_data['id']}\n"
            
            return result
            
        else:
            error_msg = response.text
            try:
                error_data = response.json()
                if isinstance(error_data, dict) and 'detail' in error_data:
                    error_msg = error_data['detail']
            except:
                pass
                
            return f"❌ Error creating booking: {response.status_code} - {error_msg}"
            
    except Exception as e:
        return f"❌ Error creating booking: {str(e)}"


if __name__ == "__main__":
    # Initialize and run the server
    logger.info("Starting MCP server...")
    mcp.run(transport='stdio')