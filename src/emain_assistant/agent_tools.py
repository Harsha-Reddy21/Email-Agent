
from datetime import datetime
from langchain_core.tools import tool 
from pydantic import BaseModel




@tool
def write_email(to:str, subject:str, content:str)->str: 
    """Write and send an email"""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def check_calendar_availability(
    attendees:list[str],
    preferred_day: datetime,
    duration_minutes: int
) -> str:
    """Check calendar availability for meeting attendees."""
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return f" All attendess available on {date_str} between 9:00-17:00"


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int, 
    preferred_day: datetime, 
    start_time: int
):
    """Schedular a calendar meeting"""
    date_str= preferred_day.strftime("%A, %B %d, %Y")
    return f" Meeting '{subject}', scheduled on {date_str} at {start_time}"


@tool
class Done(BaseModel):
    """Mark that the email processing is complete"""
    done: bool=True 


TOOLS=[
    write_email,
    check_calendar_availability,
    schedule_meeting,
    Done
]