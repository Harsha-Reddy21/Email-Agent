from typing import List, Tuple, Any
import json 



def parse_email(email_input:dict)-> Tuple[str, str, str, str]:

    return (
        email_input.get("author",""),
        email_input.get("to",""),
        email_input.get("subject",""),
        email_input.get("email_thread","")
    )


def format_email_markdown(subject:str, author:str, to:str, email_thread:str)->str:

    return f"""
    **Subject** : {subject}
    **From** : {author}
    **To**: {to}
    
    {email_thread}

    ---

    """


def format_for_display(tool_call):
    display = ""
    if tool_call["name"] == "write_email":
        display += f"""# Email Draft

                **To**: {tool_call["args"].get("to")}
                **Subject**: {tool_call["args"].get("subject")}

                {tool_call["args"].get("content")}
                """
    elif tool_call["name"] == "schedule_meeting":
        display += f"""# Calendar Invite

                **Meeting**: {tool_call["args"].get("subject")}
                **Attendees**: {', '.join(tool_call["args"].get("attendees"))}
                **Duration**: {tool_call["args"].get("duration_minutes")} minutes
                **Day**: {tool_call["args"].get("preferred_day")}
                """
    elif tool_call["name"] == "Question":
        display += f"""# Question for User

                {tool_call["args"].get("content")}
                """
    else:
        display += f"""# Tool Call: {tool_call["name"]}

                Arguments:"""
        
        if isinstance(tool_call["args"], dict):
            display += f"\n{json.dumps(tool_call['args'], indent=2)}\n"
        else:
            display += f"\n{tool_call['args']}\n"
    return display