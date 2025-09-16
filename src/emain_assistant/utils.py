from typing import List, Tuple, Any

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