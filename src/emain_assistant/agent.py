

from langchain.chat_models import init_chat_model
from typing import Literal
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from schemas import RouterSchema,State
from utils import parse_email, format_email_markdown
from prompts import DEFAULT_TRIAGE_INSTRUCTIONS, TRIAGE_SYSTEM_PROMPT, DEFAULT_BACKGROUND, TRIAGE_USER_PROMPT, AGENT_SYSTEM_PROMPT, DEFAULT_RESPONSE_PREFERENCES, DEFAULT_CAL_PREFERENCES
from agent_tools import TOOLS
from dotenv import load_dotenv
load_dotenv()
import os 

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

llm=init_chat_model("openai:gpt-4.1",temperature=0.0)
llm_router=llm.with_structured_output(RouterSchema)


tools_by_name={tool.name:tool for tool in TOOLS}
llm_with_tools=llm.bind_tools(TOOLS, tool_choice='any')

def triage_router(state:State):

    author,to, subject, email_thread=parse_email(state["email_input"])

    system_prompt=TRIAGE_SYSTEM_PROMPT.format(background=DEFAULT_BACKGROUND,triage_instructions=DEFAULT_TRIAGE_INSTRUCTIONS)

    user_prompt= TRIAGE_USER_PROMPT.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    result=llm_router.invoke([
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ])

    print(f"Email Triage: {result.classification.upper()}")
    print(f"Result",result)

    if result.classification=='respond':
        print("Routing to response agent")
        goto="response_agent"
        update={
            "classification_decision":result.classification,
            "messages":[{
                "role":"user",
                "content":f"Respond to the email: \n\n {format_email_markdown(subject,author, to, email_thread)}"
            }]
        }
    elif result.classification=='ignore':
        print("Email will be ignored")
        goto=END 
        update={"classification_decision":result.classification}
    elif result.classification=='notify':
        print("Email marked for notification only")
        goto=END 
        update={"classification_decision":{result.classification}}
    else:
        raise ValueError(f"Invalid classificaton: {result.classification}")
    
    return Command(goto=goto,update=update)

def llm_call(state:State):
    system_prompt=AGENT_SYSTEM_PROMPT.format(
        background=DEFAULT_BACKGROUND,
        responses_preferences=DEFAULT_RESPONSE_PREFERENCES,
        cal_preferences=DEFAULT_CAL_PREFERENCES
    )

    return {
        "messages":[
            llm_with_tools.invoke([
                {"role":"system","content":system_prompt}
            ]+ state["messages"]) 
        ]
    }


def tool_handler(state:State):
    last_message=state['messages'][-1]
    result=[]

    for tool_call in last_message.tool_calls:
        tool=tools_by_name[tool_call['name']]
        observation=tool.invoke(tool_call['args'])
        result.append({
            "role":"tool",
            "content":str(observation),
            "tool_call_id":tool_call['id']
        })
        print(f"Tool executed: {tool_call['name']}")

    return {'messages':result}


def should_continue(state: State)-> Literal["tool_handler","__end__"]:
    
    last_message=state['messages'][-1]

    if last_message.tool_calls:

        for tool_call in last_message.tool_calls:
            if tool_call['name']=='Done':
                print("Processing Completed")
                return END 
        return "tool_handler"
    
    return  END





response_agent=StateGraph(State)
response_agent.add_node("llm_call",llm_call)
response_agent.add_node("tool_handler",tool_handler)

response_agent.add_edge(START, "llm_call")
response_agent.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_handler":"tool_handler",
        END: END,
        },


)

response_agent.add_edge("tool_handler","llm_call")
compiled_response_agent=response_agent.compile()



email_assistant=StateGraph(State)
email_assistant.add_node("triage_router",triage_router)
email_assistant.add_node("response_agent",compiled_response_agent)

email_assistant.add_edge(START,"triage_router")

compiled_email_assistant=email_assistant.compile()


def process_email(email_input:dict)->dict:
    result=compiled_email_assistant.invoke({'email_input':email_input})
    response_text="No response generated"




    if result.get("messages"):

        for message in result['messages']:

            if (hasattr(message,'tool_calls') and message.tool_calls):
                for tool_call in message.tool_calls:
                    if tool_call.get('name')=='write_email':
                        
                        args=tool_call.get('args',{})
                        email_content=args.get('content',"")
                        if email_content:
                            response_text=email_content
                            break
                
                if response_text!="No response generated":
                    break 
            
            elif (hasattr(message,'role') and message.role=='assistant' and hasattr(message,'content') and message.content.strip()):
                response_text=message.content
                break 

    return {
        "classification":result.get("classification_decision","unknown"),
        "response": response_text, 
        "reasoning": f"Email classified as : {result.get('classification_decision','unknown')}"
    }







if __name__=='__main__':
    process_email({
    "author": "Alice <alice@company.com>",
    "to": "John <john@company.com>", 
    "subject": "Question about API",
    "email_thread": "Hi! I have a question about the API documentation. Could we schedule a quick call this week?"
  })