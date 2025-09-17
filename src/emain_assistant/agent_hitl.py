

from langchain.chat_models import init_chat_model
from typing import Literal
from langgraph.types import Command, interrupt
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, START, END
from schemas import RouterSchema,State, UserPrefernces, StateInput
from utils import parse_email, format_email_markdown, format_for_display
from prompts import DEFAULT_TRIAGE_INSTRUCTIONS, TRIAGE_SYSTEM_PROMPT, DEFAULT_BACKGROUND, TRIAGE_USER_PROMPT, AGENT_SYSTEM_PROMPT, DEFAULT_RESPONSE_PREFERENCES, DEFAULT_CAL_PREFERENCES,MEMORY_UPDATE_INSTRUCTIONS, MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT
from agent_tools import TOOLS
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
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
    print("******")
    print(f'state',state['messages'])
    print('------')

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
        print(f"Tool Args",tool_call['args'])

    return {'messages':result}



def update_memory(store, namespace, messages):

    user_preferences=store.get(namespace, "user_preferences")
    current_profile=user_preferences.value if user_preferences else "No existing preferences"

    llm_memory=init_chat_model("openai:gpt-4.1",temperature=0.0).with_structured_output(UserPrefernces)
    result=llm_memory.invoke(
        [
            {"role":"system","content":MEMORY_UPDATE_INSTRUCTIONS.format(current_profile=current_profile,namespace=namespace)},
        ] + messages
    )

    print("to_update_final_user_preferences:",result)

    store.put(namespace,'user_preferences:',result.user_preferences)






def triage_interrupt_handler(state:State, store:BaseStore) -> Command[Literal['response_agent','__end__']]:

    author, to, subject, email_thread=parse_email(state['email_input'])

    email_markdown=format_email_markdown(subject,author, to, email_thread)

    messages=[{
        "role":"user",
        "content":f"Email to notify user about: {email_markdown}"
    }]


    request={
        "action_request":{
            "action":f"Email Assistant: {state['classification_decision']}",
            "args":{}
        },
        "config":{
            "allow_ignore":True,
            "allow_respond":True,
            "allow_edit":False,
            "allow_accept":False,
        },
        "description":email_markdown
    }


    response=interrupt([request])[0]

    if response['type']=='response':
        user_input=response['args']
        messages.append({
            "role":"user",
            "content":f"User wants to reply to the email. Use this feedback to respond: {user_input}"
        })


        update_memory(store,("email_assistant","triage_preferences"),[{
            "role":"user",
            "content":f"The user decided to respond to the email, so update the triage preferences to capture this."
            }] + messages)
        goto='response_agent'
    
    elif response['type']=='ignore':
        messages.append({
            "role":"user",
            "content":f"The user decided to ignore the email even though it was classified as notify. Update triage prefernces to capture this."
        })

        update_memory(store, ("email_assistant","triage_prefernces"),messages)
        goto= END 
    else:
        raise ValueError(f"Invalid response type: {response}")
    
    update={
        "messages":messages,
        "classification_decision":state['classification_decision']
    }

    return Command(goto=goto, update=update)



def interrupt_handler(state: State, store: BaseStore)-> Command[Literal["llm_call","__end__"]]:

    result=[]
    goto='llm_call'

    for tool_call in state['messages'][-1].tool_calls:
        
        hitl_tools=['write_email','schedule_meeing','Question']

        if tool_call['name'] not in hitl_tools:
            tool=tools_by_name[tool_call['name']]
            observation=tool.invoke(tool_call['args'])
            result.append({
                "role":"tool",
                "content":str(observation),
                "tool_call_id":tool_call['id']
            })
            continue

            
        email_input=state['email_input']
        author, to, subject, email_thread=parse_email(email_input)
        original_email_markdown=format_email_markdown(subject,author, to, email_thread)

        tool_display =format_for_display(tool_call)
        description= original_email_markdown + tool_display

        if tool_call["name"] == "write_email":
            config = {
                "allow_ignore": True,    
                "allow_respond": True,   
                "allow_edit": True,    
                "allow_accept": True,    
            }
        elif tool_call["name"] == "schedule_meeting":
            config = {
                "allow_ignore": True,   
                "allow_respond": True,  
                "allow_edit": True,      
                "allow_accept": True,  
            }
        elif tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,   
                "allow_respond": True,  
                "allow_edit": False,     
                "allow_accept": False,  
            }
        else:
            raise ValueError(f"Unexpected HITL tool: {tool_call['name']}")


        request = {
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"]
            },
            "config": config,
            "description": description,
        }

        response = interrupt([request])[0]

        if response["type"] == "accept":
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({
                "role": "tool", 
                "content": str(observation), 
                "tool_call_id": tool_call["id"]
            })
        elif response["type"] == "edit":
            tool = tools_by_name[tool_call["name"]]
            edited_args = response["args"]["args"]

            ai_message = state["messages"][-1]  
            current_id = tool_call["id"]  

            updated_tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
                {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
            ]
            
            result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))

            if tool_call["name"] == "write_email":
                initial_tool_call = tool_call["args"]
                observation = tool.invoke(edited_args)
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})
                update_memory(store, ("email_assistant", "response_preferences"), [{
                    "role": "user",
                    "content": f"User edited the email response. Here is the initial email generated by the assistant: {initial_tool_call}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            elif tool_call["name"] == "schedule_meeting":
                initial_tool_call = tool_call["args"]
                observation = tool.invoke(edited_args)
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})
                update_memory(store, ("email_assistant", "cal_preferences"), [{
                    "role": "user",
                    "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {initial_tool_call}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            if tool_call["name"] == "write_email":
                result.append({"role": "tool", "content": "User ignored this email draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                goto = END
                update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            elif tool_call["name"] == "schedule_meeting":
                result.append({"role": "tool", "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                goto = END
                update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            elif tool_call["name"] == "Question":
                result.append({"role": "tool", "content": "User ignored this question. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                goto = END
                update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")
            
        elif response["type"] == "response":
            user_feedback = response["args"]
            if tool_call["name"] == "write_email":
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
                update_memory(store, ("email_assistant", "response_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"User gave feedback, which we can use to update the response preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            elif tool_call["name"] == "schedule_meeting":
               
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})

                update_memory(store, ("email_assistant", "cal_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            elif tool_call["name"] == "Question": 

                result.append({"role": "tool", "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        else:
            raise ValueError(f"Invalid response type: {response}")
            
    update = {"messages": result}
    return Command(goto=goto, update=update)











def should_continue(state: State, store: BaseStore) -> Literal["interrupt_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "interrupt_handler"
    return END


response_agent = StateGraph(State)
response_agent.add_node("llm_call", llm_call) 
response_agent.add_node("interrupt_handler", interrupt_handler)

response_agent.add_edge(START, "llm_call")
response_agent.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)
response_agent.add_edge("interrupt_handler", "llm_call")

compiled_response_agent = response_agent.compile()

email_assistant_hitl = StateGraph(State, input=StateInput)
email_assistant_hitl.add_node("triage_router", triage_router)
email_assistant_hitl.add_node("triage_interrupt_handler", triage_interrupt_handler)
email_assistant_hitl.add_node("response_agent", compiled_response_agent)

email_assistant_hitl.add_edge(START, "triage_router")

checkpointer = InMemorySaver()
store = InMemoryStore()
compiled_email_assistant_hitl = email_assistant_hitl.compile(checkpointer=checkpointer, store=store)


