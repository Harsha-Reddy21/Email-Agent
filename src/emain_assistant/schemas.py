
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict, Optional, Any, Dict, List
from langgraph.graph import MessagesState

class RouterSchema(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning behind the classification")
    classification: Literal["ignore","respond","notify"]= Field(
        description="The classification of the email: 'ignore' for irrelevant emails,"
        "'nofity' for important information that doesn't need a response,"
        "'respond' for emails that need a reply"
    )




class State(MessagesState):
    email_input:dict 
    classification_decision: Literal["ignore","responsd","notify"]



class UserPrefernces(BaseModel):
    chain_of_thought: str= Field(description="Reasoning about which user preferences need to add/update if required")
    user_preferences: str = Field(description='Updated user preferences')


class StateInput(TypedDict):

    email_input: dict


class EmailInput(BaseModel):
    author: str = Field(description="Email sender")
    to: str = Field(description="Email recipient")
    subject: str = Field(description="Email subject line")
    email_thread: str = Field(description="Email content/body")

class ProcessEmailRequest(BaseModel):
    email: EmailInput
    

class ProcessEmailResponse(BaseModel):
    classification: Literal["ignore", "respond", "notify"]
    response: str
    reasoning: str


class HumanResponse(BaseModel):
    
    type: Literal["accept", "edit", "ignore", "response"] = Field(
        description="Type of human response"
    )
    args: Optional[Any] = Field(
        default=None,
        description="Arguments for edit/response actions"
    )



class ProcessEmailHITLRequest(BaseModel):
    
    email: Optional[EmailInput] = Field(
        default=None,
        description="Email data (required for new workflows)"
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for resuming existing workflows"
    )
    human_response: Optional[HumanResponse] = Field(
        default=None,
        description="Human response to resume from interrupt"
    )


class InterruptInfo(BaseModel):
    
    action: str = Field(description="The tool/action that triggered the interrupt")
    args: Dict[str, Any] = Field(description="Original arguments for the action")
    description: str = Field(description="Human-readable description of the action")
    allowed_actions: List[str] = Field(description="List of allowed human response types")



class ProcessEmailHITLResponse(BaseModel):    
    status: Literal["interrupted", "completed", "error"] = Field(
        description="Status of the workflow"
    )
    thread_id: str = Field(description="Thread ID for this workflow")
    interrupt: Optional[InterruptInfo] = Field(
        default=None,
        description="Interrupt details when status=interrupted"
    )
    result: Optional[ProcessEmailResponse] = Field(
        default=None,
        description="Final result when status=completed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message when status=error"
    )