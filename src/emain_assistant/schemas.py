
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
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