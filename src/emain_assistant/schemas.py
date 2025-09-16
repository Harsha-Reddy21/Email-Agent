
from pydantic import BaseModel, Field
from typing_extensions import Literal


class RouterSchema(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning behind the classification")
    classification: Literal["ignore","respond","notify"]= Field(
        description="The classification of the email: 'ignore' for irrelevant emails,"
        "'nofity' for important information that doesn't need a response,"
        "'respond' for emails that need a reply"
    )


