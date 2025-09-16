

from langchain.chat_models import init_chat_model
from schemas import RouterSchema
from agent_tools import TOOLS
from dotenv import load_dotenv
load_dotenv()
import os 

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

llm=init_chat_model("openai:gpt-4.1",temperature=0.0)
llm_router=llm.with_structured_output(RouterSchema)


tools_by_name={tool.name:tool for tool in TOOLS}
llm_with_tools=llm.bind_tools(TOOLS, tool_choice='any')


if __name__=='__main__':
    response=llm_with_tools.invoke("what is bhagavadgita")
    print(response)