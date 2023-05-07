from dotenv import load_dotenv
load_dotenv()

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL

from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
python = PythonREPL()