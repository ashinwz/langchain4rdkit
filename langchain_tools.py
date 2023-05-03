from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain import OpenAI, SerpAPIWrapper
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import TextRequestsWrapper
import requests
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

import os, json

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs


os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
python = PythonREPL()


def calculate_properties(smiles):
    try:
        smiles = smiles.split(".")[0]
        mol = Chem.MolFromSmiles(str(smiles))
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "AcceptorH": Descriptors.NumHAcceptors(mol),
            "DonorsH": Descriptors.NumHDonors(mol)
        }
    except:
        return "Wrong SMILES format, calculate_properties is not valid for the wrong SMILES, try search the SMILES first"

def calculate_similarity(smiles_str):
    try:
        smiles_str = smiles_str.replace(".", "")
        smiles_list = smiles_str.split(",")
        mol1 = Chem.MolFromSmiles(smiles_list[0])
        mol2 = Chem.MolFromSmiles(smiles_list[1])

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        # Calculate Tanimoto similarity between the two fingerprints
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

        return similarity

    except:
        return "Wrong SMILES format, calculate_properties is not valid for the wrong SMILES, try search the SMILES first"

def search_SMILES(drugname):
    # pubchem API to get SMILES 
    requests = TextRequestsWrapper()
    response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drugname}/property/CanonicalSMILES/json")

    return response

def search_drug_image(drugname):
    import base64

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drugname}/PNG"

    return {"img": url}


tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current information , current events or the current state of the world"
    ),
    Tool(
        name = "Search drug",
        func=lambda s: search_SMILES(s),
        description="useful for when you need to search the SMILES by drugname.",
        # return_direct=True
    ),
    Tool(
        name = "Search drug image",
        func=lambda s: search_drug_image(s),
        description="useful for when you need to search the structure images by drugname.",
        # return_direct=True
    ),
    Tool(
        name="Calculate properties",
        func=lambda s: calculate_properties(s),
        description="useful when you want to calculate the molecular properties by give a molecule SMILES/CanonicalSMILES format. \
        If you receive two and more compounds with `and` words, first you should split it \
        For example the `O=C(Oc1ccccc1C(=O)O)C` is the input and if you want to calcuate the properties "
    ),
    Tool(
        name="Calculate similarity",
        func=lambda s: calculate_similarity(s),
        description="useful when you want to calculate the two compound similarity score. Input should be a list. Firstly split the two compounds into the list like 'CC1,CCC' "
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

def ag_plugin(questions):
    with get_openai_callback() as cb:

        response = agent.run(questions)

        response_msg = {
            'usage': {'prompt_tokens': cb.prompt_tokens, 'completion_tokens': cb.completion_tokens, 'total_tokens': cb.total_tokens},
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': response 
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ]
        }

        print(response_msg)
    return json.dumps(response_msg)

if __name__ == "__main__":
    #ag_plugin("Calculate the molecular properties of the Abemaciclib")
    #ag_plugin("Get the SMILES of Abemaciclib")
    #Get the structure image of Abemaciclib
    #ag_plugin("Get the SMILES of Abrocitinib")
    #ag_plugin("Calculate the similarity between Abemaciclib and Abrocitinib")
    ag_plugin("今天是星期几？")