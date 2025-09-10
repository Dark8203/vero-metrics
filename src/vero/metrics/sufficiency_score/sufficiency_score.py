from vero.metrics import MetricBase
import json
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from .prompt import prompt_sufficiency
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
OPENAI_API = os.getenv('OPENAI_API_KEY')
client = OpenAI()

custom_rag_prompt = PromptTemplate.from_template(prompt_sufficiency)

#TODO: still need to refactor for api and model and change import for prompt
class SufficiencyScore(MetricBase):
    name:str = 'sufficiency_score'

    def __init__(self, context:list|str, question:list|str):
        self.context = context
        self.question = question

    def evaluate(self):
        llm = ChatOpenAI(
            model='gpt-4o-mini',
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=OPENAI_API
        )
        messages = custom_rag_prompt.invoke({"question": self.question, "context": self.context})
        response = llm.invoke(messages)
        response_json = json.loads(response.content)
        return response_json['score']