import os
import json
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

import wikidata 

TEMPERATURE = 0.0

SEED = 42
#TODO: add more LLM API's
#TODO: Load prompts and model from a YAML to streamline validation

# Load env variables
load_dotenv()


#Structured Output Schemas for the LLM
class ExtractionResult(BaseModel):
    entities: List[str] = Field(description="Search terms for entities, concepts, or locations eg. ['Feynman', 'Greece']")
    properties: List[str] = Field(description="Search terms for properties eg. ['has child', 'is governor']")

class SparqlGenerationResult(BaseModel):
    sparql: str = Field(description="The SPARQL query string for Wikidata")
    explanation: str = Field(description="An explanation of how the SPARQL query was constructed.")

class SparqlCorrectionResult(BaseModel):
    sparql: str = Field(description="The corrected SPARQL query string.")
    explanation: str = Field(description="What was fixed to correct the SPARQL syntax or logical error.")


class TextToSparqlPipeline:
    
    def __init__(self, model_name:str = "gemini-3.1-flash-lite"):
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client()
        self.model_name = model_name

    def find_entities_and_properties(self, question:str) -> Tuple[List[str], List[str]]:
        """Extract entities and properties from a natural language query using LLM prompting"""
        prompt = (
            "Analyze the following natural language query and extract search terms for the wikidata entities"
            "(nouns, names, topics) and properties (attributes, relations, verbs connecting them). \n\n"
            f"Query: {question}"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=TEMPERATURE,
                seed=SEED
            )
        )
        result = json.loads(response.text)
        #print(result.get("entities", []), result.get("properties", []))
        return result.get("entities", []), result.get("properties", [])

    def get_wikidata_ids(self, entities: List[str], properties: List[str])-> Dict[str, Any]:
        """
        Use Wikidata Search API to find possible Q and P ids
        """
        candidates = {
            "entities": {},
            "properties": {}
        }

        for entity in entities:
            search_results = wikidata.search_wikidata(search_term=entity, search_type="item", limit=4)
            candidates["entities"][entity] = search_results
        for prop in properties:
            search_results = wikidata.search_wikidata(search_term=prop, search_type="property", limit=4)
            candidates["properties"][prop] = search_results

        return candidates
    
    def generate_sparql_query(self, question: str, candidates: Dict[str, Any]) -> Tuple[str, str]:
        """
        Use the LLM to convert the natural language question to SPARQL query
        """
        candidates_str = json.dumps(candidates, indent=2)
        #print(candidates_str)
        
        prompt = (
            "You are an expert SPARQL query generator for Wikidata.\n"
            "Generate a SPARQL query that answers the User Question based on the provided Candidate Items and Properties.\n\n"
            "Wikidata Rules:\n"
            "1. Use 'wd:Qxxxx' for items and 'wdt:Pxxxx' for properties.\n"
            "2. Use '?item wdt:P31 wd:Q5' for instance of human.\n"
            "3. Use the Wikidata label service for human-readable labels:\n"
            "   SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }\n"
            "   This service automatically generates a '?variableLabel' variable for any '?variable' (eg. if you select ?physicist, also select ?physicistLabel and add it to the SELECT list).\n"
            "4. Keep queries clean, concise, and executable.\n\n"
            f"Candidate Wikidata IDs:\n{candidates_str}\n\n"
            f"User Question: {question}\n"
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SparqlGenerationResult,
                temperature=TEMPERATURE
            )
        )
        result = json.loads(response.text)
        #print(result.get("sparql"), result.get("explanation"))
        return result.get("sparql"), result.get("explanation")
    
    def correct_sparql(self, question: str, failed_query: str, error_msg: str, candidates: Dict[str, Any]) -> Tuple[str, str]:
        """
        Use the LLM to parse the error from a failed query and suggest corrections
        """
        candidates_str = json.dumps(candidates, indent=2)
        
        prompt = (
            "You generated a SPARQL query that failed execution. Fix the error and return a valid query.\n\n"
            f"User Question: {question}\n"
            f"Failed SPARQL Query:\n```sparql\n{failed_query}\n```\n"
            f"Execution Error: {error_msg}\n\n"
            f"Candidate Wikidata IDs for context:\n{candidates_str}\n"
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SparqlCorrectionResult,
                temperature=TEMPERATURE
            )
        )
        
        result = json.loads(response.text)
        return result.get("sparql"), result.get("explanation")