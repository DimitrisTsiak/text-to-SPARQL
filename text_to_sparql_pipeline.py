import os
import json
import yaml
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

import wikidata 

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
    
    def __init__(self, config_path: str = "config.yaml", model_name: str = None):
        self.client = genai.Client()
        self.config_path = config_path
        self.model_name = "gemini-3.1-flash-lite"
        self.seed = 42
        self.temperature = 0.0
        self.sampling_parameters = {}
        self.prompts = {}
        self.trajectory_logging = {"enabled": False, "log_path": "trajectories.jsonl"}
        
        self.load_config(config_path)
        if model_name:
            self.model_name = model_name

    def load_config(self, config_path: str):
        """Loads configuration from a YAML file."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = os.path.join(base_dir, config_path)
        if not os.path.exists(resolved_path):
            resolved_path = config_path
            
        if os.path.exists(resolved_path):
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                if config:
                    self.model_name = config.get("model_name", self.model_name)
                    self.seed = config.get("seed", self.seed)
                    self.temperature = config.get("temperature", self.temperature)
                    self.sampling_parameters = config.get("sampling_parameters", {})
                    self.prompts = config.get("prompts", {})
                    self.trajectory_logging = config.get("trajectory_logging", self.trajectory_logging)
            except Exception as e:
                print(f"[Warning] Failed to load config file: {e}. Using default parameters.")
        else:
            print(f"[Warning] Config file not found at {resolved_path}. Using default parameters.")

    def _get_generate_config(self, response_mime_type: str = None, response_schema: Any = None) -> types.GenerateContentConfig:
        """Helper to construct GenerateContentConfig using YAML sampling parameters."""
        config_kwargs = {}
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if self.seed is not None:
            config_kwargs["seed"] = self.seed
            
        # Merge other sampling parameters from YAML
        for k, v in self.sampling_parameters.items():
            config_kwargs[k] = v
            
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if response_schema:
            config_kwargs["response_schema"] = response_schema
            
        return types.GenerateContentConfig(**config_kwargs)


    def find_entities_and_properties(self, question:str) -> Tuple[List[str], List[str]]:
        """Extract entities and properties from a natural language query using LLM prompting"""
        prompt_tmpl = self.prompts.get(
            "extraction_prompt",
            "Analyze the following natural language query and extract search terms for the wikidata entities "
            "(nouns, names, topics) and properties (attributes, relations, verbs connecting them). \n\n"
            "Query: {question}"
        )
        prompt = prompt_tmpl.replace("{question}", question)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._get_generate_config(
                response_mime_type="application/json",
                response_schema=ExtractionResult
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
        
        prompt_tmpl = self.prompts.get(
            "generation_prompt",
            "You are an expert SPARQL query generator for Wikidata.\n"
            "Generate a SPARQL query that answers the User Question based on the provided Candidate Items and Properties.\n\n"
            "Wikidata Rules:\n"
            "1. Use 'wd:Qxxxx' for items and 'wdt:Pxxxx' for properties.\n"
            "2. Use '?item wdt:P31 wd:Q5' for instance of human.\n"
            "3. Use the Wikidata label service for human-readable labels:\n"
            "   SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }\n"
            "   This service automatically generates a '?variableLabel' variable for any '?variable' (eg. if you select ?physicist, also select ?physicistLabel and add it to the SELECT list).\n"
            "4. Keep queries clean, concise, and executable.\n"
            "5. Pay close attention to the datatype of candidate properties (e.g., 'wikibase-item' links to items, 'time' is a date, 'quantity' is numeric). For 'time', compare using xsd:dateTime (e.g., ?date >= \"1900-01-01\"^^xsd:dateTime) or bind year (e.g., BIND(year(?date) AS ?year)). For 'quantity', compare or filter numerically (e.g., ?population > 100000).\n\n"
            "Candidate Wikidata IDs:\n{candidates_str}\n\n"
            "User Question: {question}\n"
        )
        prompt = prompt_tmpl.replace("{candidates_str}", candidates_str).replace("{question}", question)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._get_generate_config(
                response_mime_type="application/json",
                response_schema=SparqlGenerationResult
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
        
        prompt_tmpl = self.prompts.get(
            "correction_prompt",
            "You generated a SPARQL query that failed execution. Fix the error and return a valid query.\n\n"
            "User Question: {question}\n"
            "Failed SPARQL Query:\n```sparql\n{failed_query}\n```\n"
            "Execution Error: {error_msg}\n\n"
            "Candidate Wikidata IDs for context:\n{candidates_str}\n"
        )
        prompt = prompt_tmpl.replace("{question}", question).replace("{failed_query}", failed_query).replace("{error_msg}", error_msg).replace("{candidates_str}", candidates_str)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._get_generate_config(
                response_mime_type="application/json",
                response_schema=SparqlCorrectionResult
            )
        )
        
        result = json.loads(response.text)
        return result.get("sparql"), result.get("explanation")
    
    def execute_query_and_format(self, sparql_query:str)->Dict[str, Any]:
        """
        execute a SPARQL query
        """
        results = wikidata.sparql_query(sparql_query)
        vars_list = results.get("head", {}).get("vars", [])
        bindings = results .get("results", {}).get("bindings", [])
        
        formatted_rows = []
        for binding in bindings:
            row = {}
            for var in vars_list:
                row[var] = binding.get(var, {}).get("value", "N/A")
            formatted_rows.append(row)
            
        return {
            "columns": vars_list,
            "rows": formatted_rows,
            "raw": results 
        }
    
    def _save_trajectory(self, trajectory: Dict[str, Any]):
        """Saves a pipeline run trajectory to a JSONL file if logging is enabled."""
        if not self.trajectory_logging.get("enabled", False):
            return
        
        log_path = self.trajectory_logging.get("log_path", "trajectories.jsonl")
        
        # Resolve log path relative to the script directory if it's relative
        if not os.path.isabs(log_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_path = os.path.join(base_dir, log_path)
            
        # Ensure parent directory exists
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Warning] Failed to write trajectory log: {e}")

    def run_pipeline(self, question:str, max_retries: int=3) -> Dict[str, Any]:
        """
        Natural language pipeline of natural language to SPARQL query with a correction loop on query execution
        """
        import datetime

        print(f"Extracting entities and properties from the question...")
        entities, properties = self.find_entities_and_properties(question)
        print(f"Extracted Entities: {entities}")
        print(f"Extracted Properties: {properties}")

        print(f"Searching for candidate IDs using the Wikidata API...")

        candidates = self.get_wikidata_ids(entities, properties)
        #print(candidates)

        for entity, items in candidates["entities"].items():
            candidate_list = [f"{item.get('id')} ({item.get('label')})" for item in items]
            print(f" Entity '{entity}' candidates: {candidate_list}")
        for prop, items in candidates["properties"].items():
            candidate_list = [f"{item.get('id')} ({item.get('label')})" for item in items]
            print(f" Property '{prop}' candidates: {candidate_list}")

        print(f"Generating SPARQL query...")
        sparql_query, explanation = self.generate_sparql_query(question, candidates)
        print(f"Initial SPARQL Query:\n{sparql_query}")
        print(f"Explanation: {explanation}")

        # Initialize trajectory structure
        trajectory = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "question": question,
            "entities": entities,
            "properties": properties,
            "candidates": candidates,
            "attempts": [],
            "success": False,
            "final_sparql": None,
            "error": None
        }

        retries = 0
        current_explanation = explanation

        while retries <= max_retries:
            attempt_data = {
                "attempt": retries + 1,
                "sparql": sparql_query,
                "explanation": current_explanation,
                "success": False,
                "error": None
            }
            try:
                print(f"Executing SPARQL query (attempt {retries + 1})...")
                results = self.execute_query_and_format(sparql_query)
                
                # Check for empty results logical warning
                if not results.get("rows"):
                    # Check if we have already attempted this exact query
                    already_attempted = any(
                        att.get("sparql", "").strip() == sparql_query.strip()
                        for att in trajectory["attempts"]
                    )
                    if not already_attempted:
                        raise RuntimeError(
                            "The query executed successfully but returned 0 results. "
                            "This might be a logical error. Common causes include:\n"
                            "1. Reversing subject and object (e.g., '?item wdt:P22 wd:Q1339' instead of 'wd:Q1339 wdt:P22 ?item').\n"
                            "2. Incorrect Wikidata entity ID (Q-number) or property ID (P-number).\n"
                            "3. Overly restrictive triple patterns or FILTER conditions.\n"
                            "If 0 results are indeed correct and expected for this query, generate the EXACT SAME query again to proceed."
                        )
                
                attempt_data["success"] = True
                trajectory["attempts"].append(attempt_data)
                trajectory["success"] = True
                trajectory["final_sparql"] = sparql_query
                self._save_trajectory(trajectory)
                
                return {
                    "success": True,
                    "sparql": sparql_query,
                    "explanation": current_explanation,
                    "results": results
                }
            except Exception as e:
                error_msg = str(e)
                attempt_data["error"] = error_msg
                trajectory["attempts"].append(attempt_data)
                
                print(f"! Query failed with error: {error_msg}")
                if retries == max_retries:
                    trajectory["success"] = False
                    trajectory["final_sparql"] = sparql_query
                    trajectory["error"] = error_msg
                    self._save_trajectory(trajectory)
                    
                    return {
                        "success": False,
                        "sparql": sparql_query,
                        "error": error_msg
                    }
                print(f"Requesting correction from LLM...")
                sparql_query, correction_expl = self.correct_sparql(question, sparql_query, error_msg, candidates)
                current_explanation = correction_expl
                print(f"Corrected SPARQL Query:\n{sparql_query}")
                print(f"Correction Details: {correction_expl}")
                retries += 1
