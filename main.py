import os
import sys
import argparse
from dotenv import load_dotenv
from text_to_sparql_pipeline import TextToSparqlPipeline




def main():
    parser = argparse.ArgumentParser(description="Query Wikidata using natural language prompts")
    parser.add_argument("--query", required=True, nargs="?", help="The question in natural language: eg. when was Feynman born?")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="the name of the model")
    args = parser.parse_args()

    # print(args.query, args.model)

    pipeline = TextToSparqlPipeline(args.model)
    entities, properties = pipeline.find_entities_and_properties(question=args.query)
    candidates = pipeline.get_wikidata_ids(entities, properties)
    pipeline.generate_sparql_query(args.query, candidates)

    
if __name__ == "__main__":
    main()