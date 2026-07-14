import os
import sys
import argparse
from dotenv import load_dotenv
from text_to_sparql_pipeline import TextToSparqlPipeline
import web_server


def print_results(results_data):
    """Utility to print results in a readable format."""
    results = results_data.get("results", {})
    columns = results.get("columns", [])
    rows = results.get("rows", [])
    if not rows:
        print("\n No results found for this query :(")
        return
    print(f"\nSuccess! Found {len(rows)} results:")
    print("-~" * 40)

    col_widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val_str = str(row.get(col, ""))
            if len(val_str) > 40:
                val_str = val_str[:37] + "..."
            col_widths[col] = max(col_widths[col], len(val_str))
            
    # Print header
    header_str = " | ".join(col.ljust(col_widths[col]) for col in columns)
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in rows:
        row_cells = []
        for col in columns:
            val_str = str(row.get(col, ""))
            if len(val_str) > 40:
                val_str = val_str[:37] + "..."
            row_cells.append(val_str.ljust(col_widths[col]))
        print(" | ".join(row_cells))
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Query Wikidata using natural language prompts")
    parser.add_argument("--query", required=False, nargs="?", help="The question in natural language: eg. when was Feynman born?")
    parser.add_argument("--model", type=str, default=None, help="the name of the model (overrides config.yaml)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--web", action="store_true", help="Launch the web interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the web server on")
    args = parser.parse_args()

    try:
        pipeline = TextToSparqlPipeline(config_path=args.config, model_name=args.model)
    except Exception as e:
        print(f"There was an error initializing the text to SPARQL pipeline: {e}")
        sys.exit()

    if args.web:
        web_server.start_server(pipeline, port=args.port)
        return

    if args.query:
        question = args.query
        print(f"\nUser Question: {question}")
        print("=" * 60)
        try:
            output = pipeline.run_pipeline(question)
            if output and output.get("success"):
                print_results(output)
            else:
                print(f"\n! Failed to answer the question. Error: {output.get('error')}")
        except Exception as e:
            print(f"\n! An error occurred during pipeline run: {e}")
        return

    print("\n--- Wikidata Natural Language SPARQL Assistant ---")
    print("Select Interface Mode:")
    print("  1. Interactive Command Line Interface (CLI)")
    print("  2. Web Interface (Browser)")
    
    try:
        choice = input("\nEnter choice 1 or 2: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        sys.exit()

    if choice == "2":
        web_server.start_server(pipeline, port=args.port)
    else:
        print("\n--- Wikidata Natural Language SPARQL CLI Interface ---")
        print("Type your question and press Enter. Type 'exit' or 'quit' to close.")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nAsk Wikidata > ").strip()
                if not question:
                    continue
                if question.lower() in ("exit", "quit"):
                    break
                
                print("\n" + "=" * 60)
                output = pipeline.run_pipeline(question)
                if output and output.get("success"):
                    print_results(output)
                else:
                    print(f"\n ! Failed to answer the question. Error: {output.get('error')}")
                print("=" * 60)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\n! Error: {e}")
    
if __name__ == "__main__":
    main()