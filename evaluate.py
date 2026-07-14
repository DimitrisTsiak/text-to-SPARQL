import os
import re
import time
import json
import datetime
import sys
import argparse
from dataloader import Qald10DataLoader
from typing import List, Dict, Any, Tuple, Set, Iterator
from text_to_sparql_pipeline import TextToSparqlPipeline
import wikidata



class SparqlEvaluator:
    """
    Orchestrates the evaluation of the TextToSparqlPipeline.
    Evaluates Entity/Property Linking, Syntactic Execution Success, and Semantic Correctness.
    """
    def __init__(self, pipeline: TextToSparqlPipeline = None, impose_time_outs = None):
        self.pipeline = pipeline or TextToSparqlPipeline()
        self.impose_time_outs = impose_time_outs

    @staticmethod
    def extract_wikidata_ids(sparql: str) -> Tuple[Set[str], Set[str]]:
        """
        Extract QIDs (entities) and PIDs (properties) from a SPARQL query string using regex.
        """
        qids = set(re.findall(r"\bQ\d+\b", sparql))
        pids = set(re.findall(r"\bP\d+\b", sparql))
        return qids, pids

    @staticmethod
    def check_recall(candidates: Dict[str, List[Dict[str, Any]]], target_ids: Set[str]) -> float:
        """
        Calculates recall of target IDs in the retrieved candidate lists.
        """
        if not target_ids:
            return 1.0  
        
        retrieved_ids = set()
        for term, items in candidates.items():
            for item in items:
                if "id" in item:
                    retrieved_ids.add(item["id"])
                    
        matched = target_ids.intersection(retrieved_ids)
        return len(matched) / len(target_ids)

    @staticmethod
    def get_result_values(raw_results: Dict[str, Any]) -> Set[str]:
        """
        Extract variable values from Wikidata SPARQL JSON response,
        excluding label helper variables (variables ending with 'Label').
        """
        values = set()
        if not raw_results:
            return values
        
        bindings = raw_results.get("results", {}).get("bindings", [])
        for binding in bindings:
            for var, data in binding.items():
                if not var.endswith("Label"):
                    val = data.get("value")
                    if val:
                        values.add(str(val))
        return values

    def evaluate_case(self, question: str, target_sparql: str) -> Dict[str, Any]:
        """
        Evaluates a single test case.
        """
        # Parse target IDs from the target SPARQL query
        target_qids, target_pids = self.extract_wikidata_ids(target_sparql)
        
        start_time = time.time()
        
        entity_recall = 0.0
        property_recall = 0.0
        try:
            entities, properties = self.pipeline.find_entities_and_properties(question)
            candidates = self.pipeline.get_wikidata_ids(entities, properties)
            entity_recall = self.check_recall(candidates.get("entities", {}), target_qids)
            property_recall = self.check_recall(candidates.get("properties", {}), target_pids)
        except Exception as e:
            print(f"  [Warning] Failed to evaluate entity retrieval step: {e}")
            
        pipeline_output = {}
        try:
            pipeline_output = self.pipeline.run_pipeline(question)
        except Exception as e:
            pipeline_output = {
                "success": False,
                "error": f"Pipeline crash: {str(e)}",
                "sparql": ""
            }
            
        elapsed = time.time() - start_time
        
        success = pipeline_output.get("success", False)
        generated_sparql = pipeline_output.get("sparql", "")
        error_msg = pipeline_output.get("error", None)
        
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        semantic_match = False
        
        target_results = None
        gen_results = None
        
        if success:
            try:
                target_results = wikidata.sparql_query(target_sparql)
                target_values = self.get_result_values(target_results)
                
                gen_raw = pipeline_output.get("results", {}).get("raw", {})
                gen_values = self.get_result_values(gen_raw)
                
                if target_values or gen_values:
                    intersection = target_values.intersection(gen_values)
                    precision = len(intersection) / len(gen_values) if gen_values else 0.0
                    recall = len(intersection) / len(target_values) if target_values else 0.0
                    if precision + recall > 0:
                        f1_score = (2 * precision * recall) / (precision + recall)
                    semantic_match = (f1_score == 1.0)
                else:
                    semantic_match = True
                    precision, recall, f1_score = 1.0, 1.0, 1.0
                    
            except Exception as e:
                error_msg = f"Result comparison failed: {str(e)}"
                success = False
                
        return {
            "question": question,
            "success": success,
            "elapsed_sec": elapsed,
            "entity_recall": entity_recall,
            "property_recall": property_recall,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "semantic_match": semantic_match,
            "generated_sparql": generated_sparql,
            "target_sparql": target_sparql,
            "error": error_msg
        }

    def run_evaluation(self, dataloader) -> List[Dict[str, Any]]:
        """
        Runs evaluation on all test cases provided by the data loader.
        """
        results = []
        total = len(dataloader)
        print(f"Starting evaluation on {total} test cases...")
        print("=" * 60)
        
        for idx, (question, target_sparql) in enumerate(dataloader, 1):
            print(f"\n[{idx}/{total}] Question: {question}")
            result = self.evaluate_case(question, target_sparql)
            results.append(result)
            
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"  Status: {status}")
            print(f"  F1-Score: {result['f1_score']:.2f} | Match: {result['semantic_match']}")
            print(f"  Entity Recall: {result['entity_recall']:.2f} | Property Recall: {result['property_recall']:.2f}")
            print(f"  Time taken: {result['elapsed_sec']:.2f}s")
            if result["error"]:
                print(f"  Error: {result['error']}")
            
            # sleep time
            if self.impose_time_outs:
                time.sleep(self.impose_time_outs)
                
        self.print_summary(results)
        return results

    @staticmethod
    def print_summary(results: List[Dict[str, Any]]):
        """
        Prints a summary report of the evaluation results.
        """
        total = len(results)
        if total == 0:
            print("\nNo results to summarize.")
            return
            
        successful_runs = sum(1 for r in results if r["success"])
        semantic_matches = sum(1 for r in results if r["semantic_match"])
        avg_f1 = sum(r["f1_score"] for r in results) / total
        avg_entity_recall = sum(r["entity_recall"] for r in results) / total
        avg_property_recall = sum(r["property_recall"] for r in results) / total
        avg_time = sum(r["elapsed_sec"] for r in results) / total
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY REPORT")
        print("=" * 60)
        print(f"Total test cases:            {total}")
        print(f"Pipeline Success Rate:       {successful_runs}/{total} ({successful_runs/total*100:.1f}%)")
        print(f"Semantic Match Rate (F1=1):  {semantic_matches}/{total} ({semantic_matches/total*100:.1f}%)")
        print(f"Average Answer F1-Score:     {avg_f1:.2f}")
        print(f"Average Entity Recall:       {avg_entity_recall:.2f}")
        print(f"Average Property Recall:     {avg_property_recall:.2f}")
        print(f"Average Execution Latency:   {avg_time:.2f}s")
        print("=" * 60)


def save_evaluation_report(config_path: str, results: List[Dict[str, Any]], dataset_name: str, limit: int = None):
    # Get current timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    
    # Read config file content
    config_content = "Config file not found or not specified."
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_content = f.read()
        except Exception as e:
            config_content = f"Error reading config file: {e}"
            
    # Calculate metrics
    total = len(results)
    successful_runs = sum(1 for r in results if r["success"])
    semantic_matches = sum(1 for r in results if r["semantic_match"])
    avg_f1 = sum(r["f1_score"] for r in results) / total if total > 0 else 0
    avg_entity_recall = sum(r["entity_recall"] for r in results) / total if total > 0 else 0
    avg_property_recall = sum(r["property_recall"] for r in results) / total if total > 0 else 0
    avg_time = sum(r["elapsed_sec"] for r in results) / total if total > 0 else 0
    
    # Construct Markdown Report
    report_md = f"""# SPARQL Evaluation Report - {now.strftime('%Y-%m-%d %H:%M:%S')}

## Summary Metrics
- **Dataset**: {dataset_name} (Limit: {limit if limit is not None else "None"})
- **Total Test Cases**: {total}
- **Pipeline Success Rate**: {successful_runs}/{total} ({successful_runs/total*100:.1f}% if total > 0 else 0%)
- **Semantic Match Rate (F1=1)**: {semantic_matches}/{total} ({semantic_matches/total*100:.1f}% if total > 0 else 0%)
- **Average Answer F1-Score**: {avg_f1:.2f}
- **Average Entity Recall**: {avg_entity_recall:.2f}
- **Average Property Recall**: {avg_property_recall:.2f}
- **Average Execution Latency**: {avg_time:.2f}s

## Configuration Used
```yaml
{config_content}
```

## Detailed Test Cases
"""
    for idx, r in enumerate(results, 1):
        status = "SUCCESS" if r["success"] else "FAILED"
        match_status = "MATCH" if r["semantic_match"] else "MISMATCH"
        report_md += f"""
### [{idx}] Question: {r['question']}
- **Status**: {status} | **Semantic Match**: {match_status} | **Answer F1**: {r['f1_score']:.2f}
- **Entity Recall**: {r['entity_recall']:.2f} | **Property Recall**: {r['property_recall']:.2f} | **Latency**: {r['elapsed_sec']:.2f}s
"""
        if r["error"]:
            report_md += f"- **Error**: `{r['error']}`\n"
        
        report_md += f"""- **Generated SPARQL**:
```sparql
{r['generated_sparql']}
```
- **Target SPARQL**:
```sparql
{r['target_sparql']}
```
---
"""
    
    # Ensure reports directory exists
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    filename = f"evaluation_report_{timestamp_str}.md"
    filepath = os.path.join(reports_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"\n[Info] Saved evaluation report to: {filepath}")
    except Exception as e:
        print(f"\n[Warning] Failed to save evaluation report: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TextToSparqlPipeline")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["qald10"], 
        default="qald10",
        help="The dataset to evaluate on ('qald10')."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of test cases to run. Usefull to avoid API limits"
    )

    parser.add_argument(
        "--time_outs",
        type=int,
        default=None,
        help="Impose a time out after each validation run to avoid rpm limits"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The name of the model (overrides config.yaml)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    pipeline = TextToSparqlPipeline(config_path=args.config, model_name=args.model)
    
    # Initialize the data loader
    if args.dataset == "qald10":
        loader = Qald10DataLoader()
        print(f"Loaded Qal10 dataset with {len(loader.test_cases)} samples for validation")
    else: 
        print(f"Please enter a valid validation dataset name.")
        sys.exit()
    
        
    if args.limit is not None:
        loader.test_cases = loader.test_cases[:args.limit]
    
    # Run evaluator
    evaluator = SparqlEvaluator(pipeline, impose_time_outs=args.time_outs)
    results = evaluator.run_evaluation(loader)
    
    # Save evaluation report
    save_evaluation_report(args.config, results, args.dataset, args.limit)

