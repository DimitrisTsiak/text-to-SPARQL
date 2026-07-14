import requests
from typing import List, Dict, Any

URL = "https://query.wikidata.org/sparql"


def search_wikidata(search_term:str, search_type: str, limit: int = 4) -> List[Dict[str, Any]]:
    """
    Search wikidata for entities or properties

    Args:
        search_term: the tearm to search (eg. Feynman)
        search_type: item(Q) or property(P)
        limit: max number of returns

    Returns:
        List of Dictionaries containing id, label, description and url
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": search_term,
        "language": "en",
        "format": "json",
        "type": search_type,
        "limit": limit
    }
    headers = {
        "User-Agent": "nl-to-SPARQL-agent/1.0 (https://github.com/google/antigravity; mailto:agent-nl-sparql@example.com)"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("search", []):
            res_item = {
                "id": item.get("id"),
                "label": item.get("label", "No Label"),
                "description": item.get("description", "No Description"),
                "url": item.get("concepturi", f"https://www.wikidata.org/wiki/{item.get('id')}")
            }
            if "datatype" in item:
                res_item["datatype"] = item["datatype"]
            results.append(res_item)
        # print(results['id'])
        return results

    except Exception as e:
        print(f"Warning: Wikidata search failed for '{search_term}': {e}")
        return []

def sparql_query(query:str) -> Dict[str, Any]:
    """
    Args:
        query: The SPARQL query
    Returns:
        A dict containing the JSON respoce from endpoint
    """
    headers = {
        "User-Agent": "nl-to-SPARQL-agent/1.0 (https://github.com/google/antigravity; mailto:agent-nl-sparql@example.com)",
        "Accept": "application/sparql-results+json"
    }
    query = query.strip()
    response = requests.get(URL, params={"query": query}, headers=headers, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Wikidata Query Service returned status {response.status_code}\n"
            f"Response: {response.text[:500]}"
        )
    # print(f" The respone: {response.json}")
    # print(response.text)
    return response.json()

# if __name__ == "__main__":
#     # query = "SELECT ?child WHERE { ?child wdt:P22 wd:Q1339. }"
#     # sparql_query(query)
#     search_wikidata("Feynman", "item", limit=10)

