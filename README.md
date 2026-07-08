# Natural Language to SPARQL for Wikidata 

This program uses Gemini API and Wikidata search to translate natural language queries into SPARQL, executes them on the live Wikidata Query Service endpoint, and presents the results.

---

## Workflow

The pipeline runs the following steps for each query:

1. **Extraction**: Identifies potential entities (e.g. "Feynman") and relations (e.g. "professor") from the question.
2. **Wikidata Search**: Performs API calls using the Wikidata entity search service to find candidate Q-numbers and P-properties matching the extracted names.
3. **Context-Grounded SPARQL Generation**: Sends the user prompt alongside the candidate Q/P IDs and their descriptions to the Gemini model to write a correct SPARQL query.
4. **Execution**: Queries the Wikidata Query Service (`https://query.wikidata.org/sparql`).
5. **Self Correction Loop**: If the query fails due to syntax errors, the error details are sent back to the LLM for correction.


## Setup 

### Prerequisites
- Python 
- A Google Gemini API Key

### Environment Configuration
Create a `.env` file in the root directory and add your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

```bash
python main.py 
```

```bash
Ask Wikidata > Who was the father of Feynman?

Melville Arthur Feynman
```
