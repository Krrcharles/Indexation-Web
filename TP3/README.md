
# TP3 - Web Indexation ENSAI 2025

  

## Objective

The goal of this project is to develop a search engine that leverages previously created indexes to return and rank relevant search results. The search engine uses various signals (e.g., BM25, review scores, and exact title matching) to compute a ranking score for each document.

  

## Input Data

  

-  **Indexes:** JSON files containing field-specific indexes (e.g., `brand_index.json`, `description_index.json`, `domain_index.json`, `origin_index.json`, and `title_index.json`).

-  **Products Data:** A modified `products.jsonl` file where:

- All documents now have an origin (if the product has a `product_features` dictionary).

- The variant name appears in the title after a hyphen.

- If the description had more than two sentences, one sentence was randomly removed.

-  **Synonyms:** A JSON file containing synonyms for country names (and potentially other terms).

-  **Reviews:** A JSON file (`reviews_index.json`) containing review data for the products.

  

## Project Structure

  

```

main.py

```

Contains the complete implementation of the search engine including:

- File loading functions.

- Tokenization and query expansion.

- Document filtering (both "ANY" and "ALL" modes).

- BM25-based ranking with additional signals (review bonus and exact title matching).

- JSON generation of the final search results.


```

final_results.json

```

The output file where the final JSON-formatted search results will be exported.

  

## How to Run

  

### Prerequisites

-  **Python 3.x**

-  **NLTK Library**

- Ensure that NLTK and its necessary resources (such as `punkt` and `stopwords`) are installed. If not, they will be downloaded automatically on the first run.

  

### Installation

  

Clone or download the project repository and install the required dependencies:

  

```sh

pip  install  nltk

```

  

### Running the Search Engine

  

Ensure the following files are in the same directory as `main.py`:

  

- The index files (`brand_index.json`, `description_index.json`, `domain_index.json`, `origin_index.json`, `title_index.json`)

- The synonym file (`origin_synonyms.json`)

- The reviews file (`reviews_index.json`)

- The products data file (`products.jsonl`)

- The rearranged products file (`rearranged_products.jsonl`)

  

Run the main script:

  

```sh

python  main.py

```

  

The script will:

- Download the necessary NLTK resources (if not already present).

- Load all required files.

- Execute a sample query (e.g., "chocolate from usa").

- Print the JSON-formatted search results to the console.

  

To export the results to a file, the following line is used in the code:

  

```python

with  open("final_results.json", "w", encoding="utf-8") as f:

json.dump(results_json_dict, f, indent=2, ensure_ascii=False)

```

  

This will create (or overwrite) a file named `final_results.json` in the project directory.

  

## Configuration & Tuning

  

### BM25 Parameters:

The BM25 parameters `k` and `b` can be adjusted to tune the influence of token frequency and document length.

  

### Weighting Signals:

The final ranking score is computed with a linear combination of:

-  **BM25 Score** (`bm25_weight`, default: `1.0`)

-  **Review Bonus** (`review_weight`, default: `0.3`)

-  **Exact Title Match** (`exact_match_weight`, default: `1.0`)

  

These parameters can be modified in the function calls (e.g., in `compute_final_score`) to optimize search relevance.

  

## Output Format

  

The final results are formatted as a JSON object with the following structure:

  

```json

{

"metadata": {

"total_documents": <int>,

"filtered_documents": <int>,

"query": "<search_query>"

},

"results": [

{

"url": "<document_url>",

"title": "<document_title>",

"description": "<document_description>",

"score": "<ranking_score>"

}

]

}
