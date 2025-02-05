import json
import math
import string
import os

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#############################
# 1) FILE LOADING FUNCTIONS
#############################


def load_json(file_path):
    """
    Loads a JSON file and returns the corresponding dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path):
    """
    Loads a JSONL file (one JSON document per line) and returns a list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_all_files():
    """
    Loads all necessary JSON/JSONL files in the current directory
    and returns a dictionary containing indexed data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    brand_index_path = os.path.join(current_dir, "brand_index.json")
    description_index_path = os.path.join(current_dir, "description_index.json")
    domain_index_path = os.path.join(current_dir, "domain_index.json")
    origin_index_path = os.path.join(current_dir, "origin_index.json")
    origin_synonyms_path = os.path.join(current_dir, "origin_synonyms.json")
    reviews_index_path = os.path.join(current_dir, "reviews_index.json")
    title_index_path = os.path.join(current_dir, "title_index.json")
    products_jsonl_path = os.path.join(current_dir, "products.jsonl")
    rearranged_jsonl_path = os.path.join(current_dir, "rearranged_products.jsonl")
    
    # Load index data
    brand_index = load_json(brand_index_path)
    description_index = load_json(description_index_path)
    domain_index = load_json(domain_index_path)
    origin_index = load_json(origin_index_path)
    origin_synonyms = load_json(origin_synonyms_path)
    reviews_index = load_json(reviews_index_path)
    title_index = load_json(title_index_path)
    
    # Load product data
    products_data = load_jsonl(products_jsonl_path)
    rearranged_products = load_jsonl(rearranged_jsonl_path)
    
    # Aggregate indexes
    indexes = {
        "brand": brand_index,
        "description": description_index,
        "domain": domain_index,
        "origin": origin_index,
        "title": title_index
    }
    
    return {
        "indexes": indexes,
        "reviews_index": reviews_index,
        "origin_synonyms": origin_synonyms,
        "products_data": products_data,
        "rearranged_products": rearranged_products
    }

##################################
# 2) TOKENIZATION & EXPANSION
##################################


def tokenize(text):
    """
    Tokenizes the input text:
    - Converts to lowercase
    - Uses NLTK's word_tokenize for tokenization
    - Removes punctuation
    - Removes stopwords
    Returns a list of cleaned tokens.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    
    eng_stopwords = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    
    cleaned_tokens = [
        token for token in tokens
        if token not in eng_stopwords and token not in punctuation_set
    ]
    return cleaned_tokens


def expand_query_tokens(tokens, synonyms_dict):
    """
    Expands the given tokens using a synonym dictionary.
    If a token has synonyms, they are added as single tokens.
    
    Example:
        synonyms_dict["usa"] = ["united states", "united states of america", "america"]
        => Adds ["united states", "united states of america", "america"]
    """
    expanded_tokens = list(tokens)  # Copy original tokens
    
    for token in tokens:
        if token in synonyms_dict:
            for phrase in synonyms_dict[token]:
                single_token = phrase.lower()
                expanded_tokens.append(single_token)
    
    return expanded_tokens

#############################
# 3) DOCUMENT FILTERING FUNCTIONS
#############################


def get_documents_for_token(token, indexes):
    """
    Retrieves all documents (URLs) containing the given token
    in any of the indexed fields.
    
    Different fields store tokens in different formats:
    - title, description: { token: {"url1": [positions], "url2": [positions], ...} }
    - brand, origin, domain: { token: ["url1", "url2", ...] }
    """
    docs_set = set()
    
    for field_name, field_index in indexes.items():
        if token in field_index:
            value = field_index[token]
            if isinstance(value, dict):
                docs_set.update(value.keys())  # URLs as dictionary keys
            elif isinstance(value, list):
                docs_set.update(value)  # URLs as list elements
            else:
                print(f"[WARN] Unexpected format for '{field_name}' / token '{token}': {type(value)}")
    
    return docs_set


def filter_documents_any(tokens, indexes):
    """
    Returns documents that contain at least one of the given tokens (OR logic).
    """
    if not tokens:
        return set()
    result_docs = set()
    for token in tokens:
        docs_for_token = get_documents_for_token(token, indexes)
        result_docs = result_docs.union(docs_for_token)
    return result_docs


def filter_documents_all(tokens, indexes):
    """
    Returns documents that contain all of the given tokens (AND logic).
    """
    if not tokens:
        return set()
    result_docs = get_documents_for_token(tokens[0], indexes)
    for t in tokens[1:]:
        docs_for_t = get_documents_for_token(t, indexes)
        result_docs = result_docs.intersection(docs_for_t)
        if not result_docs:
            break
    return result_docs


def filter_documents(query, indexes, synonyms_dict, mode="ANY"):
    """
    Filters documents based on the query:
    1) Tokenizes the query
    2) Expands query tokens with synonyms
    3) Applies filtering (ANY / ALL) and returns the matching URLs.
    """
    base_tokens = tokenize(query)
    tokens_expanded = expand_query_tokens(base_tokens, synonyms_dict)
    if mode.upper() == "ANY":
        return filter_documents_any(tokens_expanded, indexes)
    else:
        return filter_documents_all(tokens_expanded, indexes)

#############################
# 4) BM25 & SCORING
#############################


def build_doc_lengths(products_data):
    """
    Constructs a dictionary mapping document URLs to the number of tokens
    (title + description) based on raw product data.
    """
    doc_lengths = {}
    for product in products_data:
        url = product["url"]
        title = product.get("title", "")
        description = product.get("description", "")
        
        combined_text = title + " " + description
        tokens = tokenize(combined_text)  # Reusing the tokenize function
        doc_lengths[url] = len(tokens)
    
    return doc_lengths


def get_term_frequency(token, doc_url, indexes):
    """
    Computes the weighted frequency of a token in a document (doc_url),
    considering different index structures.
    
    indexes: Dictionary with the following structure:
      {
        "title":       dict(token -> dict(doc_url -> [positions])),
        "description": dict(token -> dict(doc_url -> [positions])),
        "brand":       dict(token -> list(doc_url)) OR dict(token -> dict(doc_url -> [positions])),
        "origin":      dict(token -> list(doc_url)),
        "domain":      dict(token -> list(doc_url))
      }
    
    Returns a float: the sum of frequencies in each field weighted accordingly.
    """
    
    # Define weights for each field
    field_weights = {
        "title": 2.0,
        "description": 1.0,
        "brand": 1.0,
        "origin": 1.0,
        "domain": 1.0
    }
    
    freq_total = 0.0
    
    # Iterate through each indexed field
    for field, field_index in indexes.items():
        if token not in field_index:
            continue  # Skip if token is not in index
        
        value = field_index[token]  # Value can be dict or list
        
        if isinstance(value, dict):
            # Case: dict(doc_url -> [positions])
            freq = len(value.get(doc_url, []))
        elif isinstance(value, list):
            # Case: list of doc_url
            freq = 1 if doc_url in value else 0
        else:
            freq = 0  # Unexpected format
        
        weight = field_weights.get(field, 1.0)  # Apply field weight
        freq_total += freq * weight
    
    return freq_total


def compute_idf(token, indexes, N):
    """
    Computes the BM25-style IDF score:
    IDF(t) = log((N - n_t + 0.5) / (n_t + 0.5), base 2).
    
    indexes: Dictionary of indexes with various token mappings.
    N: Total number of documents.
    
    Returns the computed IDF score for the given token.
    """
    doc_set = set()
    
    # Iterate through indexed fields
    for field_name, field_index in indexes.items():
        if token in field_index:
            value = field_index[token]
            
            if isinstance(value, dict):
                doc_set.update(value.keys())  # Add URLs (keys)
            elif isinstance(value, list):
                doc_set.update(value)  # Add URLs directly
            else:
                print(f"[WARN] Unexpected format in compute_idf for '{field_name}' / '{token}': {type(value)}")
    
    n_t = len(doc_set)  # Number of documents containing the token
    if n_t == 0:
        return 0.0
    
    return math.log((N - n_t + 0.5) / (n_t + 0.5), 2)


def bm25_score(doc_url, tokens, indexes, doc_lengths, avgdl, k=1.2, b=0.75):
    """
    Computes the BM25 score for a given document (doc_url) based on a list of tokens.
    """
    score = 0.0
    N = len(doc_lengths)
    doc_len = doc_lengths.get(doc_url, 0)
    
    for t in tokens:
        idf_t = compute_idf(t, indexes, N)
        freq_t = get_term_frequency(t, doc_url, indexes)
        
        numerator = freq_t * (k + 1)
        denominator = freq_t + k * (1 - b + b * (doc_len / avgdl))
        partial = 0.0
        if denominator != 0:
            partial = idf_t * (numerator / denominator)
        score += partial
    
    return score

#############################
# 5) ADDITIONAL SIGNALS
#############################


def get_review_bonus(doc_url, reviews_index, max_bonus=1.0):
    """
    Adds a bonus score proportional to the average review rating (0 to 5).
    The bonus is calculated as (mean_mark / 5) * max_bonus.
    """
    if doc_url not in reviews_index:
        return 0.0
    mean_mark = reviews_index[doc_url].get("mean_mark", 0)
    return (mean_mark / 5.0) * max_bonus


def is_exact_title_match(doc_url, query_tokens, products_info):
    """
    Checks if the document's title (tokenized) exactly matches
    the tokenized set of query terms.
    """
    if doc_url not in products_info:
        return False
    
    title_text = products_info[doc_url].get("title", "")
    title_tokens = set(tokenize(title_text))
    query_tokens_set = set(query_tokens)
    
    return (title_tokens == query_tokens_set) and len(title_tokens) > 0


def compute_final_score(doc_url, query_tokens, indexes, doc_lengths, avgdl,
                        reviews_index, products_info,
                        alpha=1.0, beta=0.3, gamma=1.0):
    """
    Computes the final ranking score:
    final_score = (alpha * BM25) + (beta * review_bonus) + (gamma * exact_match_indicator)
    
    - alpha, beta, gamma are configurable weights.
    """
    bm25_val = bm25_score(doc_url, query_tokens, indexes, doc_lengths, avgdl)
    review_b = get_review_bonus(doc_url, reviews_index, max_bonus=1.0)
    exact_m = 1.0 if is_exact_title_match(doc_url, query_tokens, products_info) else 0.0
    
    final = alpha * bm25_val + beta * review_b + gamma * exact_m
    return final


#############################
# 6) DOCUMENT RANKING FUNCTION
#############################

def rank_documents(query, indexes, reviews_index, products_data_dict,
                   doc_lengths, avgdl, synonyms_dict, mode="ANY"):
    """
    Ranks documents based on the query:
    1) Filters documents using (ANY or ALL) logic.
    2) Computes a final score (BM25 + additional signals).
    3) Sorts the documents in descending order of score.
    """
    # Filter documents based on query and synonym expansion
    filtered_docs = filter_documents(query, indexes, synonyms_dict, mode=mode)
    
    # Tokenize query for exact match checking
    query_tokens = tokenize(query)
    
    results = []
    for doc_url in filtered_docs:
        score = compute_final_score(
            doc_url=doc_url,
            query_tokens=query_tokens,
            indexes=indexes,
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            reviews_index=reviews_index,
            products_info=products_data_dict,
            alpha=1.0,     # BM25 weight
            beta=0.3,      # Review score weight
            gamma=1.0      # Exact match weight
        )
        results.append((doc_url, score))
    
    # Sort by descending score
    results.sort(key=lambda x: x[1], reverse=True)
    return results

#############################
# 7) SEARCH RESULT JSON GENERATION
#############################


def produce_search_results_json(query, indexes, reviews_index,
                                products_list, synonyms_dict,
                                mode="ANY"):
    """
    Generates a JSON output containing ranked search results:
    1) Computes document lengths and average document length.
    2) Prepares a dictionary {url: {title:..., description:...}} for fast access.
    3) Ranks the documents.
    4) Constructs the final JSON output.
    """
    # Compute document lengths and average document length
    doc_lengths = build_doc_lengths(products_list)
    total_docs = len(doc_lengths)
    if total_docs == 0:
        return {
            "metadata": {
                "total_documents": 0,
                "filtered_documents": 0,
                "query": query
            },
            "results": []
        }
    avgdl = sum(doc_lengths.values()) / total_docs
    
    # Prepare dictionary for quick lookup of title and description
    products_data_dict = {}
    for p in products_list:
        products_data_dict[p["url"]] = {
            "title": p.get("title", ""),
            "description": p.get("description", "")
        }
    
    # Rank documents based on query
    ranked_results = rank_documents(query, indexes, reviews_index,
                                    products_data_dict, doc_lengths, avgdl,
                                    synonyms_dict, mode)
    
    # Construct final JSON structure
    docs_filtered = len(ranked_results)
    output = {
        "metadata": {
            "total_documents": total_docs,
            "filtered_documents": docs_filtered,
            "query": query
        },
        "results": []
    }
    
    for doc_url, score in ranked_results:
        # Retrieve title and description
        title = products_data_dict[doc_url]["title"]
        description = products_data_dict[doc_url]["description"]
        
        output["results"].append({
            "url": doc_url,
            "title": title,
            "description": description,
            "score": score
        })
        
    return output


#############################
# 8) MAIN TEST SCRIPT
#############################

if __name__ == "__main__":
    """
    Main script to test the search functionality:
    - Downloads necessary NLTK resources if not already available.
    - Loads all required data files.
    - Executes a sample query.
    - Prints the formatted JSON output of the search results.
    """
    
    # Download NLTK resources if not already available
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    
    # Load all necessary files
    data = load_all_files()
    indexes = data["indexes"]
    reviews_index = data["reviews_index"]
    origin_synonyms = data["origin_synonyms"]
    products_data = data["products_data"]
    
    # Example search query
    user_query = "chocolate from usa"
    
    # Filtering mode: "ANY" (match at least one token) or "ALL" (match all tokens)
    filter_mode = "ANY"
    
    # Generate final search results in JSON format
    results_json_dict = produce_search_results_json(
        query=user_query,
        indexes=indexes,
        reviews_index=reviews_index,
        products_list=products_data,
        synonyms_dict=origin_synonyms,
        mode=filter_mode
    )
    
    # Pretty print the JSON output
    print(json.dumps(results_json_dict, indent=2, ensure_ascii=False))