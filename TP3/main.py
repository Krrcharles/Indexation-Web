import json
import math
import string
import os

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#################################
# 1) FILE LOADING FUNCTIONS
#################################

def load_json(file_path):
    """
    Loads a JSON file and returns its content as a dictionary.
    
    Parameters:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Parsed JSON content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path):
    """
    Loads a JSONL file (one JSON document per line) and returns a list of dictionaries.
    
    Parameters:
        file_path (str): Path to the JSONL file.
        
    Returns:
        list: A list of dictionaries representing each JSON document.
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
    Loads all necessary JSON/JSONL files from the current directory and aggregates the data.
    
    Returns:
        dict: A dictionary containing:
              - "indexes": a dictionary with field indexes,
              - "reviews_index": review data,
              - "origin_synonyms": synonym mappings,
              - "products_data": list of product documents,
              - "rearranged_products": list of rearranged products.
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

#################################
# 2) TOKENIZATION & EXPANSION
#################################

def tokenize(text):
    """
    Tokenizes the input text by converting it to lowercase, splitting into words,
    and removing punctuation and stopwords.
    
    Parameters:
        text (str): The text to tokenize.
        
    Returns:
        list: A list of cleaned tokens.
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
    Expands the list of query tokens using the provided synonym dictionary.
    
    Parameters:
        tokens (list): List of tokens from the query.
        synonyms_dict (dict): Dictionary mapping tokens to their synonyms.
        
    Returns:
        list: The expanded list of tokens including synonyms.
    """
    expanded_tokens = list(tokens)  # Copy original tokens
    
    for token in tokens:
        if token in synonyms_dict:
            for phrase in synonyms_dict[token]:
                expanded_tokens.append(phrase.lower())
    
    return expanded_tokens

#################################
# 3) DOCUMENT FILTERING FUNCTIONS
#################################

def get_documents_for_token(token, indexes):
    """
    Retrieves all documents (URLs) that contain the given token in any of the field indexes.
    
    Parameters:
        token (str): The token to search for.
        indexes (dict): Dictionary of indexes by field.
        
    Returns:
        set: A set of document URLs that contain the token.
    """
    docs_set = set()
    
    for field_name, field_index in indexes.items():
        if token in field_index:
            value = field_index[token]
            if isinstance(value, dict):
                docs_set.update(value.keys())
            elif isinstance(value, list):
                docs_set.update(value)
            else:
                print(f"[WARN] Unexpected format for field '{field_name}' / token '{token}': {type(value)}")
    
    return docs_set


def filter_documents_any(tokens, indexes):
    """
    Returns documents that contain at least one of the given tokens (OR logic).
    
    Parameters:
        tokens (list): List of tokens.
        indexes (dict): Dictionary of indexes by field.
        
    Returns:
        set: A set of document URLs matching any of the tokens.
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
    
    Parameters:
        tokens (list): List of tokens.
        indexes (dict): Dictionary of indexes by field.
        
    Returns:
        set: A set of document URLs matching all of the tokens.
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
    Filters documents based on the query after tokenizing and expanding with synonyms.
    
    Parameters:
        query (str): The search query.
        indexes (dict): Dictionary of indexes by field.
        synonyms_dict (dict): Synonym dictionary for query expansion.
        mode (str): Filtering mode, either "ANY" or "ALL".
        
    Returns:
        set: A set of document URLs that match the query.
    """
    base_tokens = tokenize(query)
    tokens_expanded = expand_query_tokens(base_tokens, synonyms_dict)
    if mode.upper() == "ANY":
        return filter_documents_any(tokens_expanded, indexes)
    else:
        return filter_documents_all(tokens_expanded, indexes)

#################################
# 4) BM25 & SCORING FUNCTIONS
#################################

def compute_field_idf(token, field_index, N):
    """
    Computes the logarithmic IDF (base 2) for a token in a specific field index.
    IDF_field(t) = log((N - n_t + 0.5) / (n_t + 0.5), base 2)
    
    Parameters:
        token (str): The token for which to compute the IDF.
        field_index (dict): The index for the field (a dictionary mapping token -> [documents]
                            or token -> dict(doc_url -> positions)).
        N (int): Total number of documents (for this field).
        
    Returns:
        float: The computed IDF value for the token.
    """
    if token not in field_index:
        return 0.0
    value = field_index[token]
    if isinstance(value, dict):
        n_t = len(value)
    elif isinstance(value, list):
        n_t = len(value)
    else:
        n_t = 0
    if n_t == 0:
        return 0.0
    return math.log((N - n_t + 0.5) / (n_t + 0.5), 2)


def bm25_field_score(doc_url, tokens, field, field_index, doc_field_lengths, avgdl_field, k=1.2, b=0.75):
    """
    Computes the BM25 contribution for a document (doc_url) within a specific field.
    
    Parameters:
        doc_url (str): The URL of the document.
        tokens (list): List of query tokens.
        field (str): The name of the field (e.g., "title", "description", etc.).
        field_index (dict): The index specific to the field.
        doc_field_lengths (dict): A dictionary mapping doc_url to the field's length (in tokens).
        avgdl_field (float): The average length of the field.
        k (float): BM25 parameter k (default is 1.2).
        b (float): BM25 parameter b (default is 0.75).
        
    Returns:
        float: The sum of the BM25 contributions for this field.
    """
    score = 0.0
    N = len(doc_field_lengths)
    
    for t in tokens:
        idf = compute_field_idf(t, field_index, N)
        
        # Retrieve the frequency of the token in this field for the document
        freq = 0
        if t in field_index:
            value = field_index[t]
            if isinstance(value, dict):
                freq = len(value.get(doc_url, []))
            elif isinstance(value, list):
                freq = 1 if doc_url in value else 0
        
        doc_len = doc_field_lengths.get(doc_url, 0)
        numerator = freq * (k + 1)
        denominator = freq + k * (1 - b + b * (doc_len / avgdl_field)) if avgdl_field > 0 else freq + k
        if denominator != 0:
            score += idf * (numerator / denominator)
    return score


def bm25_score(doc_url, tokens, indexes, products_data, k=1.2, b=0.75):
    """
    Computes the BM25 score for a document (doc_url) by summing the BM25 scores
    calculated separately for each field and then weighting them.
    
    Parameters:
        doc_url (str): The URL of the document.
        tokens (list): List of query tokens.
        indexes (dict): Dictionary of indexes by field.
        products_data (list): The complete list of products (used to build field lengths for "title" and "description").
        k (float): BM25 parameter k (default is 1.2).
        b (float): BM25 parameter b (default is 0.75).
        
    Returns:
        float: The overall BM25 score for the document.
    """
    total_score = 0.0
    # Field weights (adjustable if needed)
    field_weights = {
        "title": 2.0,
        "description": 1.0,
        "brand": 1.0,
        "origin": 1.0,
        "domain": 1.0
    }
    
    # For each field in the index, compute the field-specific BM25 score
    for field, field_index in indexes.items():
        # For textual fields in products_data, compute the token length
        if field in ["title", "description"]:
            doc_field_lengths = {}
            for product in products_data:
                url = product["url"]
                text = product.get(field, "")
                tokens_field = tokenize(text)
                doc_field_lengths[url] = len(tokens_field)
        else:
            # For other fields (brand, origin, domain), assume a default length of 1
            doc_field_lengths = {product["url"]: 1 for product in products_data}
        
        total_length = sum(doc_field_lengths.values())
        count_docs = len(doc_field_lengths)
        avgdl_field = total_length / count_docs if count_docs > 0 else 0
        
        bm25_field = bm25_field_score(doc_url, tokens, field, field_index, doc_field_lengths, avgdl_field, k, b)
        weight = field_weights.get(field, 1.0)
        total_score += weight * bm25_field
    return total_score


def get_review_bonus(doc_url, reviews_index, max_bonus=1.0):
    """
    Adds a bonus score proportional to the average review rating (ranging from 0 to 5).
    The bonus is calculated as (mean_mark / 5) * max_bonus.
    
    Parameters:
        doc_url (str): The URL of the document.
        reviews_index (dict): Dictionary containing review data with doc_url as keys.
        max_bonus (float): The maximum bonus score (default is 1.0).
        
    Returns:
        float: The review bonus score.
    """
    if doc_url not in reviews_index:
        return 0.0
    mean_mark = reviews_index[doc_url].get("mean_mark", 0)
    return (mean_mark / 5.0) * max_bonus


def is_exact_title_match(doc_url, query_tokens, products_info):
    """
    Checks if the tokenized title of the document exactly matches
    the tokenized set of query terms.
    
    Parameters:
        doc_url (str): The URL of the document.
        query_tokens (list): List of query tokens.
        products_info (dict): Dictionary {doc_url: {title: ..., description: ...}} used for exact matching.
        
    Returns:
        bool: True if the tokenized title exactly matches the query tokens, False otherwise.
    """
    if doc_url not in products_info:
        return False
    
    title_text = products_info[doc_url].get("title", "")
    title_tokens = set(tokenize(title_text))
    query_tokens_set = set(query_tokens)
    
    return (title_tokens == query_tokens_set) and len(title_tokens) > 0


def compute_final_score(doc_url, query_tokens, indexes, products_data,
                        reviews_index, products_info,
                        bm25_weight=1.0, review_weight=0.3, exact_match_weight=1.0,
                        k=1.2, b=0.75):
    """
    Computes the final ranking score:
    
        final_score = (bm25_weight * BM25) + (review_weight * review_bonus) + (exact_match_weight * exact_match_indicator)
    
    where:
      - BM25 is the weighted sum of the BM25 scores across fields.
      - products_data is the complete list of products (used for BM25 calculation per field).
      - products_info is a dictionary {doc_url: {title: ..., description: ...}} used for exact matching.
    
    Parameters:
        doc_url (str): The URL of the document.
        query_tokens (list): List of query tokens.
        indexes (dict): Dictionary of indexes by field.
        products_data (list): The complete list of products.
        reviews_index (dict): Dictionary of review data.
        products_info (dict): Dictionary containing product information (e.g., title and description).
        bm25_weight (float): Weight for the BM25 score (default is 1.0).
        review_weight (float): Weight for the review bonus (default is 0.3).
        exact_match_weight (float): Weight for the exact title match indicator (default is 1.0).
        k (float): BM25 parameter k (default is 1.2).
        b (float): BM25 parameter b (default is 0.75).
        
    Returns:
        float: The final ranking score for the document.
    """
    bm25_val = bm25_score(doc_url, query_tokens, indexes, products_data, k, b)
    review_b = get_review_bonus(doc_url, reviews_index, max_bonus=1.0)
    exact_m = 1.0 if is_exact_title_match(doc_url, query_tokens, products_info) else 0.0
    
    final = bm25_weight * bm25_val + review_weight * review_b + exact_match_weight * exact_m
    return final

#################################
# 5) DOCUMENT RANKING FUNCTION
#################################

def rank_documents(query, indexes, reviews_index, products_info, products_data, synonyms_dict, mode="ANY"):
    """
    Ranks documents based on the query by filtering them and computing the final ranking score.
    
    Parameters:
        query (str): The search query.
        indexes (dict): Dictionary of indexes by field.
        reviews_index (dict): Dictionary of review data.
        products_info (dict): Dictionary {doc_url: {title: ..., description: ...}} for quick access.
        products_data (list): The complete list of products.
        synonyms_dict (dict): Synonym dictionary for query expansion.
        mode (str): Filtering mode ("ANY" or "ALL").
        
    Returns:
        list: A list of tuples (doc_url, score) sorted in descending order by score.
    """
    # Filter documents based on query and synonym expansion
    filtered_docs = filter_documents(query, indexes, synonyms_dict, mode=mode)
    
    # Tokenize query for exact matching
    query_tokens = tokenize(query)
    
    results = []
    for doc_url in filtered_docs:
        score = compute_final_score(
            doc_url=doc_url,
            query_tokens=query_tokens,
            indexes=indexes,
            products_data=products_data,
            reviews_index=reviews_index,
            products_info=products_info,
            bm25_weight=1.0,         # BM25 score weight
            review_weight=0.3,         # Review bonus weight
            exact_match_weight=1.0,    # Exact match weight
            k=1.2,
            b=0.75
        )
        results.append((doc_url, score))
    
    # Sort results by descending score
    results.sort(key=lambda x: x[1], reverse=True)
    return results

#################################
# 6) SEARCH RESULT JSON GENERATION
#################################

def produce_search_results_json(query, indexes, reviews_index, products_list, synonyms_dict, mode="ANY"):
    """
    Generates a JSON object containing the ranked search results.
    
    Parameters:
        query (str): The search query.
        indexes (dict): Dictionary of indexes by field.
        reviews_index (dict): Dictionary of review data.
        products_list (list): List of product dictionaries.
        synonyms_dict (dict): Synonym dictionary for query expansion.
        mode (str): Filtering mode ("ANY" or "ALL").
        
    Returns:
        dict: A dictionary representing the search results with metadata and results.
    """
    # Prepare a dictionary for quick access to product title and description
    products_data_dict = {}
    for product in products_list:
        products_data_dict[product["url"]] = {
            "title": product.get("title", ""),
            "description": product.get("description", "")
        }
    
    # Rank documents based on the query
    ranked_results = rank_documents(query, indexes, reviews_index, products_data_dict, products_list, synonyms_dict, mode)
    
    # Construct final JSON structure
    output = {
        "metadata": {
            "total_documents": len(products_list),
            "filtered_documents": len(ranked_results),
            "query": query
        },
        "results": []
    }
    
    for doc_url, score in ranked_results:
        title = products_data_dict[doc_url]["title"]
        description = products_data_dict[doc_url]["description"]
        output["results"].append({
            "url": doc_url,
            "title": title,
            "description": description,
            "score": score
        })
    
    return output

#################################
# 7) MAIN TEST SCRIPT
#################################

if __name__ == "__main__":
    """
    Main script to test the search functionality.
    
    This script downloads the necessary NLTK resources (if not already available),
    loads all required data files, executes a sample query, and prints the JSON
    formatted search results.
    """
    # Download NLTK resources if not already available
    nltk.download('punkt')
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
    
    # Print the formatted JSON output
    print(json.dumps(results_json_dict, indent=2, ensure_ascii=False))
