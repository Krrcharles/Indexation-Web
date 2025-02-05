import os
import json
import string
from urllib.parse import urlparse, parse_qs
from datetime import datetime

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def read_jsonl_file(file_path):
    """
    Read lines from a JSONL file and parse each line as JSON.
    Returns a list of document dictionaries.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Ignore empty lines
                doc = json.loads(line)
                documents.append(doc)
    return documents


def get_product_id_and_variant(url):
    """
    Parse the product URL to extract:
      - product ID (if the path matches '/product/<id>')
      - variant (if present in query params, e.g. '?variant=some-variant')

    Returns a tuple: (product_id, variant)
    If not found, product_id or variant might be None.
    """
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    product_id = None
    variant = None

    # Check if path looks like 'product/<id>'
    if len(path_parts) == 2 and path_parts[0] == 'product':
        try:
            product_id = int(path_parts[1])
        except ValueError:
            product_id = None

    # Check query string for variant
    query_params = parse_qs(parsed_url.query)
    if 'variant' in query_params:
        variant_values = query_params.get('variant')
        if variant_values:
            variant = variant_values[0]

    return product_id, variant


def enrich_docs_with_product_info(docs):
    """
    Enrich each document with 'product_id' and 'variant' extracted from its 'url'.
    Returns the same list of documents, updated in place.
    """
    for doc in docs:
        url = doc.get('url', '')
        product_id, variant = get_product_id_and_variant(url)
        doc['product_id'] = product_id
        doc['variant'] = variant
    return docs


def tokenize_and_clean(text):
    """
    Tokenize the text, convert it to lowercase,
    remove punctuation and stopwords.
    
    Returns a list of clean tokens.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    
    cleaned_tokens = [
        token for token in tokens
        if token not in stop_words and token not in punctuation_set
    ]
    return cleaned_tokens


def build_inverted_index(docs, field_name):
    """
    Build an inverted index for a specific field (e.g., 'title' or 'description').
    Each key in the returned dictionary is a token (string),
    and the value is a list of URLs where the token appears.
    """
    inverted_index = {}

    for doc in docs:
        url = doc.get("url", "")
        field_content = doc.get(field_name, "")
        
        tokens = tokenize_and_clean(field_content)
        unique_tokens = set(tokens)
        
        for token in unique_tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(url)
    
    return inverted_index


def get_reviews_summary(product_reviews):
    """
    Compute:
      - The total number of reviews (review_count),
      - The average rating (average_rating),
      - The last rating (last_rating) by the most recent date.

    Returns a dict:
    {
        "review_count":  <int>,
        "average_rating":  <float>,
        "last_rating":  <int or float>
    }
    If product_reviews is empty, returns default values.
    """
    if not product_reviews:
        return {
            "review_count": 0,
            "average_rating": 0.0,
            "last_rating": None
        }
    
    review_count = len(product_reviews)
    ratings = [rev.get("rating", 0) for rev in product_reviews]
    average_rating = sum(ratings) / review_count

    # Sort reviews by date (YYYY-MM-DD) to find the most recent
    try:
        sorted_reviews = sorted(
            product_reviews,
            key=lambda r: datetime.strptime(r.get("date", ""), "%Y-%m-%d")
        )
        last_rating = sorted_reviews[-1].get("rating", None)
    except ValueError:
        # If date is invalid or missing, use the last in the list
        last_rating = product_reviews[-1].get("rating", None)

    return {
        "review_count": review_count,
        "average_rating": average_rating,
        "last_rating": last_rating
    }


def build_reviews_index(docs):
    """
    Build a reviews index (not inverted).
    Key = URL, Value = {
      "review_count": <int>,
      "average_rating": <float>,
      "last_rating": <int or float>
    }
    """
    reviews_index = {}
    
    for doc in docs:
        url = doc.get("url", "")
        product_reviews = doc.get("product_reviews", [])
        
        summary = get_reviews_summary(product_reviews)
        reviews_index[url] = summary
    
    return reviews_index


def build_feature_indexes(docs):
    """
    Automatically build an index for ALL features found
    in doc["product_features"] across all documents.

    Returns a dictionary of the form:
    {
      "brand": {
        "ChocoDelight": [url1, url2],
        "MegaChoc": [url3, ...]
      },
      "material": {
        "Premium quality chocolate": [url1, ...]
      },
      ...
    }
    """
    all_features_index = {}
    
    for doc in docs:
        
        product_features = doc.get("product_features", {})
        
        # For each (feature_name, feature_value) in product_features
        for feature_name, feature_value in product_features.items():
            if feature_name not in all_features_index:
                all_features_index[feature_name] = {}
            
            if feature_value not in all_features_index[feature_name]:
                all_features_index[feature_name][feature_value] = []
            
            all_features_index[feature_name][feature_value].append(doc.get("url"))
    
    return all_features_index


def save_dict_to_json(data_dict, output_path):
    """
    Save a Python dictionary to a JSON file with indentation,
    creating directories if needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2)


def build_positional_inverted_index(docs, field_name):
    """
    Build a positional inverted index for a specific field (e.g., 'title', 'description').
    This index stores the positions of each token within each document.

    Returns a dict of the form:
    {
      token: {
        doc_identifier: [pos1, pos2, pos3, ...],
        ...
      },
      ...
    }
    """
    pos_index = {}
    stop_words = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    
    for doc in docs:
        url = doc.get("url", "")
        text = doc.get(field_name, "")
        
        tokens = word_tokenize(text.lower())
        
        position = 0
        for raw_token in tokens:
            # Filter stopwords and punctuation
            if raw_token in stop_words or raw_token in punctuation_set:
                position += 1
                continue
            
            if raw_token not in pos_index:
                pos_index[raw_token] = {}
            if url not in pos_index[raw_token]:
                pos_index[raw_token][url] = []
            
            pos_index[raw_token][url].append(position)
            
            position += 1
    
    return pos_index


def build_positional_indexes_for_fields(docs, fields):
    """
    Build positional inverted indexes for multiple fields (e.g., ['title', 'description']).
    Returns a dict of the form:
    {
      "title": { ...positional index... },
      "description": { ...positional index... }
    }
    """
    result = {}
    for field_name in fields:
        result[field_name] = build_positional_inverted_index(docs, field_name)
    return result


if __name__ == "__main__":
    # 1) Read the JSONL file
    docs = read_jsonl_file("products.jsonl")
    
    # 2) Enrich documents with product_id and variant
    docs = enrich_docs_with_product_info(docs)
    
    # Prepare base output folders
    base_output_folder = "output"
    features_output_folder = os.path.join(base_output_folder, "features")
    
    # 3) Build and save the positional index for "title"
    title_position_index = build_positional_inverted_index(docs, field_name="title")
    title_output_path = os.path.join(base_output_folder, "title_position_index.json")
    save_dict_to_json(title_position_index, title_output_path)
    print(f"Created '{title_output_path}'")
    
    # 4) Build and save the positional index for "description"
    description_position_index = build_positional_inverted_index(docs, field_name="description")
    description_output_path = os.path.join(base_output_folder, "description_position_index.json")
    save_dict_to_json(description_position_index, description_output_path)
    print(f"Created '{description_output_path}'")
    
    # 5) Build and save the reviews index
    reviews_index = build_reviews_index(docs)
    reviews_output_path = os.path.join(base_output_folder, "reviews_index.json")
    save_dict_to_json(reviews_index, reviews_output_path)
    print(f"Created '{reviews_output_path}'")
    
    # 6) Build the features index (all features discovered automatically)
    all_features_index = build_feature_indexes(docs)
    
    # 7) Save each feature in a separate JSON file
    #    e.g., output/features/feature_brand.json
    for feature_name, feature_mapping in all_features_index.items():
        filename = f"feature_{feature_name}.json"
        feature_path = os.path.join(features_output_folder, filename)
        save_dict_to_json(feature_mapping, feature_path)
        print(f"Created '{feature_path}'")
