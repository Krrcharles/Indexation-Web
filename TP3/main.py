import json
import math
import string
import os

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#############################
# 1) CHARGEMENT DES FICHIERS
#############################

def load_json(file_path):
    """Charge un fichier JSON et retourne le dictionnaire correspondant."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(file_path):
    """Charge un fichier JSONL (une ligne = un document JSON) et retourne une liste de dict."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_all_files():
    """
    Charge tous les fichiers JSON / JSONL dans le répertoire courant
    et retourne un dictionnaire + listes de données.
    """
    
    # On suppose que tous les fichiers sont dans le même dossier que ce script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    brand_index_path       = os.path.join(current_dir, "brand_index.json")
    description_index_path = os.path.join(current_dir, "description_index.json")
    domain_index_path      = os.path.join(current_dir, "domain_index.json")
    origin_index_path      = os.path.join(current_dir, "origin_index.json")
    origin_synonyms_path   = os.path.join(current_dir, "origin_synonyms.json")
    reviews_index_path     = os.path.join(current_dir, "reviews_index.json")
    title_index_path       = os.path.join(current_dir, "title_index.json")
    products_jsonl_path    = os.path.join(current_dir, "products.jsonl")
    rearranged_jsonl_path  = os.path.join(current_dir, "rearranged_products.jsonl")
    
    # Chargement
    brand_index       = load_json(brand_index_path)
    description_index = load_json(description_index_path)
    domain_index      = load_json(domain_index_path)
    origin_index      = load_json(origin_index_path)
    origin_synonyms   = load_json(origin_synonyms_path)
    reviews_index     = load_json(reviews_index_path)
    title_index       = load_json(title_index_path)
    
    products_data          = load_jsonl(products_jsonl_path)
    rearranged_products    = load_jsonl(rearranged_jsonl_path)
    
    # Regrouper nos index inversés dans un seul dict
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
    Tokenize le texte (en anglais) :
      - Mise en minuscules
      - Découpage via word_tokenize de NLTK
      - Retrait de la ponctuation
      - Retrait des stopwords
    Retourne une liste de tokens.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    
    eng_stopwords = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    
    cleaned_tokens = [
        token for token in tokens
        if token not in eng_stopwords
        and token not in punctuation_set
    ]
    return cleaned_tokens

def expand_query_tokens(tokens, synonyms_dict):
    """
    Pour chaque token, si le token est dans synonyms_dict,
    on ajoute les synonymes comme "single token".
    
    Ex:
      synonyms_dict["usa"] = ["united states", "united states of america", "america"]
      => On ajoute : ["united_states", "united_states of america", "america"]
    """
    expanded_tokens = list(tokens)  # on part de la liste d'origine
    
    for token in tokens:
        if token in synonyms_dict:
            # Pour chaque phrase synonyme
            for phrase in synonyms_dict[token]:
                single_token = phrase.lower()
                expanded_tokens.append(single_token)
                
    return expanded_tokens

#############################
# 3) FONCTIONS DE FILTRAGE
#############################

def get_documents_for_token(token, indexes):
    """
    Retourne l'ensemble des documents (URLs) qui contiennent le token
    dans l'un des champs indexés. 
    
    Pour certains champs (title, description), la structure de l'index est :
       field_index[token] = { "url1": [positions], "url2": [positions], ... }
    Pour d'autres (brand, origin, domain), la structure de l'index est :
       field_index[token] = ["url1", "url2", ...]
    """
    docs_set = set()
    
    for field_name, field_index in indexes.items():
        if token in field_index:
            value = field_index[token]
            # Cas 1: On a un dict => { doc_url: [positions] }
            if isinstance(value, dict):
                docs_set.update(value.keys())   # On récupère .keys() (les URLs)
            
            # Cas 2: On a une liste => [ "url1", "url2", ... ]
            elif isinstance(value, list):
                docs_set.update(value)          # On ajoute directement les URLs
                
            # Au cas où il y aurait d'autres types (peu probable), on peut gérer/exclure
            else:
                print(f"[WARN] format inattendu pour '{field_name}' / token '{token}': {type(value)}")
    
    return docs_set

def filter_documents_any(tokens, indexes):
    """
    Retourne les documents qui contiennent AU MOINS UN token (logique OR).
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
    Retourne les documents qui contiennent TOUS les tokens (logique AND).
    """
    if not tokens:
        return set()
    # On initialise avec le set des documents du premier token
    result_docs = get_documents_for_token(tokens[0], indexes)
    for t in tokens[1:]:
        docs_for_t = get_documents_for_token(t, indexes)
        result_docs = result_docs.intersection(docs_for_t)
        if not result_docs:
            break
    return result_docs

def filter_documents(query, indexes, synonyms_dict, mode="ANY"):
    """
    1) Tokenize la requête
    2) Expandit les tokens avec les synonymes
    3) Applique un filtrage (ANY / ALL) et retourne l'ensemble des URLs
    """
    # 1) Tokenize
    base_tokens = tokenize(query)
    # 2) Expand synonyms
    tokens_expanded = expand_query_tokens(base_tokens, synonyms_dict)
    # 3) Filtrage
    if mode.upper() == "ANY":
        return filter_documents_any(tokens_expanded, indexes)
    else:
        return filter_documents_all(tokens_expanded, indexes)

#############################
# 4) BM25 & SCORING
#############################

def build_doc_lengths(products_data):
    """
    Construit un dict doc_lengths qui mappe:
      doc_lengths[url] = nombre de tokens (titre + description)
    en s'appuyant sur les données brutes de products_data.
    """
    doc_lengths = {}
    for product in products_data:
        url = product["url"]
        title = product.get("title", "")
        description = product.get("description", "")
        
        combined_text = title + " " + description
        tokens = tokenize(combined_text)  # On réutilise la fonction tokenize
        doc_lengths[url] = len(tokens)
    
    return doc_lengths


def get_term_frequency(token, doc_url, indexes):
    """
    Calcule la "fréquence" pondérée du token dans le document doc_url,
    en tenant compte des différentes structures d'index.
    
    indexes : dict de la forme :
      {
        "title":       dict(token -> dict(doc_url -> [positions])),
        "description": dict(token -> dict(doc_url -> [positions])),
        "brand":       dict(token -> list(doc_url)) OU dict(token -> dict(doc_url -> [positions])),
        "origin":      dict(token -> list(doc_url)),
        "domain":      dict(token -> list(doc_url))
      }
    
    Retourne un float : la somme des fréquences dans chaque champ * le poids du champ.
    """
    
    # Définir des poids pour chaque champ
    field_weights = {
        "title": 2.0,
        "description": 1.0,
        "brand": 1.0,
        "origin": 1.0,
        "domain": 1.0
    }
    
    freq_total = 0.0
    
    # On parcourt chaque champ
    for field, field_index in indexes.items():
        # Si le token n'existe pas dans cet index, on saute
        if token not in field_index:
            continue
        
        value = field_index[token]  # Peut être dict ou list
        
        if isinstance(value, dict):
            # Cas : dict(doc_url -> [positions])
            if doc_url in value:
                positions = value[doc_url]  # ex: [3, 17]
                freq = len(positions)
            else:
                freq = 0
        elif isinstance(value, list):
            # Cas : liste de doc_url
            # => si doc_url est dedans, on considère freq = 1, sinon 0
            if doc_url in value:
                freq = 1
            else:
                freq = 0
        else:
            # Format inattendu
            freq = 0
        
        # On applique le poids du champ (par défaut = 1.0, plus fort pour title)
        weight = field_weights.get(field, 1.0)
        freq_total += freq * weight
    
    return freq_total



def compute_idf(token, indexes, N):
    """
    Calcule l'IDF de type BM25:
    IDF(t) = log( (N - n_t + 0.5) / (n_t + 0.5 ) ), base 2.
    
    indexes est un dict comme :
      {
         "title": title_index,       # => dict(token -> dict(URL->positions))
         "description": description_index,
         "brand": brand_index,       # => dict(token -> list(URL))
         "origin": origin_index,     # => dict(token -> list(URL))
         "domain": domain_index      # => dict(token -> list(URL))
      }
    N = nombre total de documents.
    """
    doc_set = set()  # contiendra tous les URL qui contiennent le token
    
    # Parcourir chaque champ indexé
    for field_name, field_index in indexes.items():
        if token in field_index:
            value = field_index[token]
            
            # Cas 1 : dict (title, description)
            if isinstance(value, dict):
                doc_set.update(value.keys())  # on ajoute les URLs (les clés)
            
            # Cas 2 : liste (brand, origin, domain)
            elif isinstance(value, list):
                doc_set.update(value)         # on ajoute directement les URLs
                
            # Autre format inattendu (juste en debug)
            else:
                print(f"[WARN] Format inattendu dans compute_idf pour '{field_name}' / '{token}': {type(value)}")
    
    # Nombre de docs qui contiennent le token
    n_t = len(doc_set)
    
    if n_t == 0:
        return 0.0
    
    # Formule BM25 IDF
    return math.log((N - n_t + 0.5) / (n_t + 0.5), 2)


def bm25_score(doc_url, tokens, indexes, doc_lengths, avgdl, k=1.2, b=0.75):
    """
    Calcule le score BM25 pour un doc_url donné, par rapport à la liste de tokens.
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
# 5) AUTRES SIGNAUX
#############################

def get_review_bonus(doc_url, reviews_index, max_bonus=1.0):
    """
    Ajoute un bonus proportionnel à la note moyenne (0 à 5).
    mean_mark / 5 => [0..1], multiplié par max_bonus.
    """
    if doc_url not in reviews_index:
        return 0.0
    mean_mark = reviews_index[doc_url].get("mean_mark", 0)
    return (mean_mark / 5.0) * max_bonus

def is_exact_title_match(doc_url, query_tokens, products_info):
    """
    Vérifie si le titre du document (tokenisé) == l'ensemble des tokens de la requête (tokenisée).
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
    Score final = alpha * BM25 + beta * review_bonus + gamma * exact_match_indicator
    
    - alpha, beta, gamma : poids configurables
    """
    bm25_val = bm25_score(doc_url, query_tokens, indexes, doc_lengths, avgdl)
    review_b = get_review_bonus(doc_url, reviews_index, max_bonus=1.0)
    exact_m  = 1.0 if is_exact_title_match(doc_url, query_tokens, products_info) else 0.0
    
    final = alpha * bm25_val + beta * review_b + gamma * exact_m
    return final

#############################
# 6) FONCTION DE RANKING
#############################

def rank_documents(query, indexes, reviews_index, products_data_dict,
                   doc_lengths, avgdl, synonyms_dict, mode="ANY"):
    """
    1) Filtre les documents via (ANY ou ALL).
    2) Calcule un score final (BM25 + bonus).
    3) Trie par score décroissant.
    """
    # On filtre en se basant sur la requête + expansion
    filtered_docs = filter_documents(query, indexes, synonyms_dict, mode=mode)
    
    # Pour le calcul d'exact match, on utilise la tokenization basique
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
            alpha=1.0,     # poids BM25
            beta=0.3,      # poids review
            gamma=1.0      # poids exact match
        )
        results.append((doc_url, score))
    
    # Tri par ordre décroissant de score
    results.sort(key=lambda x: x[1], reverse=True)
    return results

#############################
# 7) GENERATION RESULTAT JSON
#############################

def produce_search_results_json(query, indexes, reviews_index,
                                products_list, synonyms_dict,
                                mode="ANY"):
    """
    1) Construit doc_lengths et avgdl
    2) Prépare un dict {url: {title:..., description:...}} pour accès rapide
    3) Fait le ranking
    4) Construit le JSON de sortie
    """
    
    # 1) doc_lengths & avgdl
    doc_lengths = build_doc_lengths(products_data)
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
    
    # 2) Dictionnaire rapide pour (title, description, etc.)
    products_data_dict = {}
    for p in products_list:
        products_data_dict[p["url"]] = {
            "title": p.get("title", ""),
            "description": p.get("description", "")
        }
    
    # 3) Ranker
    ranked_results = rank_documents(query, indexes, reviews_index,
                                    products_data_dict, doc_lengths, avgdl,
                                    synonyms_dict, mode)
    
    # 4) Construire la structure JSON finale
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
        # Récupération du titre/description
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
# 8) MAIN DE TEST
#############################

if __name__ == "__main__":
    # Télécharge les ressources NLTK si non déjà fait (décommenter au besoin)
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    
    # Charger tous les fichiers
    data = load_all_files()
    indexes         = data["indexes"]
    reviews_index   = data["reviews_index"]
    origin_synonyms = data["origin_synonyms"]
    products_data   = data["products_data"] 
    
    # Exemple de requête
    user_query = "chocolate from usa"
    # Mode de filtrage "ANY" ou "ALL"
    filter_mode = "ANY"
    
    # Générer le résultat final en JSON
    results_json_dict = produce_search_results_json(
        query=user_query,
        indexes=indexes,
        reviews_index=reviews_index,
        products_list=products_data,
        synonyms_dict=origin_synonyms,
        mode=filter_mode
    )
    
    # Affichage "pretty"
    print(json.dumps(results_json_dict, indent=2, ensure_ascii=False))
