import time
import json
import urllib.request
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from collections import deque
import sys


def fetch_url_content(url, delay=1):
    """
    Récupère le contenu HTML d'une URL via urllib.
    Retourne le HTML sous forme d'octets ou None en cas d'erreur.
    """
    time.sleep(delay)
    try:
        with urllib.request.urlopen(url) as response:
            return response.read()
    except HTTPError as e:
        print(f"[HTTPError] {url} : {e.code} - {e.reason}")
    except URLError as e:
        print(f"[URLError] {url} : {e.reason}")
    return None


def get_base_url(url):
    """
    Retourne la base de l'URL (protocole + nom de domaine).
    """
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


def can_crawl(url, user_agent="MyCrawler"):
    """
    Vérifie si l'URL est autorisée à être crawlée via le fichier robots.txt.
    Retourne True si autorisée, False sinon.
    """
    base_url = get_base_url(url)
    robots_url = urljoin(base_url, "robots.txt")

    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
    except:
        # En cas de robots.txt introuvable/inaccessible,
        # on autorise par défaut le crawl.
        return True

    return rp.can_fetch(user_agent, url)


def parse_html(html_content, page_url):
    """
    Extrait le titre, le premier paragraphe et les liens d'une page HTML.
    Retourne un dictionnaire avec ces informations.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Récupérer le titre
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "No title"

    # 2. Récupérer le premier paragraphe
    first_p_tag = soup.find("p")
    first_paragraph = first_p_tag.get_text(strip=True) if first_p_tag else ""

    # 3. Récupérer les liens
    links = [urljoin(page_url, a_tag["href"]) for a_tag in soup.find_all("a", href=True)]

    return {
        "title": title,
        "url": page_url,
        "first_paragraph": first_paragraph,
        "links": links,
    }


def crawl_site(start_url, max_pages=50):
    """
    Explore le site à partir d'une URL de départ, en priorisant les liens
    qui contiennent 'product'. S'arrête après max_pages pages.
    Retourne la liste des résultats (un dict par page visitée).
    """
    visited = set()               # Pour éviter de revisiter les mêmes URLs
    product_queue = deque()       # File d'attente contenant 'product'
    normal_queue = deque()        # File d'attente normale

    # Ajoute l'URL de départ dans la file appropriée
    if "product" in start_url:
        product_queue.append(start_url)
    else:
        normal_queue.append(start_url)

    results = []         # Contiendra toutes les données de pages visitées
    pages_visited = 0    # Compteur de pages visitées

    while pages_visited < max_pages and (product_queue or normal_queue):
        # Choisir la queue la plus prioritaire non vide
        if product_queue:
            current_url = product_queue.popleft()
        else:
            current_url = normal_queue.popleft()

        # Vérifier si on l'a déjà visitée
        if current_url in visited:
            continue

        # Vérifier l'autorisation robots.txt
        if not can_crawl(current_url):
            print(f"[robots] Skipping (not allowed): {current_url}")
            visited.add(current_url)
            continue

        # Télécharger le contenu HTML
        html = fetch_url_content(current_url, delay=1)
        if html is None:
            visited.add(current_url)
            continue

        # Parser le contenu
        page_data = parse_html(html, current_url)
        results.append(page_data)
        visited.add(current_url)
        pages_visited += 1

        print(f"[INFO] Visited: {current_url} (Total: {pages_visited})")

        # Ajouter les nouveaux liens trouvés
        for link_url in page_data["links"]:  # Correction ici
            if link_url not in visited:
                if "product" in link_url:
                    product_queue.append(link_url)
                else:
                    normal_queue.append(link_url)

    print(f"Fin du crawling. {pages_visited} pages visitées.")
    return results


def save_to_json(data, filename="results.json"):
    """
    Sauvegarde 'data' dans un fichier JSON.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """
    Point d'entrée du script:
      - Lit les arguments de ligne de commande :
        1) L'URL de départ
        2) Le nombre max de pages
      - Lance le crawl
      - Sauvegarde les résultats dans un fichier JSON
      - Si aucun argument n'est fourni, utilise les valeurs par défaut :
          URL de départ = "https://web-scraping.dev/products"
          max_pages = 50
    """
    if len(sys.argv) == 1:
        # Aucun argument => valeurs par défaut
        print("Aucun argument fourni. Utilisation des paramètres par défaut :")
        start_url = "https://web-scraping.dev/products"
        max_pages = 50
    else:
        # start_url = 1er argument, sinon défaut
        start_url = sys.argv[1] if len(sys.argv) > 1 else "https://web-scraping.dev/products"
        # max_pages = 2e argument, sinon défaut
        max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # Lancement du crawl
    data_crawled = crawl_site(start_url, max_pages)

    # Sauvegarde en JSON
    save_to_json(data_crawled, "results.json")
    print("Les résultats ont été enregistrés dans 'results.json'.")


if __name__ == "__main__":
    main()
