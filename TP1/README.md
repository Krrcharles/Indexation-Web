# Web Crawler en Python

## Description
Ce projet est un web crawler simple écrit en Python qui explore un site web à partir d'une URL donnée. Il extrait les titres des pages, les premiers paragraphes et les liens internes tout en respectant les règles définies dans le fichier `robots.txt` du site. Une priorité est accordée aux pages contenant le mot-clé `product`.

## Fonctionnalités
- Extraction automatique des titres, premiers paragraphes et liens internes.
- Respect des règles définies dans `robots.txt`.
- Priorisation des pages contenant le mot `product`.
- Limitation du nombre de pages visitées (par défaut : 50).
- Sauvegarde des résultats dans un fichier JSON.

## Utilisation
Exécutez le script avec les paramètres suivants :
```sh
python main.py <URL_de_départ> <nombre_max_de_pages>
```
Exemple :
```sh
python main.py https://web-scraping.dev/products 50
```
Si aucun argument n'est fourni, le script utilisera les valeurs par défaut :
- **URL de départ** : `https://web-scraping.dev/products`
- **Nombre maximum de pages** : `50`

## Résultats
Le crawler génère un fichier `results.json` contenant une liste de dictionnaires avec les informations extraites :
```json
[
    {
        "title": "Exemple de Titre",
        "url": "https://web-scraping.dev/example",
        "first_paragraph": "Ceci est le premier paragraphe extrait.",
        "links": ["https://web-scraping.dev/page1", "https://web-scraping.dev/page2"]
    }
]
```

