# Web Indexation Project (ENSAI 2025)

## Project Overview
Each line in the `products.jsonl` file corresponds to one product document, containing:

- URL of the product or products page
- Title
- Description
- `product_features` (arbitrary key-value pairs such as brand, material, origin, etc.)
- `product_reviews` (list of reviews, each with a date, rating, and text)
- `links` (list of related URLs)

We generate the following types of indexes:

- **Positional Inverted Index for title**
- **Positional Inverted Index for description**
- **Reviews Index (non-inverted)**
- **Features Index** (one index per feature, automatically discovered)

Each index is saved in a JSON file for future retrieval and usage.

## Indexes Structure

### Title Positional Index

- **File**: `title_position_index.json`
- **Structure**: A nested dictionary where keys are tokens, and values are another dictionary mapping document URLs (or IDs) to a list of positions where the token appears.

Example:
```json
{
  "chocolate": {
    "https://web-scraping.dev/product/1": [0, 3],
    "https://web-scraping.dev/product/24": [2]
  },
  "box": {
    "https://web-scraping.dev/product/1": [1]
  }
}
```

### Description Positional Index

- **File**: `description_position_index.json`
- **Structure**: Identical to the title index format, but built from the description field.

Example:
```json
{
  "indulge": {
    "https://web-scraping.dev/product/1": [0]
  },
  "sweet": {
    "https://web-scraping.dev/product/1": [3]
  }
}
```

### Reviews Index

- **File**: `reviews_index.json`
- **Structure**: A dictionary keyed by the document’s URL. Each value contains review statistics:
  - `review_count`: total number of reviews
  - `average_rating`: float representing the mean of all ratings
  - `last_rating`: the rating of the most recent review (based on the date field)

Example:
```json
{
  "https://web-scraping.dev/product/1": {
    "review_count": 5,
    "average_rating": 4.6,
    "last_rating": 4
  },
  "https://web-scraping.dev/product/2": {
    "review_count": 2,
    "average_rating": 5.0,
    "last_rating": 5
  }
}
```

### Features Index

- **Output Folder**: `output/features/`
- **Files**: One JSON file per discovered feature, named `feature_<featureName>.json` (e.g., `feature_brand.json`).
- **Structure**: A dictionary keyed by the feature value, mapping to a list of URLs (or IDs) where that feature value occurs.

Example (`feature_brand.json`):
```json
{
  "ChocoDelight": [
    "https://web-scraping.dev/product/1",
    "https://web-scraping.dev/product/16"
  ],
  "CandyWorld": [
    "https://web-scraping.dev/product/8"
  ]
}
```

## How to Run the Code

### Requirements

- Python 3.7+
- `nltk` (Natural Language Toolkit)

Install dependencies:
```sh
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Steps to Execute

1. Ensure `products.jsonl` exists in the current directory.
2. Install dependencies:
   ```sh
   pip install nltk
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```
3. Run the script:
   ```sh
   cd TP2
   python main.py
   ```
4. Check the `output/` folder for results.

