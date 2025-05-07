import os
import csv
import random
import spacy
from spacy.training import Example
from spacy.matcher import Matcher
from spacy.util import minibatch
import requests
from requests.exceptions import RequestException, HTTPError
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import json

CSV_PATH = '/Users/golubdima/Desktop/ML/ml_ner_model/data/urls.csv'
MODEL_DIR = 'model'
RESULTS_PATH = 'extracted_product_names.json'

# Список мебельных ключевых слов
KEYWORDS = [
    "chair", "armchair", "rocking chair", "stool", "bench",
    "table", "dining table", "coffee table", "side table", "desk",
    "sofa", "couch", "loveseat", "sectional", "futon", "daybed",
    "bed", "bunk bed", "canopy bed", "four-poster bed", "twin bed", "queen bed", "king bed",
    "pillow", "cushion", "blanket", "quilt", "duvet", "comforter",
    "dresser", "wardrobe", "closet", "cabinet",
    "shelf", "bookshelf", "bookcase", "nightstand",
    "recliner", "ottoman", "bean bag", "lounger", "chaise", "folding chair", "folding table",
    "lamp", "floor lamp", "desk lamp", "chandelier", "pendant light",
    "rug", "carpet", "mirror", "wall mirror", "vanity mirror",
    "vase", "wall art", "sculpture", "figurine", "clock",
    "tray", "basket", "decorative bowl", "planter", "curtains",
    "shoe rack", "fireplace tools", "room divider"
]

nlp_en = spacy.load("en_core_web_sm")
lemmatized_keywords = set([nlp_en(keyword)[0].lemma_ for keyword in KEYWORDS])

def annotate_text(text):
    """Аннотация с лемматизацией и шаблонами spaCy."""
    doc = nlp_en(text)
    matcher = Matcher(nlp_en.vocab)

    # Пример шаблонов: предмет мебели с прилагательным (например, "wooden chair")
    patterns = [
        [{"POS": "ADJ", "OP": "?"}, {"LEMMA": {"IN": list(lemmatized_keywords)}}],
        [{"LEMMA": {"IN": ["queen", "king", "twin"]}}, {"LEMMA": "bed"}]
    ]
    matcher.add("PRODUCT", patterns)
    matches = matcher(doc)

    entities = []
    spans = []
    for match_id, start, end in matches:
        span = doc[start:end]
        if not any(span.start < s.end and span.end > s.start for s in spans):  # без перекрытий
            entities.append((span.start_char, span.end_char, "PRODUCT"))
            spans.append(span)
    return entities

def load_urls_from_csv(file_path):
    urls = []
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        return urls
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                urls.append(row[0])
    return urls

def fetch_text_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all(['p', 'h1'])])
    except (HTTPError, RequestException) as e:
        print(f"Ошибка при загрузке {url}: {e}")
    return None

def generate_training_data(texts):
    training_data = []
    for text in texts:
        if text:
            entities = annotate_text(text)
            training_data.append((text, {"entities": entities}))
    return training_data

def split_data(urls, test_size=0.2, valid_size=0.1):
    train_urls, test_urls = train_test_split(urls, test_size=test_size, random_state=42)
    train_urls, valid_urls = train_test_split(train_urls, test_size=valid_size / (1 - test_size), random_state=42)
    return train_urls, valid_urls, test_urls

def train_model(training_data, model_dir, n_iter = 50):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)

    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in training_data]
    optimizer = nlp.initialize()

    for i in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size = 4)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)
        print(f"Итерация {i + 1}, потери: {losses}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    nlp.to_disk(model_dir)
    print("Модель сохранена!")

def load_model(model_dir):
    if os.path.exists(model_dir):
        return spacy.load(model_dir)
    else:
        print(f"Модель не найдена в {model_dir}")
        return None

def extract_product_names(text, nlp_model):
    doc = nlp_model(text)
    seen = set()
    unique_products = []
    for ent in doc.ents:
        if ent.label_ == "PRODUCT" and ent.text not in seen:
            unique_products.append(ent.text)
            seen.add(ent.text)
    return unique_products

def evaluate_model(nlp_model, texts):
    total = len(texts)
    correct = 0
    for text in texts:
        entities = annotate_text(text)
        annotated_names = [text[start:end] for (start, end, label) in entities if label == "PRODUCT"]
        extracted_names = extract_product_names(text, nlp_model)
        if set(extracted_names) == set(annotated_names):
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Точность модели: {accuracy:.2%}")

def main():
    urls = load_urls_from_csv(CSV_PATH)
    train_urls, valid_urls, test_urls = split_data(urls)

    train_texts = [fetch_text_from_url(url) for url in train_urls if fetch_text_from_url(url)]
    valid_texts = [fetch_text_from_url(url) for url in valid_urls if fetch_text_from_url(url)]

    training_data = generate_training_data(train_texts)
    train_model(training_data, MODEL_DIR)

    nlp_model = load_model(MODEL_DIR)
    if nlp_model:
        print("\nОценка на валидации:")
        evaluate_model(nlp_model, valid_texts)

        test_results = {}
        for url in test_urls:
            text = fetch_text_from_url(url)
            if text:
                product_names = extract_product_names(text, nlp_model)
                test_results[url] = product_names

        with open(RESULTS_PATH, 'w') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)
        print(f"Результаты сохранены в {RESULTS_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Остановлено пользователем")
