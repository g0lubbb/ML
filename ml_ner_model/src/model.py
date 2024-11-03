import os
import csv
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch
import requests
from requests.exceptions import RequestException, HTTPError
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import json

# Путь к CSV файлу и папке с моделью
CSV_PATH = '/Users/golubdima/PycharmProjects/ml_ner_model/data/urls.csv'
MODEL_DIR = 'model'
RESULTS_PATH = 'extracted_product_names.json'

def annotate_text(text):
    """Улучшенная разметка текста для извлечения объектов мебели с проверкой перекрывающихся сущностей."""
    entities = []
    keywords = [
        "chair", "chairs", "armchair", "rocking chair", "stool", "bench",
        "table", "tables", "dining table", "coffee table", "side table", "desk",
        "sofa", "sofas", "couch", "loveseat", "sectional", "futon", "daybed",
        "bed", "beds", "bunk bed", "canopy bed", "four-poster bed", "twin bed", "queen bed", "king bed",
        "pillow", "pillows", "cushion", "throw pillow",
        "blanket", "blankets", "throw blanket", "quilt", "duvet", "comforter",
        "dresser", "chest of drawers", "wardrobe", "closet", "armoire", "cabinet",
        "shelf", "shelves", "bookshelf", "bookcase", "display shelf",
        "nightstand", "sideboard", "buffet", "hutch", "console",
        "recliner", "ottoman", "bean bag", "lounger", "chaise", "folding chair", "folding table",
        "barstool", "high chair", "patio chair", "outdoor table",
        "storage bench", "trunk", "hope chest"
        # Освещение
                                  "lamp", "table lamp", "floor lamp", "desk lamp", "chandelier", "pendant light",
        "ceiling light", "wall sconce", "track lighting", "recessed lighting",
        "lantern", "outdoor light", "spotlight", "torchère",

        # Ковры
        "rug", "area rug", "carpet", "doormat", "runner", "throw rug",

        # Зеркала
        "mirror", "wall mirror", "floor mirror", "vanity mirror", "makeup mirror",

        # Декор
        "vase", "picture frame", "wall art", "sculpture", "figurine", "clock",
        "candlestick", "candle holder", "tray", "basket", "decorative bowl",
        "planter", "indoor plant", "artificial plant", "curtains", "drapes", "blinds",

        # Дополнительные аксессуары
        "coat rack", "umbrella stand", "shoe rack", "fireplace tools", "mantel",
        "room divider", "screen", "wall shelf", "floating shelf"
    ]

    for keyword in keywords:
        start = 0
        while start < len(text):
            start = text.lower().find(keyword, start)
            if start == -1:
                break
            end = start + len(keyword)
            # Проверка, перекрывается ли новая сущность с уже существующей
            if not any(s <= start < e or s < end <= e for s, e, _ in entities):
                entities.append((start, end, "PRODUCT"))
            start = end  # Продолжить с конца найденного объекта

    return entities


def load_urls_from_csv(file_path):
    """Загружает URL из CSV файла."""
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
    """Извлекает текст из HTML страницы по ссылке."""
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
    """Генерирует данные для обучения на основе текстов с автоматической разметкой."""
    training_data = []
    for text in texts:
        if text:
            entities = annotate_text(text)
            training_data.append((text, {"entities": entities}))
    return training_data

def split_data(urls, test_size=0.2, valid_size=0.1):
    """Разделяет данные на тренировочную, валидационную и тестовую выборки."""
    train_urls, test_urls = train_test_split(urls, test_size=test_size, random_state=42)
    train_urls, valid_urls = train_test_split(train_urls, test_size=valid_size / (1 - test_size), random_state=42)
    return train_urls, valid_urls, test_urls


def train_model(training_data, model_dir, n_iter=30):
    """Обучает модель NER для извлечения названий продуктов."""
    nlp = spacy.blank("en")  # Создание пустой английской модели
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)

    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])  # Добавляем метку "PRODUCT"

    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in training_data]
    optimizer = nlp.initialize()

    for i in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=2)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)
        print(f"Итерация {i + 1}, потери: {losses}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    nlp.to_disk(model_dir)
    print("Модель успешно сохранена!")


def load_model(model_dir):
    """Загружает модель NER, если она существует."""
    if os.path.exists(model_dir):
        return spacy.load(model_dir)
    else:
        print(f"Модель не найдена в {model_dir}. Сначала обучите модель.")
        return None


def extract_product_names(text, nlp_model):
    """Извлекает уникальные названия продуктов из текста без дублирования."""
    doc = nlp_model(text)
    seen = set()
    unique_products = []

    for ent in doc.ents:
        # Если метка "PRODUCT" и текст продукта ещё не был добавлен
        if ent.label_ == "PRODUCT" and ent.text not in seen:
            unique_products.append(ent.text)
            seen.add(ent.text)  # Добавляем текст продукта в множество для отслеживания

    return unique_products

def evaluate_model(nlp_model, texts):
    """Оценивает модель на валидационном наборе данных."""
    total = len(texts)
    correct = 0
    for text in texts:
        entities = annotate_text(text)
        # Сохраняем только текстовые представления извлеченных названий
        annotated_names = [text[start:end] for (start, end, label) in entities if label == "PRODUCT"]
        extracted_names = extract_product_names(text, nlp_model)

        # Преобразуем в множества, чтобы сравнить уникальные значения
        if set(extracted_names) == set(annotated_names):
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Точность модели на валидационном наборе данных: {accuracy:.2%}")



def main():
    # Загрузка предобученной модели для автоматической разметки
    nlp = spacy.load("en_core_web_sm")

    # Загрузка URL-адресов и деление на тренировочный, валидационный и тестовый наборы
    urls = load_urls_from_csv(CSV_PATH)
    train_urls, valid_urls, test_urls = split_data(urls)

    # Загрузка и обработка текста с тренировочных и валидационных URL-адресов
    train_texts = [fetch_text_from_url(url) for url in train_urls if fetch_text_from_url(url)]
    valid_texts = [fetch_text_from_url(url) for url in valid_urls if fetch_text_from_url(url)]

    # Генерация обучающих данных и обучение модели
    training_data = generate_training_data(train_texts)
    train_model(training_data, MODEL_DIR)

    # Оценка модели
    nlp_model = load_model(MODEL_DIR)
    if nlp_model:
        print("\nОценка модели на валидационных данных:")
        evaluate_model(nlp_model, valid_texts)

        # Извлечение данных из тестового набора и сохранение результатов
        test_results = {}
        for url in test_urls:
            text = fetch_text_from_url(url)
            if text:
                product_names = extract_product_names(text, nlp_model)
                test_results[url] = product_names

        # Сохранение результатов
        with open(RESULTS_PATH, 'w') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)
        print(f"Результаты сохранены в {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Процесс прерван пользователем")
