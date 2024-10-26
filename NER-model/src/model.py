# model.py
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from data_collection import load_urls


def fetch_text_from_url(url):
    """
    Извлекает текст со страницы по URL.

    :param url: URL страницы
    :return: текст с HTML страницы
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Например, получаем текст с тегов <p> и <h1> для простоты
        page_text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1'])])
        return page_text
    except Exception as e:
        print(f"Ошибка при извлечении данных с URL {url}: {e}")
        return ""


def train_model(train_data):
    """
    Используем предобученную NER модель с библиотеки transformers.

    :param train_data: данные для тренировки (тексты с пометками)
    :return: обученная модель NER
    """
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    return ner_pipeline


def extract_product_names(text, ner_pipeline):
    """
    Извлекает названия продуктов из текста с использованием обученной NER модели.

    :param text: исходный текст
    :param ner_pipeline: обученный пайплайн для NER
    :return: список найденных названий продуктов
    """
    ner_results = ner_pipeline(text)
    products = [item['word'] for item in ner_results if 'PRODUCT' in item['entity']]
    return products


def process_urls(urls, ner_pipeline):
    """
    Обрабатывает список URL, извлекая названия продуктов из каждого.

    :param urls: список URL
    :param ner_pipeline: обученная NER модель
    :return: словарь с URL и названиями продуктов
    """
    url_product_map = {}
    for url in urls:
        print(f"Обработка {url}...")
        extracted_text = fetch_text_from_url(url)
        if extracted_text:
            product_names = extract_product_names(extracted_text, ner_pipeline)
            url_product_map[url] = product_names
    return url_product_map


def main(csv_file):
    # Загружаем список URL из CSV файла
    urls = load_urls(csv_file)

    # Разделяем на обучающую и тестовую выборки
    train_urls, test_urls = train_test_split(urls, test_size=0.5, random_state=42)

    # Генерируем данные для тренировки (это эмуляция, так как данные NER разметки могут быть отсутствующими)
    train_data = [
        "Пример текста для тренировки NER"]  # Этот текст будет заглушкой, в реальной задаче нужны помеченные данные.

    # Обучаем модель NER
    ner_pipeline = train_model(train_data)

    # Обрабатываем тестовые URL для извлечения названий продуктов
    test_results = process_urls(test_urls, ner_pipeline)

    return test_results
