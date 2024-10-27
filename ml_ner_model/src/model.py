import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch
import random


def train_model(model_dir, n_iter=30):
    """Обучает кастомную модель NER для извлечения названий предметов."""
    nlp = spacy.blank("en")  # Создаем пустую модель

    # Добавляем компонент NER
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)

    # Метки, связанные с предметами
    labels = ["PRODUCT"]
    for label in labels:
        ner.add_label(label)

    # Генерация примеров для обучения
    training_data = [
        ("IPhone 12 is a new product by Apple", {"entities": [(0, 8, "PRODUCT")]}),
        ("Samsung Galaxy S21 is another device", {"entities": [(0, 16, "PRODUCT")]}),
    ]

    # Подготовка данных
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # Тренировка модели
    optimizer = nlp.initialize()
    for i in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=2)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)

    nlp.to_disk(model_dir)


def load_model(model_dir):
    """Загружает кастомную модель NER."""
    return spacy.load(model_dir)


def extract_product_names(text, nlp_model):
    """Извлекает названия предметов из текста."""
    doc = nlp_model(text)
    products = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]
    return products if products else ["Нет данных"]

