import os
from flask import Flask, render_template, request
from src.model import train_model, load_model, extract_product_names
from data_loader import fetch_text_from_url

app = Flask(__name__)

# Путь для сохранения обученной модели
MODEL_DIR = "model_dir"
print("File 'index.html' exists:", os.path.exists('templates/index.html'))


@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        urls = request.form.get('urls').splitlines()
        if not os.path.exists(MODEL_DIR):
            train_model(MODEL_DIR)  # Обучаем модель и сохраняем
        nlp_model = load_model(MODEL_DIR)  # Загружаем модель

        for url in urls:
            text = fetch_text_from_url(url)
            if text:
                results[url] = extract_product_names(text, nlp_model)
            else:
                results[url] = ["Нет данных или текст недоступен"]

    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
