# app.py

from flask import Flask, render_template, request, jsonify
from model import load_model, fetch_text_from_url, extract_product_names

app = Flask(__name__)

# Загрузка обученной модели
nlp_model = load_model("model")


@app.route("/", methods=["GET", "POST"])
def index():
    extracted_products = None
    error_message = None

    if request.method == "POST":
        url = request.form.get("url")

        if not url:
            error_message = "Пожалуйста, введите URL."
        else:
            # Получаем текст из указанного URL
            text = fetch_text_from_url(url)
            if not text:
                error_message = "Не удалось загрузить текст по указанному URL."
            else:
                # Извлекаем названия продуктов с помощью модели
                extracted_products = extract_product_names(text, nlp_model)

    return render_template("index.html", products=extracted_products, error=error_message)


if __name__ == "__main__":
    app.run(debug=True)
