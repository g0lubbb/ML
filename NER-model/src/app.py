# app.py
from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    product_names = []
    if request.method == 'POST':
        url = request.form['url']
        results = model.main('/Users/golubdima/PycharmProjects/mlTask/data/urls.csv')
        if url in results:
            product_names = results[url]
    return render_template('index.html', product_names=product_names)

if __name__ == '__main__':
    app.run(debug=True)
