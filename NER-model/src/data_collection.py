# data_collection.py
import pandas as pd


def load_urls(csv_file_path):
    """
    Функция для загрузки URL из CSV файла.

    :param csv_file_path: путь до csv файла
    :return: список URL
    """
    try:
        df = pd.read_csv(csv_file_path)
        # Предположим, что CSV файл содержит столбец 'url'
        urls = df['url'].tolist()
        return urls
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return []
