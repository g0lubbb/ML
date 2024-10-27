import requests
from bs4 import BeautifulSoup

def fetch_text_from_url(url):
    """Извлекает текст статьи с веб-страницы."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1'])])
        return page_text if page_text else None
    except requests.RequestException as e:
        print(f"Ошибка при извлечении данных с URL {url}: {e}")
        return None
