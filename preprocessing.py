import re
import string
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    except Exception:
        # Offline/firewalled fallback: continue without stopword filtering.
        stop_words = set()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ''.join(char for char in text if char in string.printable)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)
