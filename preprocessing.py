import os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# init stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def load_corpus(folder_path):
    documents = []
    filenames = []

    # biar urutan doc1, doc2, ..., doc10
    files = sorted(os.listdir(folder_path), key=lambda x: int(''.join(filter(str.isdigit, x))))

    for filename in files:
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
                filenames.append(filename)

    return documents, filenames


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # hapus simbol
    tokens = text.split()

    # stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens