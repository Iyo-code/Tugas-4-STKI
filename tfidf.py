import math

# =========================
# TF (Term Frequency)
# =========================
def compute_tf(processed_corpus):
    tf_list = []

    for doc in processed_corpus:
        tf = {}
        total_terms = len(doc)

        for term in doc:
            tf[term] = tf.get(term, 0) + 1

        # normalisasi
        for term in tf:
            tf[term] = tf[term] / total_terms

        tf_list.append(tf)

    return tf_list


# =========================
# IDF (Inverse Document Frequency)
# =========================
def compute_idf(processed_corpus):
    N = len(processed_corpus)
    idf = {}

    # semua term unik
    all_terms = set(term for doc in processed_corpus for term in doc)

    for term in all_terms:
        # Document Frequency (DF)
        df = sum(1 for doc in processed_corpus if term in doc)

        # rumus IDF
        idf[term] = math.log10(N / df)

    return idf


# =========================
# TF-IDF
# =========================
def compute_tfidf(tf_list, idf):
    tfidf_list = []

    for tf in tf_list:
        tfidf = {}

        for term, val in tf.items():
            tfidf[term] = val * idf.get(term, 0)

        tfidf_list.append(tfidf)

    return tfidf_list


# =========================
# QUERY TF-IDF
# =========================
def compute_query_tfidf(query_tokens, idf):
    tf = {}

    # hitung TF query
    for term in query_tokens:
        tf[term] = tf.get(term, 0) + 1

    # normalisasi
    for term in tf:
        tf[term] = tf[term] / len(query_tokens)

    # hitung TF-IDF query
    tfidf = {}
    for term in tf:
        tfidf[term] = tf[term] * idf.get(term, 0)

    return tfidf


# =========================
# COSINE SIMILARITY
# =========================
def cosine_similarity(vec1, vec2):
    # dot product
    dot_product = 0
    for term in vec1:
        dot_product += vec1.get(term, 0) * vec2.get(term, 0)

    # norm vector 1
    norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))

    # norm vector 2
    norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)


# =========================
# SEARCH FUNCTION
# =========================
def search(query_tokens, tfidf_list, idf, filenames):
    query_vec = compute_query_tfidf(query_tokens, idf)

    scores = []

    for i, doc_vec in enumerate(tfidf_list):
        sim = cosine_similarity(query_vec, doc_vec)
        scores.append((filenames[i], sim))

    # urutkan dari terbesar
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores