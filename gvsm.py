import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# TF-IDF MATRIX
# =========================
def build_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)

    return matrix.toarray(), vectorizer


# =========================
# TERM SIMILARITY MATRIX
# =========================
def term_similarity_matrix(tfidf_matrix):
    # transpose → term x doc
    term_matrix = tfidf_matrix.T

    # cosine similarity antar term
    sim_matrix = cosine_similarity(term_matrix)

    return sim_matrix


# =========================
# GVSM SEARCH
# =========================
def gvsm_search(query, vectorizer, tfidf_matrix, term_sim_matrix):
    query_vec = vectorizer.transform([query]).toarray()

    # transform dengan term similarity
    new_query = np.dot(query_vec, term_sim_matrix)

    scores = []

    for i, doc_vec in enumerate(tfidf_matrix):
        score = np.dot(new_query, doc_vec)
        scores.append((i, score[0]))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores