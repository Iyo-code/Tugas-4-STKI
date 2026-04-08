import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# BUILD TF-IDF MATRIX
# =========================
def build_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return tfidf_matrix.toarray(), vectorizer


# =========================
# SVD (LSI)
# =========================
def apply_svd(matrix, k=2):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # reduce dimensi
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    return U_k, S_k, Vt_k


# =========================
# QUERY PROCESSING
# =========================
def lsi_search(query, vectorizer, U_k, S_k, Vt_k):
    query_vec = vectorizer.transform([query]).toarray()

    # project ke ruang LSI
    query_lsi = np.dot(np.dot(query_vec, Vt_k.T), np.linalg.inv(S_k))

    # representasi dokumen di ruang LSI
    docs_lsi = np.dot(U_k, S_k)

    # hitung similarity (dot product)
    scores = []

    for i, doc_vec in enumerate(docs_lsi):
        score = np.dot(query_lsi, doc_vec)
        scores.append((i, score[0]))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores