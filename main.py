from preprocessing import load_corpus, preprocess_text
from fuzzy import fuzzy_search
from gvsm import build_tfidf, term_similarity_matrix, gvsm_search
from lsi import build_tfidf_matrix, apply_svd, lsi_search

# =========================
# LOAD CORPUS
# =========================
corpus, filenames = load_corpus("corpus")
processed = [preprocess_text(doc) for doc in corpus]

print("=== DOKUMEN ===")
for i, doc in enumerate(corpus):
    print(f"{filenames[i]}: {doc[:50]}...")

# =========================
# PILIH METODE
# =========================
print("\nPilih Metode:")
print("1. Fuzzy")
print("2. GVSM")
print("3. LSI")

choice = input("Masukkan pilihan (1/2/3): ")

# =========================
# INPUT QUERY
# =========================
query = input("\nMasukkan query: ")

# =========================
# 1. FUZZY
# =========================
if choice == "1":
    print("\n===== FUZZY RETRIEVAL =====")
    fuzzy_search(query, processed, filenames)

# =========================
# 2. GVSM
# =========================
elif choice == "2":
    print("\n===== GVSM =====")

    # STEP 1: TF-IDF
    tfidf_matrix, vectorizer = build_tfidf(corpus)
    print("\nSTEP 1: TF-IDF MATRIX")
    print(tfidf_matrix)

    # STEP 2: TERM SIMILARITY
    term_sim = term_similarity_matrix(tfidf_matrix)
    print("\nSTEP 2: TERM SIMILARITY MATRIX")
    print(term_sim)

    # STEP 3: RANKING
    results = gvsm_search(query, vectorizer, tfidf_matrix, term_sim)

    print("\nSTEP 3: RANKING")
    for idx, score in results:
        print(f"{filenames[idx]} -> {score:.4f}")

# =========================
# 3. LSI
# =========================
elif choice == "3":
    print("\n===== LSI =====")

    # STEP 1: TF-IDF
    tfidf_matrix, vectorizer = build_tfidf_matrix(corpus)
    print("\nSTEP 1: TF-IDF MATRIX")
    print(tfidf_matrix)

    # STEP 2: SVD
    U, S, Vt = apply_svd(tfidf_matrix)

    print("\nSTEP 2: MATRIX U")
    print(U)

    print("\nSTEP 3: MATRIX S")
    print(S)

    print("\nSTEP 4: MATRIX Vt")
    print(Vt)

    # STEP 5: RANKING
    results = lsi_search(query, vectorizer, U, S, Vt)

    print("\nSTEP 5: RANKING")
    for idx, score in results:
        print(f"{filenames[idx]} -> {score:.4f}")

# =========================
# ERROR HANDLING
# =========================
else:
    print("Pilihan tidak valid.")