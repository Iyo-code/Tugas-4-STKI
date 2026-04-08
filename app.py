import streamlit as st
from preprocessing import load_corpus, preprocess_text
from fuzzy import similarity
from gvsm import build_tfidf, term_similarity_matrix, gvsm_search
from lsi import build_tfidf_matrix, apply_svd, lsi_search

# =========================
# LOAD DATA
# =========================
corpus, filenames = load_corpus("corpus")
processed = [preprocess_text(doc) for doc in corpus]

st.set_page_config(page_title="IR System", layout="wide")

# =========================
# HIGHLIGHT FUNCTION (FIX)
# =========================
def highlight_text(text, query_tokens):
    words = text.split()
    highlighted = []

    for w in words:
        clean_w = w.lower()
        is_match = False

        for q in query_tokens:
            sim = similarity(q, clean_w)
            if sim > 0.7:  # threshold highlight
                is_match = True
                break

        if is_match:
            highlighted.append(f"<mark>{w}</mark>")
        else:
            highlighted.append(w)

    return " ".join(highlighted)

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("📚 Menu")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["🏠 Home", "🔍 Fuzzy", "📊 GVSM", "🧠 LSI"]
)

# =========================
# HOME
# =========================
if menu == "🏠 Home":
    st.title("📚 Information Retrieval System")
    st.write("""
    Aplikasi ini menggunakan 3 metode:
    
    🔍 Fuzzy → pencarian toleran typo  
    📊 GVSM → mempertimbangkan relasi antar kata  
    🧠 LSI → menemukan makna tersembunyi  
    """)

# =========================
# FUZZY
# =========================
elif menu == "🔍 Fuzzy":

    st.title("🔍 Fuzzy Retrieval")

    query = st.text_input("Masukkan Query")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.6)

    # PENJELASAN (DITAMBAH)
    st.info("""
    Proses Fuzzy Retrieval:
    1. Query di-preprocessing (lowercase + stemming)
    2. Setiap kata query dibandingkan dengan semua kata di dokumen
    3. Menggunakan Levenshtein Distance untuk menghitung similarity
    4. Jika similarity > threshold → dianggap cocok
    5. Skor dijumlahkan → menentukan ranking dokumen
    """)

    if st.button("Cari Fuzzy"):

        tokens = preprocess_text(query)

        st.subheader("STEP 1: Query")
        st.write(query)

        st.subheader("STEP 2: Preprocessing")
        st.write(tokens)

        st.subheader("STEP 3: Perbandingan Kata (Similarity)")

        results = []

        for i, doc in enumerate(processed):
            score = 0

            with st.expander(f"Dokumen: {filenames[i]}"):

                for q in tokens:
                    for term in doc:
                        sim = similarity(q, term)

                        # TAMPILKAN SEMUA (sesuai request kamu)
                        st.write(f"{q} ↔ {term} = {sim:.2f}")

                        if sim > threshold:
                            score += sim

                st.write(f"**Total Score: {score:.2f}**")

            if score > 0:
                results.append((i, score))

        st.subheader("STEP 4: Ranking")

        results.sort(key=lambda x: x[1], reverse=True)

        for idx, score in results:
            st.success(f"{filenames[idx]} → {score:.4f}")
            st.markdown(
                highlight_text(corpus[idx], tokens),
                unsafe_allow_html=True
            )

# =========================
# GVSM
# =========================
elif menu == "📊 GVSM":

    st.title("📊 Generalized Vector Space Model")

    query = st.text_input("Masukkan Query")

    if st.button("Cari GVSM"):

        tfidf_matrix, vectorizer = build_tfidf(corpus)
        term_sim = term_similarity_matrix(tfidf_matrix)

        st.subheader("STEP 1: TF-IDF Matrix")
        st.write(tfidf_matrix)

        st.subheader("STEP 2: Term Similarity Matrix")
        st.write(term_sim)

        st.subheader("STEP 3: Ranking")

        results = gvsm_search(query, vectorizer, tfidf_matrix, term_sim)
        tokens = preprocess_text(query)

        for idx, score in results:
            st.success(f"{filenames[idx]} → {score:.4f}")
            st.markdown(
                highlight_text(corpus[idx], tokens),
                unsafe_allow_html=True
            )

# =========================
# LSI
# =========================
elif menu == "🧠 LSI":

    st.title("🧠 Latent Semantic Indexing")

    query = st.text_input("Masukkan Query")

    if st.button("Cari LSI"):

        tfidf_matrix, vectorizer = build_tfidf_matrix(corpus)
        U, S, Vt = apply_svd(tfidf_matrix)

        st.subheader("STEP 1: TF-IDF Matrix")
        st.write(tfidf_matrix)

        st.subheader("STEP 2: SVD")
        st.write("U:", U)
        st.write("S:", S)
        st.write("Vt:", Vt)

        st.subheader("STEP 3: Ranking")

        results = lsi_search(query, vectorizer, U, S, Vt)
        tokens = preprocess_text(query)

        for idx, score in results:
            st.success(f"{filenames[idx]} → {score:.4f}")
            st.markdown(
                highlight_text(corpus[idx], tokens),
                unsafe_allow_html=True
            )