# 📚 Sistem Information Retrieval (IR)

Aplikasi ini merupakan implementasi **Information Retrieval (IR)** menggunakan beberapa metode pencarian modern dan klasik, yaitu:

* 🔍 **Fuzzy Retrieval**
* 📊 **Generalized Vector Space Model (GVSM)**
* 🧠 **Latent Semantic Indexing (LSI)**

Aplikasi dilengkapi dengan **visualisasi step-by-step**, **highlight keyword**, dan **antarmuka interaktif berbasis Streamlit**.

---

## 🚀 Teknologi yang Digunakan

Aplikasi dibangun menggunakan:

* **Python** → bahasa utama
* **Streamlit** → antarmuka web interaktif
* **NumPy** → perhitungan matriks (LSI)
* **Scikit-learn** → TF-IDF & cosine similarity
* **Sastrawi** → preprocessing Bahasa Indonesia (stemming)

---

## 🚀 Fitur Utama

### 🔍 Fuzzy Retrieval

* Toleran terhadap typo (contoh: *machin → machine*)
* Menggunakan **Levenshtein Distance**
* Menghasilkan skor kemiripan (0–1)
* Dilengkapi threshold untuk filtering hasil

---

### 📊 Generalized Vector Space Model (GVSM)

* Menggunakan **TF-IDF**
* Menghitung **kemiripan antar term**
* Mempertimbangkan hubungan antar kata
* Menghasilkan ranking dokumen

---

### 🧠 Latent Semantic Indexing (LSI)

* Menggunakan **Singular Value Decomposition (SVD)**
* Menangkap **makna tersembunyi (latent semantic)**
* Mengurangi dimensi data
* Lebih cerdas dalam memahami konteks

---

### 🎯 Fitur Tambahan

* 📄 Highlight keyword pada dokumen
* 📊 Tampilan **step-by-step proses**
* 📁 Menampilkan seluruh dokumen corpus
* 🧪 Input query interaktif

---

## 🧠 Alur Sistem

```
Teks → Preprocessing → Representasi (TF-IDF) → 
Metode (Fuzzy / GVSM / LSI) → Ranking → Hasil
```

---

## 🗂️ Struktur Project

```
project-ir/
│
├── corpus/               # Dataset dokumen (.txt)
│   ├── doc1.txt
│   ├── doc2.txt
│   └── ...
│
├── preprocessing.py      # Tokenisasi & stemming
├── fuzzy.py              # Fuzzy similarity (Levenshtein)
├── gvsm.py               # GVSM model
├── lsi.py                # LSI (SVD)
├── app.py                # Streamlit UI
├── main.py               # Testing (opsional)
│
└── README.md             # Dokumentasi
```

---

## ⚙️ Instalasi & Menjalankan Aplikasi

### 1. Clone Repository

```
git clone https://github.com/username/project-ir.git
cd project-ir
```

---

### 2. Install Dependencies

```
pip install streamlit numpy scikit-learn Sastrawi
```

---

### 3. Jalankan Aplikasi

```
python -m streamlit run app.py
```

Aplikasi akan terbuka di browser secara otomatis.

---

## 📊 Metode yang Digunakan

### 1. Fuzzy Similarity

Menggunakan pendekatan **Levenshtein Distance** untuk mengukur kemiripan kata:

* Nilai 0 → tidak mirip
* Nilai 1 → identik

---

### 2. TF-IDF

Digunakan pada GVSM dan LSI untuk merepresentasikan dokumen.

* **TF (Term Frequency)** → frekuensi kata
* **IDF (Inverse Document Frequency)** → pentingnya kata

---

### 3. Cosine Similarity

Digunakan untuk menghitung kemiripan antara query dan dokumen.

---

### 4. Singular Value Decomposition (SVD)

Digunakan pada LSI untuk:

* Reduksi dimensi
* Menemukan hubungan tersembunyi antar kata

---

## 🔍 Cara Kerja Sistem

1. User memasukkan query
2. Query dipreprocessing:

   * Lowercase
   * Tokenisasi
   * Stemming
3. Sistem memilih metode:

   * Fuzzy / GVSM / LSI
4. Menghitung kemiripan
5. Menampilkan ranking dokumen + highlight keyword

---

## 📌 Contoh Penggunaan

### Input:

```
machin learnin
```

### Output:

* Dokumen relevan ditampilkan
* Keyword di-highlight
* Skor similarity ditampilkan

---

## 🎯 Tujuan Proyek

Proyek ini dibuat untuk:

* Memahami konsep **Information Retrieval**
* Mengimplementasikan berbagai metode IR
* Membandingkan performa metode
* Membangun aplikasi interaktif berbasis web

---

## 👤 Author

Nama: **Trio Suro Wibowo**
NIM: **2405551168**
Mata Kuliah: **Sistem Temu Kembali Informasi (STKI)**

---

## 📄 Lisensi

Proyek ini digunakan untuk keperluan pembelajaran dan akademik.
