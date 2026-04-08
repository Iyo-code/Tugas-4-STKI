from preprocessing import preprocess_text

# =========================
# LEVENSHTEIN DISTANCE
# =========================
def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)

    if len(b) == 0:
        return len(a)

    previous_row = range(len(b) + 1)

    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


# =========================
# SIMILARITY
# =========================
def similarity(a, b):
    distance = levenshtein(a, b)
    max_len = max(len(a), len(b))
    return 1 - (distance / max_len)


# =========================
# FUZZY SEARCH (STEP BY STEP)
# =========================
def fuzzy_search(query, processed_corpus, filenames, threshold=0.6):
    print("\n=== STEP 1: QUERY ASLI ===")
    print(query)

    query_tokens = preprocess_text(query)

    print("\n=== STEP 2: HASIL PREPROCESSING QUERY ===")
    print(query_tokens)

    results = []

    print("\n=== STEP 3: PERHITUNGAN SIMILARITY ===")

    for i, doc in enumerate(processed_corpus):
        score = 0
        print(f"\nDokumen: {filenames[i]}")

        for q in query_tokens:
            for term in doc:
                sim = similarity(q, term)

                if sim > threshold:
                    print(f"{q} ~ {term} = {sim:.2f}")
                    score += sim

        print(f"Total Score: {score:.2f}")

        if score > 0:
            results.append((filenames[i], score))

    print("\n=== STEP 4: RANKING ===")

    results.sort(key=lambda x: x[1], reverse=True)

    for doc, score in results:
        print(f"{doc} -> {score:.4f}")

    return results