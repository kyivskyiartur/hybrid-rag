from minsearch import Index, VectorSearch

from embedder import Embedder


def build_text_index(chunks):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )

    index.fit(chunks)

    return index


def build_vector_index(chunks, embedder):
    chunk_texts = [chunk["content"] for chunk in chunks]

    vectors = embedder.encode_batch(chunk_texts)

    index = VectorSearch(
        keyword_fields=["filename"]
    )

    index.fit(vectors, chunks)

    return index


def text_search(index, query, num_results=10):
    results = index.search(
        query=query,
        num_results=num_results
    )

    return results


def vector_search(index, embedder, query, num_results=10):
    query_vector = embedder.encode(query)

    results = index.search(
        query_vector,
        num_results=num_results
    )

    return results


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)

    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(
    text_index,
    vector_index,
    embedder,
    query,
    num_results=5
):
    text_results = text_search(
        text_index,
        query,
        num_results=10
    )

    vector_results = vector_search(
        vector_index,
        embedder,
        query,
        num_results=10
    )

    results = rrf(
        [vector_results, text_results],
        num_results=num_results
    )

    return results