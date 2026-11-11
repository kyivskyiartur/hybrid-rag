from minsearch import Index, VectorSearch

from embedder import Embedder

from sqlitesearch import TextSearchIndex, VectorSearchIndex



def build_text_index(chunks):
    text_index = TextSearchIndex(
        text_fields=["content"],
        keyword_fields=["filename"],
        db_path="chunks_text.db"
    )

    text_index.fit(chunks)

    return text_index


def build_vector_index(chunks, embedder):
    chunk_texts = [chunk["content"] for chunk in chunks]
    
    vectors = embedder.encode_batch(chunk_texts)

    vector_index = VectorSearchIndex(
        keyword_fields=["filename"],
        mode="ivf",
        db_path="chunks_vector.db"
    )

    vector_index.fit(vectors, chunks)

    return vector_index


def text_search (text_index, query, num_results=5):
    results = text_index.search(
        query = query,
        num_results = num_results
    )

    return results


def vector_search (vector_index, query, embedder, num_results=5):
    
    query_vector=embedder.encode(query)

    results = vector_index.search(
        query_vector,
        num_results = num_results
    )

    return results


'''
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
'''

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


def hybrid_search(text_index, vector_index, embedder, query, num_results=5):

    text_results = text_search(text_index, query, num_results=10)

    vector_results = vector_search(vector_index, query, embedder, num_results=10)

    results = rrf([vector_results, text_results], num_results=num_results)

    return results