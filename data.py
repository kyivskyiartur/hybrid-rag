from gitsource import GithubRepositoryDataReader, chunk_documents


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path
    )

    files = reader.read()

    documents = []

    for file in files:
        document = file.parse()
        documents.append(document)

    return documents


def create_chunks(documents, size=2000, step=1000):
    chunks = chunk_documents(
        documents,
        size=size,
        step=step
    )

    return chunks