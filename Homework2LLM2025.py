from fastembed import TextEmbedding
import numpy as np
import requests
from qdrant_client import QdrantClient, models

# Q1: Embedding the query
def question1():
    embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-small-en")
    query = "I just discovered the course. Can I join now?"
    # Convert generator to list and then get first element
    query_embedding = list(embedding_model.embed([query]))[0]
    min_value = np.min(query_embedding)
    print(f"Q1: Minimal value in the array: {min_value}")
    # Also print the shape to verify it's 512-dimensional
    print(f"Shape of embedding: {query_embedding.shape}")
    return query_embedding, embedding_model, query

# Q2: Cosine similarity with another vector
def question2(embedding_model, query_embedding):
    doc = 'Can I still join the course after the start date?'
    doc_embedding = list(embedding_model.embed([doc]))[0]
    
    # Computing cosine similarity using dot product
    cosine_sim = query_embedding.dot(doc_embedding)
    print(f"Q2: Cosine similarity between query and document: {cosine_sim}")
    return cosine_sim

# Q3: Ranking by cosine (text only)
def question3(embedding_model, query_embedding):
    documents = [
        {'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
         'section': 'General course-related questions',
         'question': 'Course - Can I still join the course after the start date?',
         'course': 'data-engineering-zoomcamp'},
        {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
         'section': 'General course-related questions',
         'question': 'Course - Can I follow the course after it finishes?',
         'course': 'data-engineering-zoomcamp'},
        {'text': 'The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first \'Office Hours\' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon\'t forget to register in DataTalks.Club\'s Slack and join the channel.',
         'section': 'General course-related questions',
         'question': 'Course - When will the course start?',
         'course': 'data-engineering-zoomcamp'},
        {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
         'section': 'General course-related questions',
         'question': 'Course - What can I do before the course starts?',
         'course': 'data-engineering-zoomcamp'},
        {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
         'section': 'General course-related questions',
         'question': 'How can we contribute to the course?',
         'course': 'data-engineering-zoomcamp'}
    ]
    
    # Get embeddings for all documents' text fields
    text_embeddings = list(embedding_model.embed([doc['text'] for doc in documents]))
    text_embeddings = np.array(text_embeddings)
    similarities = text_embeddings.dot(query_embedding)
    
    max_idx = np.argmax(similarities)
    print(f"\nQ3: Document index with highest similarity (text only): {max_idx}")
    print(f"Q3: Highest similarity score: {similarities[max_idx]}")
    return max_idx, documents

# Q4: Ranking by cosine (question + text)
def question4(embedding_model, query_embedding, documents):
    # Concatenate question and text for each document
    full_texts = [f"{doc['question']} {doc['text']}" for doc in documents]
    
    # Get embeddings for concatenated texts
    full_embeddings = list(embedding_model.embed(full_texts))
    full_embeddings = np.array(full_embeddings)
    
    # Compute similarities
    similarities = full_embeddings.dot(query_embedding)
    
    # Find index of highest similarity
    max_idx = np.argmax(similarities)
    print(f"\nQ4: Document index with highest similarity (question + text): {max_idx}")
    print(f"Q4: Highest similarity score: {similarities[max_idx]}")
    
    # Print all similarities for verification
    for i, sim in enumerate(similarities):
        print(f"Document {i} similarity: {sim}")
    
    return max_idx

# Q5: Check dimensionality of BAAI/bge-small-en model
def question5():
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en")
    test_text = "Test text for dimensionality check"
    embedding = list(embedding_model.embed([test_text]))[0]
    print(f"\nQ5: Dimensionality of BAAI/bge-small-en model: {embedding.shape[0]}")
    return embedding_model

# Q6: Index documents in Qdrant and query
def question6():
    # Load documents
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    # Filter for ML Zoomcamp documents
    documents = []
    for course in documents_raw:
        course_name = course['course']
        if course_name != 'machine-learning-zoomcamp':
            continue
        
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)

    # Initialize the BAAI/bge-small-en model
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en")

    # Create collection
    collection_name = "ml_zoomcamp_docs"
    
    # Recreate collection
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,  # BAAI/bge-small-en dimensionality
            distance=models.Distance.COSINE
        )
    )

    # Prepare documents for insertion
    points = []
    for idx, doc in enumerate(documents):
        # Concatenate question and text
        full_text = f"{doc['question']} {doc['text']}"
        
        # Get embedding
        embedding = list(embedding_model.embed([full_text]))[0]
        
        # Create point
        points.append(models.PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload=doc
        ))

    # Insert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )

    # Query
    query = "I just discovered the course. Can I join now?"
    query_embedding = list(embedding_model.embed([query]))[0]
    
    # Search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=1
    )
    
    if search_result:
        print(f"\nQ6: Top result score: {search_result[0].score}")
    
    return search_result

# Run all questions
query_embedding, embedding_model, query = question1()
cosine_sim = question2(embedding_model, query_embedding)
max_idx_q3, documents = question3(embedding_model, query_embedding)
max_idx_q4 = question4(embedding_model, query_embedding, documents)

# Run Q5
small_model = question5()

# Run Q6
results = question6()