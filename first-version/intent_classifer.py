import joblib
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')
def route_query(user_query):
    file_path = './intent_router.pkl'
    classifier = joblib.load(file_path)
    query_embedding = encoder.encode([user_query])
    prediction = classifier.predict(query_embedding)[0]
    prob = classifier.predict_proba(query_embedding)[0]
    if max(prob) < 0.7:
        return "Unsure"

    return "Retrieval" if prediction == 1 else "Chat"