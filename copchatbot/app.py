from flask import Flask, jsonify, request, render_template
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def load_dataset():
    dataset = []
    try:
        with open('ipc_sections.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset.append({
                    'description': row['Description'].strip(),
                    'offense': row['Offense'].strip(),
                    'punishment': row['Punishment'].strip(),
                    'section': row['Section'].strip()
                })
    except Exception as e:
        print(f"Error loading CSV dataset: {e}")
    return dataset

# Load dataset once when the server starts
dataset = load_dataset()
descriptions = [item['description'] for item in dataset]

# Initialize TF-IDF Vectorizer and fit on the dataset descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(descriptions)

def preprocess_query(query):
    """Map synonyms to keywords in the dataset."""
    synonyms = {
        'cheat': 'fraud',
        'cheating': 'fraud',
        'false product': 'fraud',
        'deception': 'fraud',
        'deceive': 'fraud'
    }
    for key, value in synonyms.items():
        if key in query:
            query = query.replace(key, value)
    return query

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['GET'])
def query():
    user_query = request.args.get('query', '').lower()
    if not user_query:
        return jsonify({'error': 'Empty query'}), 400

    # Preprocess the query to map synonyms to dataset keywords
    user_query = preprocess_query(user_query)

    # Transform the user query using the same vectorizer
    query_vector = vectorizer.transform([user_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the best match from the dataset
    best_match_index = cosine_similarities.argmax()
    best_score = cosine_similarities[best_match_index]

    # Set a threshold so that if similarity is too low, return an error.
    threshold = 0.2
    if best_score < threshold:
        return jsonify({'error': 'Sorry, I could not understand your query. Please try rephrasing it.'}), 404

    best_match = dataset[best_match_index]
    return jsonify(best_match)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
