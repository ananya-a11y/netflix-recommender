import os
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

# Global variables
data = None
vectorizer = None
suggestions_list = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model/vectorizer
model_path = os.path.join(BASE_DIR, 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

def load_data(reduce_data=False, num_samples=3000):
    global data, suggestions_list
    try:
        print("\U0001F4C2 Trying to read CSV file...")
        datapath = os.path.join(BASE_DIR, 'main_data.csv')
        df = pd.read_csv(datapath)
        if reduce_data:
            df = df.sample(n=min(num_samples, len(df)), random_state=42)
        data = df
        suggestions_list = df['movie_title'].tolist()
        print("\u2705 DataFrame loaded. Shape:", df.shape)
    except Exception as e:
        logging.error(f"Error loading data: {e}")


def load_model_and_vectorizer():
    global vectorizer
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("\u2705 Vectorizer loaded.")
    except Exception as e:
        logging.error(f"Error loading vectorizer: {e}")


def get_suggestions():
    global suggestions_list
    if suggestions_list is None:
        load_data()
    return suggestions_list if suggestions_list is not None else []


def rcmd(movie):
    global data, vectorizer

    if data is None:
        load_data()
    if vectorizer is None:
        load_model_and_vectorizer()
    if data is None or vectorizer is None:
        return "Error: Data or model is not available."

    try:
        movie_input = movie.strip().lower()
        print("🔍 Movie searched:", movie_input)

        movie_titles_lower = data['movie_title'].str.lower().str.strip()
        print("🎯 Sample titles in data:", movie_titles_lower.head(5).tolist())

        if movie_input not in movie_titles_lower.values:
            logging.warning(f"Movie '{movie_input}' not found in dataset.")
            return 'Sorry! Try another movie name.'

        movie_index = movie_titles_lower[movie_titles_lower == movie_input].index[0]
        print("🎯 Matched index:", movie_index)

        movie_comb = data['comb'].iloc[movie_index]
        movie_vector = vectorizer.transform([movie_comb])
        similarity_scores = cosine_similarity(movie_vector, vectorizer.transform(data['comb']))[0]
        similar_movies = sorted(
            list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True
        )[1:11]

        return [data['movie_title'].iloc[i[0]] for i in similar_movies]

    except Exception as e:
        logging.error(f"🔥 Error calculating similarity for '{movie}': {e}")
        return "Error calculating movie recommendations."
@app.route("/", methods=['GET', 'HEAD'])
@app.route("/home", methods=['GET', 'HEAD'])
def home():
    try:
        suggestions = get_suggestions()
        print("\u2705 Suggestions passed to template:", suggestions[:5])
        return render_template('home.html', suggestions=suggestions)
    except Exception as e:
        logging.error(f"Error rendering home.html: {e}")
        return f"Error rendering home.html: {e}", 500


@app.route("/similarity", methods=["POST"])
def get_similarity():
    movie = request.form.get('name')
    if not movie:
        return jsonify({"error": "No movie name provided."})
    try:
        recs = rcmd(movie)
        if isinstance(recs, str):
            return jsonify({"error": recs})
        return jsonify({"recommendations": recs})
    except Exception as e:
        logging.error(f"Error in /similarity: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        details = {key: request.form[key] for key in request.form}
        suggestions = get_suggestions()
        for k in ['rec_movies', 'rec_posters', 'cast_names', 'cast_chars', 'cast_profiles', 'cast_bdays', 'cast_bios', 'cast_places']:
            if k in details:
                details[k] = details[k].split('","')
                details[k][0] = details[k][0].replace('["', '')
                details[k][-1] = details[k][-1].replace('"]', '')
        cast_ids = details['cast_ids'].strip('[]').split(',')
        movie_cards = {details['rec_posters'][i]: details['rec_movies'][i] for i in range(len(details['rec_movies']))}
        casts = {details['cast_names'][i]: [cast_ids[i], details['cast_chars'][i], details['cast_profiles'][i]] for i in range(len(details['cast_names']))}
        cast_details = {details['cast_names'][i]: [cast_ids[i], details['cast_profiles'][i], details['cast_bdays'][i], details['cast_places'][i], details['cast_bios'][i]] for i in range(len(details['cast_names']))}
        movie_reviews = {}
        return render_template('recommend.html',
                               title=details['title'],
                               poster=details['poster'],
                               overview=details['overview'],
                               vote_average=details['rating'],
                               vote_count=details['vote_count'],
                               release_date=details['release_date'],
                               runtime=details['runtime'],
                               status=details['status'],
                               genres=details['genres'],
                               movie_cards=movie_cards,
                               reviews=movie_reviews,
                               casts=casts,
                               cast_details=cast_details,
                               suggestions=suggestions)
    except Exception as e:
        logging.error(f"Error in /recommend: {e}")
        return f"Error in /recommend: {e}", 500


@app.route('/api/movies')
def get_movies():
    try:
        data_path = os.path.join(BASE_DIR, 'data1.csv')
        df = pd.read_csv(data_path)
        movie_titles = df['movie_title'].tolist()
        return jsonify({"movies": movie_titles})
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return jsonify({"error": "Failed to load movie data"}), 500

# Preload for Render
load_data(reduce_data=True, num_samples=3000)
load_model_and_vectorizer()

# Expose for gunicorn
temp_app = app
application = temp_app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


