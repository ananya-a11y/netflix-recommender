# 🍿 Netflix Movie Recommender System

A content-based movie recommendation system trained on Netflix movie data using TF-IDF and cosine similarity — wrapped in a slick Flask web app.

---

## 🚀 Live Demo

👉 [Click here to try it out](https://netflix-recommender-pnp0.onrender.com)

---

## 🎯 Features

- 🔍 Search your favorite movie
- 🤖 Get 10 similar recommendations instantly
- 🧠 Content-based filtering using NLP
- 🌐 Clean & responsive UI with AJAX
- 💡 Real-time autocomplete using jQuery

---

## 🛠️ Tech Stack

| Component       | Tech Used                        |
|----------------|----------------------------------|
| Backend         | Python, Flask                    |
| ML/NLP          | Scikit-learn, TF-IDF, Cosine Similarity |
| Frontend        | HTML, CSS, JS, jQuery, Bootstrap |
| Deployment      | Render.com                       |

---

## 📁 File Structure

netflix-recommender/
├── app.py # Flask app
├── templates/
│ ├── home.html
│ └── recommend.html
├── static/
│ ├── style.css
│ ├── autocomplete.js
│ └── recommend.js
├── data1.csv # Cleaned Netflix dataset
├── vectorizer.pkl # Saved TF-IDF model
├── requirements.txt
└── README.md

---

## 🔧 Setup Instructions

1. Clone the repo  
   bash
   git clone https://github.com/ananya-a11y/netflix-recommender.git
   cd netflix-recommender
2. Install dependencies
   pip install -r requirements.txt
3. Run locally
   python app.py


🧠 How It Works
Combines movie title, cast, genre, and keywords into a single string.

Applies TF-IDF vectorization.

Computes similarity scores using cosine similarity.

Returns top 10 most similar movies.

👨‍💻 Author
Built by Ananya Ahlawat

🧿 License
MIT License – use it, remix it, just don’t forget to tag the creators 😉
