from pathlib import Path
import json
import os
import pickle
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
    <style>
        :root {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --accent-color: #ec4899;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        h1 {
            color: #f1f5f9 !important;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        h2, h3 {
            color: #e2e8f0 !important;
        }
        
        .stSelectbox label, .stButton label, .stTextInput label {
            color: #cbd5e1 !important;
            font-weight: 500;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3) !important;
        }
        
        .stMetric {
            background: linear-gradient(135deg, #312e81, #1a1a4d) !important;
            border: 2px solid #6366f1 !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            box-shadow: 0 8px 16px rgba(99, 102, 241, 0.2) !important;
        }
        
        .stMetric-value {
            color: #fbbf24 !important;
            font-weight: 800 !important;
            font-size: 1.5rem !important;
        }
        
        .stCaption {
            color: #fca5a5 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            background: rgba(99, 102, 241, 0.1) !important;
            padding: 0.5rem 0.75rem !important;
            border-radius: 6px !important;
            border-left: 3px solid #ec4899 !important;
            margin-top: 0.5rem !important;
        }
        
        .stSelectbox {
            background-color: #1e293b !important;
            border-radius: 8px !important;
        }
        
        .stSelectbox>div>div {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }
        
        .stTextInput>div>div>input {
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }
        
        .stInfo, .stWarning, .stError {
            border-radius: 8px !important;
        }
        
        hr {
            margin: 2rem 0 !important;
            border: none !important;
            height: 1px;
            background: linear-gradient(90deg, transparent, #334155, transparent);
        }
        
        .stMarkdown {
            color: #e2e8f0 !important;
        }
        
        .stMarkdown strong {
            color: #fbbf24 !important;
            font-size: 1.05rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MOVIES_PKL = BASE_DIR / "movies.pkl"
SIMILARITY_PKL = BASE_DIR / "similarity.pkl"
KEY_FILE = BASE_DIR / "tmdb_key.txt"

st.title("🎬 Movie Recommender")
st.markdown("*Content-based recommendation engine using NLP & cosine similarity*")

api_key = os.getenv("TMDB_API_KEY", "").strip()
# fallback: read key from a local file if env var isn't set
if not api_key and KEY_FILE.exists():
	try:
		api_key = KEY_FILE.read_text(encoding="utf-8").strip()
	except Exception:
		api_key = ""


@st.cache_resource
def load_artifacts():
	with open(MOVIES_PKL, "rb") as handle:
		movies_df = pickle.load(handle)
	with open(SIMILARITY_PKL, "rb") as handle:
		similarity = pickle.load(handle)
	return movies_df, similarity


def recommend(movie_title, movies_df, similarity, top_n=5):
	idx = movies_df[movies_df["title"].str.lower() == movie_title.lower()].index
	if idx.empty:
		return []
	idx = idx[0]
	distances = similarity[idx]
	matches = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : top_n + 1]
	return [
		{"title": movies_df.iloc[i].title, "movie_id": int(movies_df.iloc[i].movie_id), "similarity": float(score)}
		for i, score in matches
	]


@st.cache_data(show_spinner=False)
def poster_url(movie_id, api_key_value):
	if not api_key_value:
		return None
	url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key_value}"
	try:
		with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=10) as response:
			data = json.load(response)
	except (HTTPError, URLError, TimeoutError):
		return None
	path = data.get("poster_path")
	return f"https://image.tmdb.org/t/p/w500{path}" if path else None


def get_shared_themes(movie1_tags, movie2_tags, top_n=3):
	if not movie1_tags or not movie2_tags:
		return []
	
	words1 = [w.lower() for w in str(movie1_tags).split() if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 2]
	words2_set = set(w.lower() for w in str(movie2_tags).split() if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 2)
	
	overlap = [w for w in words1 if w in words2_set]
	return list(dict.fromkeys(overlap))[:top_n]


try:
	movies_df, similarity = load_artifacts()
	selected_movie = st.selectbox("Select a movie", movies_df["title"].tolist())
	if st.button("Recommend"):
		# if no API key, prompt user to enter & save it (persistent file)
		if not api_key:
			st.info("TMDB API key not found. Enter it below to display posters.")
			input_key = st.text_input("Enter TMDB API Key", type="password")
			if st.button("Save TMDB key") and input_key:
				try:
					KEY_FILE.write_text(input_key.strip(), encoding="utf-8")
					api_key = input_key.strip()
					st.experimental_rerun()
				except Exception as e:
					st.error(f"Failed to save key: {e}")

		recs = recommend(selected_movie, movies_df, similarity)
		if not recs:
			st.warning("No recommendations found.")
		else:
			selected_tags = movies_df[movies_df["title"].str.lower() == selected_movie.lower()]["tags"].iloc[0]
			cols = st.columns(5)
			for col, rec in zip(cols, recs):
				with col:
					img = poster_url(rec["movie_id"], api_key)
					if img:
						st.image(img, use_container_width=True)
					
					st.markdown(f"**✨ {rec['title']}**", help=rec["title"])
					
					similarity_pct = f"{rec['similarity'] * 100:.2f}%"
					st.markdown(f"<h3 style='color: #fbbf24; text-align: center; margin: 0.5rem 0;'>⭐ {similarity_pct}</h3>", unsafe_allow_html=True)
					
					rec_tags = movies_df[movies_df["title"] == rec["title"]]["tags"].iloc[0]
					shared = get_shared_themes(selected_tags, rec_tags, top_n=3)
					if shared:
						explanation = f"Shares themes of {', '.join(shared)}"
						st.caption(explanation)
					else:
						st.caption("Similar content")
except FileNotFoundError:
	st.error("Missing movies.pkl or similarity.pkl. Run movie_recommender_.py first.")
