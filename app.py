from pathlib import Path
import json
import os
import pickle
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #FFB6D9 !important;
        }
        body {
            background-color: #FFB6D9 !important;
        }
        h1 {
            color: #C2185B !important;
        }
        .stMetric {
            background-color: rgba(255, 255, 255, 0.8) !important;
        }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MOVIES_PKL = BASE_DIR / "movies.pkl"
SIMILARITY_PKL = BASE_DIR / "similarity.pkl"

st.title("Movie Recommender System")
st.write("Pick a movie to see your top 5 recommendations")

api_key = os.getenv("TMDB_API_KEY", "").strip()


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
		{"title": movies_df.iloc[i].title, "movie_id": int(movies_df.iloc[i].movie_id)}
		for i, _ in matches
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


try:
	movies_df, similarity = load_artifacts()
	selected_movie = st.selectbox("Select a movie", movies_df["title"].tolist())
	if st.button("Recommend"):
		if not api_key:
			st.error("TMDB_API_KEY is not set. Posters require a TMDB API key.")
			st.stop()
		recs = recommend(selected_movie, movies_df, similarity)
		if not recs:
			st.warning("No recommendations found.")
		else:
			cols = st.columns(5)
			for col, rec in zip(cols, recs):
				with col:
					img = poster_url(rec["movie_id"], api_key)
					if img:
						st.image(img, use_container_width=True)
					st.caption(rec["title"])
except FileNotFoundError:
	st.error("Missing movies.pkl or similarity.pkl. Run movie_recommender_.py first.")
