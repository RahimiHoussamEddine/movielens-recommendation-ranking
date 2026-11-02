"""Local Streamlit demo for title-seeded recommendation strategies."""

from pathlib import Path

import streamlit as st

from movielens_recommendation_ranking.bundle import STRATEGIES, load_bundle, recommend_from_titles

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "artifacts" / "v1" / "model_bundle.npz"

st.set_page_config(page_title="MovieLens Recommendation Lab", page_icon="🎬", layout="wide")
st.title("MovieLens Recommendation Lab")
st.caption(
    "Explore title-seeded recommendations from a historical benchmark. "
    "Selections stay in this browser session and are not sent to an external service."
)


@st.cache_resource
def get_bundle():
    return load_bundle(BUNDLE_PATH)


try:
    bundle = get_bundle()
except FileNotFoundError as error:
    st.info(str(error))
    st.stop()

labels = [
    f"{title} [#{movie_id}]"
    for movie_id, title in zip(bundle["movie_ids"], bundle["titles"], strict=True)
]
with st.sidebar:
    st.header("Your seed titles")
    selected = st.multiselect(
        "Choose movies you like",
        options=labels,
        max_selections=10,
        help="Selections stay in this browser session and are not written to disk.",
    )
    strategy = st.selectbox("Ranking strategy", STRATEGIES, index=1)
    limit = st.slider("Recommendations", 5, 20, 10)

if not selected:
    st.write("Choose one or more titles in the sidebar to see recommendations.")
else:
    try:
        recommendations = recommend_from_titles(bundle, selected, strategy=strategy, limit=limit)
    except ValueError as error:
        st.error(str(error))
    else:
        st.subheader(f"{strategy} recommendations")
        display = recommendations.rename(
            columns={
                "movie_id": "Movie ID",
                "title": "Title",
                "genres": "Genres",
                "score": "Score",
            }
        )
        display["Score"] = display["Score"].map(lambda score: f"{score:.4f}")
        try:
            st.dataframe(display, use_container_width=True, hide_index=True)
        except ImportError as error:
            message = str(error).lower()
            if "pyarrow" not in message or "numpy" not in message:
                raise
            st.caption("Recommendations are shown in a compact list view.")
            for rank, row in enumerate(display.itertuples(index=False), start=1):
                st.markdown(f"**{rank}. {row.Title}**  \n{row.Genres} · score `{row.Score}`")
        st.caption(
            "Offline recommender scores are not calibrated probabilities or guarantees of preference."
        )

with st.expander("How this demo works"):
    st.write(
        "The study compares global positive-rating popularity, shrunk item-item "
        "co-occurrence, movie genres, and low-rank factors. This form averages "
        "each selected method's item representations and removes the selected "
        "titles from the results. The evaluation and limitations are in `reports/RESULTS.md`."
    )
