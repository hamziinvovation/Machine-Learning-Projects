"""
Movie Recommender System
=========================
A Streamlit product layer built on top of an existing, already-trained
content-based recommendation engine (CountVectorizer + cosine similarity
over a `tags` feature). See utils/recommender.py for the engine and
README.md for the full architecture and limitations.

Run with:
    streamlit run app.py
"""

import random

import streamlit as st

from utils import helpers, recommender, tmdb
from utils.recommender import ArtifactError
import streamlit as st

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

# ---------------------------------------------------------------------------
# Page config + global styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --mrs-bg: #0b0d14;
    --mrs-surface: #141824;
    --mrs-surface-2: #1b2030;
    --mrs-accent: #e63946;
    --mrs-text: #f2f3f5;
    --mrs-muted: #9aa0ae;
}
.stApp {
    background: radial-gradient(circle at top, #171b28 0%, var(--mrs-bg) 55%);
    color: var(--mrs-text);
}
#MainMenu, footer {visibility: hidden;}
section[data-testid="stSidebar"] {
    background-color: var(--mrs-surface);
    border-right: 1px solid #262c3d;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--mrs-surface);
    border-radius: 14px;
    border: 1px solid #262c3d;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--mrs-accent);
}
.hero-section {
    padding: 42px 36px;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(230,57,70,0.18), rgba(20,24,36,0.9));
    border: 1px solid #262c3d;
    margin-bottom: 28px;
}
.hero-badge {
    display: inline-block;
    background: var(--mrs-accent);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 14px;
    text-transform: uppercase;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 8px 0 14px 0;
    color: var(--mrs-text);
}
.section-subtitle {
    color: var(--mrs-muted);
    margin-top: -10px;
    margin-bottom: 16px;
    font-size: 0.9rem;
}
.movie-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 2px;
    color: var(--mrs-text);
}
.movie-meta {
    color: var(--mrs-muted);
    font-size: 0.82rem;
    margin-bottom: 6px;
}
.genre-pill {
    display: inline-block;
    background: var(--mrs-surface-2);
    color: var(--mrs-muted);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px 4px 2px 0;
    border: 1px solid #262c3d;
}
.poster-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    height: 260px;
    border-radius: 12px;
    background: var(--mrs-surface-2);
    color: var(--mrs-muted);
    font-size: 2.4rem;
    border: 1px dashed #33394d;
}
.poster-fallback span.caption {
    font-size: 0.75rem;
    margin-top: 8px;
}
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--mrs-muted);
}
.empty-state .icon {
    font-size: 2.6rem;
    margin-bottom: 10px;
}
.profile-card {
    background: var(--mrs-surface-2);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    border: 1px solid #262c3d;
}
.stButton > button {
    border-radius: 8px;
    border: 1px solid #33394d;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

helpers.initialize_session_state()


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------
def render_poster(poster_url, caption="Poster unavailable", height=None):
    if poster_url:
        st.image(poster_url, use_container_width=True)
    else:
        st.markdown(
            f'<div class="poster-fallback">🎬<span class="caption">{caption}</span></div>',
            unsafe_allow_html=True,
        )


def display_movie_card(movie_id, title, context="card", extra_line=None):
    """
    One reusable movie-card component used everywhere in the app:
    Home, Search, Discover, Mood, Watchlist, History, Favorites,
    Recommendations. `context` must be unique per call-site so button
    keys never collide when the same movie appears more than once.
    """
    details = tmdb.get_movie_details(movie_id)

    with st.container(border=True):
        render_poster(details["poster_url"])

        st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)

        rating = helpers.format_rating(details["vote_average"]) if details["available"] else "Not rated"
        year = helpers.get_year(details["release_date"]) if details["available"] else "—"
        meta_bits = [f"⭐ {rating}", year]
        if extra_line:
            meta_bits.append(extra_line)
        st.markdown(
            f'<div class="movie-meta">{" • ".join(meta_bits)}</div>',
            unsafe_allow_html=True,
        )

        genres = details["genres"] if details["available"] else []
        if not genres:
            row = recommender.get_movie_row_by_id(movie_id)
            if row:
                genres = helpers.extract_genres(row.get("tags", ""))
        if genres:
            pills = "".join(f'<span class="genre-pill">{g}</span>' for g in genres[:3])
            st.markdown(pills, unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        if b1.button("Details", key=f"details_{context}_{movie_id}", use_container_width=True):
            helpers.go_to_details(title)
            st.rerun()

        in_watchlist = movie_id in st.session_state.watchlist
        watchlist_label = "✓ Saved" if in_watchlist else "＋ Watchlist"
        if b2.button(watchlist_label, key=f"wl_{context}_{movie_id}", use_container_width=True):
            if in_watchlist:
                helpers.remove_from_watchlist(movie_id)
                st.toast("Removed from Watchlist")
            else:
                helpers.add_to_watchlist(movie_id)
                st.toast("✓ Added to Watchlist")
            st.rerun()

        is_fav = movie_id in st.session_state.favorites
        fav_label = "♥ Loved" if is_fav else "♡ Favorite"
        if b3.button(fav_label, key=f"fav_{context}_{movie_id}", use_container_width=True):
            added = helpers.toggle_favorite(movie_id)
            st.toast("❤️ Added to Favorites" if added else "Removed from Favorites")
            st.rerun()


def display_movie_grid(movie_rows, context, columns=4, extra_line_fn=None):
    """movie_rows: iterable of dicts with at least movie_id + title."""
    rows = list(movie_rows)
    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, row in zip(cols, rows[start : start + columns]):
            with col:
                extra = extra_line_fn(row) if extra_line_fn else None
                display_movie_card(
                    row["movie_id"], row["title"], context=f"{context}_{start}", extra_line=extra
                )


def empty_state(icon, title, subtitle, cta_label=None, cta_page=None):
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="icon">{icon}</div>
            <div style="font-size:1.1rem;color:#f2f3f5;font-weight:600;">{title}</div>
            <div style="margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if cta_label and cta_page:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            if st.button(cta_label, use_container_width=True):
                st.session_state.page = cta_page
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("## 🎬 MOVIE RECOMMENDER")
    st.sidebar.caption("Find something you'll love to watch.")
    st.sidebar.markdown("---")

    nav_items = [
        ("🏠", "Home"),
        ("🔎", "Search"),
        ("🎭", "Discover"),
        ("😊", "Mood"),
        ("🔖", "Watchlist"),
        ("🕐", "History"),
        ("❤️", "Favorites"),
        ("👤", "Profile"),
    ]
    for icon, name in nav_items:
        label = f"{icon}  {name}"
        if st.session_state.page == name:
            label = f"➤ {label}"
        if st.sidebar.button(label, use_container_width=True, key=f"nav_{name}"):
            st.session_state.page = name
            st.rerun()

    st.sidebar.markdown("---")
    if not tmdb.is_configured():
        st.sidebar.caption(
            "ℹ️ Posters & ratings are limited — add a `TMDB_API_KEY` in "
            "`.streamlit/secrets.toml` to enable them."
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    movies_df = recommender.get_movies_df()

    if st.session_state.hero_movie_id is None:
        st.session_state.hero_movie_id = int(movies_df.sample(1, random_state=random.randint(0, 10_000))["movie_id"].iloc[0])

    hero_row = recommender.get_movie_row_by_id(st.session_state.hero_movie_id)
    hero_details = tmdb.get_movie_details(st.session_state.hero_movie_id)

    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    hc1, hc2 = st.columns([1, 2])
    with hc1:
        render_poster(hero_details["poster_url"])
    with hc2:
        st.markdown('<span class="hero-badge">Featured</span>', unsafe_allow_html=True)
        st.markdown(f"## {hero_row['title']}")
        rating = helpers.format_rating(hero_details["vote_average"]) if hero_details["available"] else "Not rated"
        year = helpers.get_year(hero_details["release_date"]) if hero_details["available"] else "—"
        runtime = helpers.format_runtime(hero_details["runtime"]) if hero_details["available"] else "Runtime unavailable"
        genres = hero_details["genres"] if hero_details["available"] else helpers.extract_genres(hero_row.get("tags", ""))
        st.markdown(f"⭐ {rating}  •  {year}  •  {runtime}")
        if genres:
            st.markdown("".join(f'<span class="genre-pill">{g}</span>' for g in genres), unsafe_allow_html=True)
        overview = hero_details["overview"] if hero_details["available"] else None
        st.write(overview or "No overview available.")

        a1, a2, a3 = st.columns(3)
        if a1.button("▶ Explore", use_container_width=True, key="hero_explore"):
            helpers.go_to_details(hero_row["title"])
            st.rerun()
        in_wl = st.session_state.hero_movie_id in st.session_state.watchlist
        if a2.button("✓ Saved" if in_wl else "＋ Watchlist", use_container_width=True, key="hero_wl"):
            if in_wl:
                helpers.remove_from_watchlist(st.session_state.hero_movie_id)
            else:
                helpers.add_to_watchlist(st.session_state.hero_movie_id)
                st.toast("✓ Added to Watchlist")
            st.rerun()
        is_fav = st.session_state.hero_movie_id in st.session_state.favorites
        if a3.button("♥ Loved" if is_fav else "♡ Favorite", use_container_width=True, key="hero_fav"):
            helpers.toggle_favorite(st.session_state.hero_movie_id)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Trending (real TMDB trending data when available, honestly labeled)
    trending = tmdb.get_trending_week()
    if trending:
        st.markdown('<div class="section-title">🔥 Trending This Week</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Live data from TMDB</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for col, m in zip(cols, trending[:5]):
            with col:
                with st.container(border=True):
                    render_poster(m["poster_url"])
                    st.markdown(f'<div class="movie-title">{m["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="movie-meta">⭐ {helpers.format_rating(m["vote_average"])} • {helpers.get_year(m["release_date"])}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown('<div class="section-title">🎬 Featured Picks</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">A rotating sample from the catalog</div>',
            unsafe_allow_html=True,
        )
        sample = recommender.sample_movies(5, seed=st.session_state.hero_movie_id)
        display_movie_grid(sample.to_dict("records"), context="home_featured", columns=5)

    # Because You Watched
    if st.session_state.history:
        last_watched = st.session_state.history[0]
        recs = recommender.recommend_movies(last_watched["title"], n=5)
        if recs:
            st.markdown(
                f'<div class="section-title">🎯 Because You Watched "{last_watched["title"]}"</div>',
                unsafe_allow_html=True,
            )
            display_movie_grid(recs, context="home_bwatched", columns=5)

    # Recommended For You (from favorites/watchlist genre signal)
    preferred = helpers.preferred_genres_from_activity(recommender.get_movie_row_by_id)
    if preferred:
        st.markdown('<div class="section-title">✨ Recommended For You</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Personalized from your activity</div>', unsafe_allow_html=True)
        candidates = movies_df[movies_df["tags"].apply(lambda t: bool(set(helpers.extract_genres(t)) & set(preferred)))]
        candidates = candidates[~candidates["movie_id"].isin(
            st.session_state.watchlist + st.session_state.favorites + [h["movie_id"] for h in st.session_state.history]
        )]
        if not candidates.empty:
            display_movie_grid(candidates.sample(min(5, len(candidates))).to_dict("records"), context="home_reco", columns=5)

    # Mood teaser
    st.markdown('<div class="section-title">😊 What\'s Your Mood?</div>', unsafe_allow_html=True)
    mood_cols = st.columns(len(helpers.MOOD_GENRES))
    for col, (mood, info) in zip(mood_cols, helpers.MOOD_GENRES.items()):
        with col:
            if st.button(f"{info['emoji']}\n{mood}", use_container_width=True, key=f"home_mood_{mood}"):
                st.session_state["mood_selected"] = mood
                st.session_state.page = "Mood"
                st.rerun()

    # Watchlist preview
    if st.session_state.watchlist:
        st.markdown('<div class="section-title">🔖 From Your Watchlist</div>', unsafe_allow_html=True)
        preview_rows = [recommender.get_movie_row_by_id(mid) for mid in st.session_state.watchlist[-5:]]
        preview_rows = [r for r in preview_rows if r]
        display_movie_grid(preview_rows, context="home_wl", columns=5)


def page_search():
    st.markdown('<div class="section-title">🔍 Search Movies</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Search", placeholder="Type a movie, actor, director, or genre keyword...", label_visibility="collapsed",
        value=st.session_state.search_query,
    )
    st.session_state.search_query = query

    if not query.strip():
        st.info("Start typing to search across titles, cast, directors and genres.")
        return

    results = recommender.search_movies(query, limit=24)
    st.caption(f"{len(results)} movie{'s' if len(results) != 1 else ''} found")

    if results.empty:
        empty_state("🔍", "No movies found", f'Nothing matched "{query}". Try a different title or keyword.')
        return

    display_movie_grid(results.to_dict("records"), context="search", columns=4)


def page_discover():
    st.markdown('<div class="section-title">🎭 Discover Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Browse by genre, pulled directly from the catalog</div>', unsafe_allow_html=True)

    all_genres = sorted(set(helpers.GENRE_TOKENS.values()))
    selected = st.multiselect("Genre", all_genres, default=st.session_state.discover_genre)
    st.session_state.discover_genre = selected

    sort_by = st.selectbox("Sort by", ["Title (A-Z)", "Random shuffle"])

    movies_df = recommender.get_movies_df()
    if selected:
        filtered = movies_df[movies_df["tags"].apply(lambda t: bool(set(helpers.extract_genres(t)) & set(selected)))]
    else:
        filtered = movies_df

    st.caption(f"{len(filtered)} movies match your filters")

    if filtered.empty:
        empty_state("🎭", "No matches", "Try selecting fewer genres.")
        return

    if sort_by == "Title (A-Z)":
        filtered = filtered.sort_values("title")
    else:
        filtered = filtered.sample(frac=1, random_state=42)

    limit = st.session_state.discover_offset
    display_movie_grid(filtered.head(limit).to_dict("records"), context="discover", columns=4)

    if limit < len(filtered):
        if st.button("Load More", use_container_width=True):
            st.session_state.discover_offset += 12
            st.rerun()


def page_mood():
    st.markdown('<div class="section-title">😊 What\'s Your Mood Today?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A transparent genre-mapping layer on top of the recommender — '
        "not a trained emotion model.</div>",
        unsafe_allow_html=True,
    )

    moods = list(helpers.MOOD_GENRES.items())
    for row_start in range(0, len(moods), 4):
        cols = st.columns(4)
        for col, (mood, info) in zip(cols, moods[row_start : row_start + 4]):
            with col:
                if st.button(f"{info['emoji']}  {mood}", use_container_width=True, key=f"mood_btn_{mood}"):
                    st.session_state["mood_selected"] = mood

    selected_mood = st.session_state.get("mood_selected")
    if selected_mood:
        info = helpers.MOOD_GENRES[selected_mood]
        st.markdown(
            f'<div class="section-title">{info["emoji"]} Movies for a {selected_mood.lower()} mood</div>',
            unsafe_allow_html=True,
        )
        st.caption("Matched genres: " + ", ".join(info["genres"]))

        movies_df = recommender.get_movies_df()
        target_genres = set(info["genres"])
        matches = movies_df[movies_df["tags"].apply(lambda t: bool(set(helpers.extract_genres(t)) & target_genres))]

        if matches.empty:
            empty_state("😶", "No matches yet", "Try a different mood.")
        else:
            sample = matches.sample(min(12, len(matches)), random_state=hash(selected_mood) % 1000)
            display_movie_grid(sample.to_dict("records"), context="mood", columns=4)


def page_watchlist():
    st.markdown('<div class="section-title">🔖 My Watchlist</div>', unsafe_allow_html=True)
    if not st.session_state.watchlist:
        empty_state(
            "🔖", "Your watchlist is empty.",
            "Save movies here to watch them later.",
            cta_label="Discover Movies", cta_page="Discover",
        )
        return

    rows = [recommender.get_movie_row_by_id(mid) for mid in reversed(st.session_state.watchlist)]
    rows = [r for r in rows if r]
    display_movie_grid(rows, context="watchlist", columns=4)


def page_history():
    st.markdown('<div class="section-title">🕐 Watch History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        empty_state("🕐", "No watch history yet.", "Start exploring movies.", cta_label="Discover Movies", cta_page="Discover")
        return

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

    for item in st.session_state.history:
        details = tmdb.get_movie_details(item["movie_id"])
        with st.container(border=True):
            c1, c2 = st.columns([1, 5])
            with c1:
                render_poster(details["poster_url"], caption="No poster")
            with c2:
                st.markdown(f"**{item['title']}**")
                st.caption(f"Watched: {item['timestamp']}")
                if st.button("View Details", key=f"hist_view_{item['movie_id']}"):
                    helpers.go_to_details(item["title"])
                    st.rerun()


def page_favorites():
    st.markdown('<div class="section-title">❤️ Favorites</div>', unsafe_allow_html=True)
    if not st.session_state.favorites:
        empty_state("❤️", "No favorites yet.", "Find a movie you love.", cta_label="Discover Movies", cta_page="Discover")
        return

    rows = [recommender.get_movie_row_by_id(mid) for mid in reversed(st.session_state.favorites)]
    rows = [r for r in rows if r]
    display_movie_grid(rows, context="favorites", columns=4)


def page_profile():
    st.markdown('<div class="section-title">👤 Profile</div>', unsafe_allow_html=True)
    st.caption("Session-based profile — your activity is remembered for this browser session only, not stored in a database.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div class="profile-card">
                <div style="font-size:2.4rem;">👤</div>
                <div style="font-size:1.2rem;font-weight:700;margin-top:6px;">{st.session_state.username}</div>
                <div style="color:#9aa0ae;margin-top:14px;">
                    🎬 Watched &nbsp;<b>{len(st.session_state.history)}</b><br>
                    🔖 Watchlist &nbsp;<b>{len(st.session_state.watchlist)}</b><br>
                    ❤️ Favorites &nbsp;<b>{len(st.session_state.favorites)}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        new_name = st.text_input("Display name", value=st.session_state.username)
        if new_name != st.session_state.username and new_name.strip():
            st.session_state.username = new_name.strip()
            st.rerun()

    with col2:
        st.markdown("**Favorite Genres**")
        preferred = helpers.preferred_genres_from_activity(recommender.get_movie_row_by_id)
        if preferred:
            st.markdown(
                "".join(f'<span class="genre-pill">{g}</span>' for g in preferred[:6]),
                unsafe_allow_html=True,
            )
            st.caption("Calculated from your watch history, watchlist and favorites this session.")
        else:
            st.caption("Not enough activity yet to detect a genre pattern — watch or save a few movies first.")

        st.markdown("**Recent Activity**")
        if st.session_state.history:
            for item in st.session_state.history[:5]:
                st.write(f"• {item['title']} — {item['timestamp']}")
        else:
            st.caption("No recent activity.")


def page_movie_details():
    title = st.session_state.selected_movie
    if not title:
        empty_state("🎬", "No movie selected", "Search or browse to pick a movie.", cta_label="Discover Movies", cta_page="Discover")
        return

    row = recommender.get_movie_row(title)
    if row is None:
        st.error(f"Movie not found: '{title}'. Try another title.")
        return

    movie_id = int(row["movie_id"])
    details = tmdb.get_movie_details(movie_id)

    if st.button("← Back"):
        st.session_state.selected_movie = None
        st.rerun()

    c1, c2 = st.columns([1, 2])
    with c1:
        render_poster(details["poster_url"])
    with c2:
        st.markdown(f"## {row['title']}")
        rating = helpers.format_rating(details["vote_average"]) if details["available"] else "Not rated"
        year = helpers.get_year(details["release_date"]) if details["available"] else "—"
        runtime = helpers.format_runtime(details["runtime"]) if details["available"] else "Runtime unavailable"
        vote_count = helpers.format_number(details["vote_count"]) if details["available"] and details["vote_count"] else None
        rating_line = f"⭐ {rating}"
        if vote_count:
            rating_line += f" ({vote_count} votes)"
        st.markdown(f"{rating_line}  •  {year}  •  {runtime}")

        genres = details["genres"] if details["available"] else helpers.extract_genres(row.get("tags", ""))
        if genres:
            st.markdown("".join(f'<span class="genre-pill">{g}</span>' for g in genres), unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)
        in_wl = movie_id in st.session_state.watchlist
        if a1.button("✓ Saved" if in_wl else "＋ Watchlist", use_container_width=True, key="details_wl"):
            if in_wl:
                helpers.remove_from_watchlist(movie_id)
            else:
                helpers.add_to_watchlist(movie_id)
                st.toast("✓ Added to Watchlist")
            st.rerun()
        if a2.button("✓ Watched", use_container_width=True, key="details_watched"):
            helpers.mark_watched(movie_id, row["title"])
            st.toast("✓ Marked as Watched")
            st.rerun()
        is_fav = movie_id in st.session_state.favorites
        if a3.button("♥ Loved" if is_fav else "♡ Favorite", use_container_width=True, key="details_fav"):
            helpers.toggle_favorite(movie_id)
            st.rerun()

    st.markdown("### Overview")
    st.write(details["overview"] if details["available"] and details["overview"] else "No overview available.")

    if details["available"] and details["cast"]:
        st.markdown("### Cast")
        st.write(", ".join(details["cast"]))
    if details["available"] and details["director"]:
        st.markdown("### Director")
        st.write(details["director"])

    st.markdown(f'<div class="section-title">Because You Like This Movie</div>', unsafe_allow_html=True)
    recs = recommender.recommend_movies(title, n=5)
    if recs:
        display_movie_grid(
            recs, context="details_recs", columns=5,
            extra_line_fn=lambda r: f"Match {r['score']*100:.0f}%",
        )
    else:
        st.caption("No similar movies found.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Home": page_home,
    "Search": page_search,
    "Discover": page_discover,
    "Mood": page_mood,
    "Watchlist": page_watchlist,
    "History": page_history,
    "Favorites": page_favorites,
    "Profile": page_profile,
    "Movie Details": page_movie_details,
}


def main():
    try:
        recommender.get_movies_df()  # eagerly validate artifacts once
    except ArtifactError as e:
        st.error(str(e))
        st.stop()

    render_sidebar()

    page_fn = PAGES.get(st.session_state.page, page_home)
    page_fn()


if __name__ == "__main__":
    main()
