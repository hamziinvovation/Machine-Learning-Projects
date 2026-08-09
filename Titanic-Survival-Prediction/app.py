"""
Titanic Survival Prediction System
===================================
Main Streamlit entry point.

Wires together a custom glassmorphism UI, a sidebar navigation menu, and
the five app pages defined under `views/`. All Machine Learning logic
lives in `model/` and `utils/` — this file is purely presentation and
routing.

Run with:
    streamlit run app.py
"""

import streamlit as st

from utils.styling import load_css
from views import about, dataset_info, home, model_details, prediction

st.set_page_config(
    page_title="Titanic Survival Prediction System",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

PAGES = {
    "Home": ("🏠", home),
    "Prediction": ("📊", prediction),
    "Model Details": ("🤖", model_details),
    "Dataset Information": ("📈", dataset_info),
    "About Developer": ("👨‍💻", about),
}


def _navigate(page_name: str) -> None:
    """Programmatically switch the active page and rerun."""
    st.session_state["active_page"] = page_name
    st.rerun()


def _sidebar() -> None:
    """Render the branded sidebar with icon navigation buttons."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">🚢 Titanic AI</div>
            <div class="sidebar-tag">Survival Prediction System</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        for name, (icon, _module) in PAGES.items():
            is_active = st.session_state["active_page"] == name
            label = f"{icon}  {name}"
            if st.button(
                label,
                key=f"nav_{name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _navigate(name)

        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.78rem;color:#94A3B8;line-height:1.5;">
            Machine Learning Powered<br/>
            Developed by <b style="color:#E2E8F0;">Hamza Khan</b><br/>
            CECOS University
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Home"

    _sidebar()

    active_page = st.session_state["active_page"]
    _icon, module = PAGES[active_page]
    module.render(_navigate)

    st.markdown(
        """
        <div class="app-footer">
            🚢 Machine Learning Powered &nbsp;|&nbsp; Developed by
            <b>Hamza Khan</b> &nbsp;|&nbsp; CECOS University
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
