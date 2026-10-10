"""
Insight4 - NYC Airbnb Price Category Predictor
Frontend (Streamlit) - connects to the FastAPI backend at http://127.0.0.1:8000
Owner: IT23728844
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="NYC Airbnb Price Category",
    page_icon="\U0001F3D9",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Styling - category colors
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Styling - Light & Dark Mode
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --parchment: #F1EAD8;
        --sand: #D5C7AD;
        --olive: #BEC5A4;
        --sage: #8A8E75;
        --bark: #68604D;
        --olivewood: #2D2F22;
        --card: #3A3D2E;
        --track: #4A4D3C;
        --sage-deep: #70745C;
    }

    /* ===== FORCED DARK MODE (MATCHING SCREENSHOT) ===== */
    .stApp { background-color: var(--olivewood) !important; color: var(--parchment) !important; }
    h1, h2, h3 { color: var(--parchment) !important; }
    p, label, .stMarkdown { color: var(--parchment) !important; }
    [data-testid="stWidgetLabel"] p { font-weight: 600 !important; color: var(--parchment) !important; }

    /* Form card */
    [data-testid="stForm"] {
        background-color: var(--card) !important;
        border: 1px solid var(--bark) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    /* Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #2A2E20 !important;
        border: 1px solid var(--sage) !important;
        border-radius: 7px !important;
    }
    div[data-baseweb="select"] span { color: var(--parchment) !important; }
    div[role="listbox"] { background-color: #2A2E20 !important; }
    div[role="option"] { color: var(--parchment) !important; }
    div[role="option"]:hover { background-color: var(--bark) !important; }

    /* Number inputs */
    div[data-testid="stNumberInput"] { background-color: #2A2E20 !important; border-radius: 7px !important; }
    div[data-testid="stNumberInput"] input { background-color: #2A2E20 !important; color: var(--parchment) !important; }
    div[data-testid="stNumberInput"] button { color: var(--sand) !important; }

    /* Predict button */
    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        background-color: var(--olive) !important;
        border: none !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover { background-color: var(--sand) !important; }
    .stButton > button p,
    [data-testid="stFormSubmitButton"] button p { color: var(--olivewood) !important; }

    /* Section headings */
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        padding: 6px 14px; margin: 18px 0 10px;
        border-radius: 6px; border-left: 5px solid;
    }
    .sec-location { background: var(--olive) !important;     color: var(--olivewood) !important; border-left-color: var(--sage-deep) !important; }
    .sec-property { background: var(--sand) !important;      color: var(--olivewood) !important; border-left-color: var(--bark) !important; }
    .sec-capacity { background: var(--sand) !important;      color: var(--olivewood) !important; border-left-color: var(--sand) !important; }
    .sec-result   { background: var(--card) !important;      color: var(--parchment) !important; border-left-color: var(--olive) !important; }

    /* Result badge */
    .price-badge {
        display: inline-block; padding: 10px 22px; border-radius: 8px;
        font-size: 1.4rem; font-weight: 700; letter-spacing: 0.02em; margin-bottom: 8px;
    }
    .badge-low    { background-color: #DCFCE7 !important; color: #166534 !important; }
    .badge-medium { background-color: #FEF3C7 !important; color: #92400E !important; }
    .badge-high   { background-color: #FEE2E2 !important; color: #991B1B !important; }
    .explain-line { font-size: 1rem; color: var(--sand) !important; margin-top: 4px; }

    /* Probability bars */
    .prob-row { display: flex; justify-content: space-between; font-size: 0.95rem; color: var(--parchment) !important; margin-top: 10px; }
    .prob-track { background: var(--track) !important; border-radius: 6px; height: 12px; overflow: hidden; margin-top: 4px; }
    .prob-fill { height: 100%; border-radius: 6px; }
    .fill-low { background: #DCFCE7 !important; }
    .fill-medium { background: #FEF3C7 !important; }
    .fill-high { background: #FEE2E2 !important; }

    /* NOTE CALLOUT BANNER (MATCHING SCREENSHOT) */
    [data-testid="stForm"] [data-testid="stCaptionContainer"] {
        background-color: #8E8A74 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        border-left: 6px solid var(--olive) !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
    }

    [data-testid="stForm"] [data-testid="stCaptionContainer"] p {
        color: #2D2F22 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        line-height: 1.6 !important;
    }

    [data-testid="stForm"] [data-testid="stCaptionContainer"] code {
        background-color: #FAF7EE !important;
        color: #2D2F22 !important;
        font-weight: 700 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
    }

    [data-testid="stTooltipIcon"] svg { color: var(--sand) !important; fill: var(--sand) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("NYC Airbnb \u2014 Price Category Predictor")
st.caption(
    "Fill in a listing's details below to estimate whether it falls in the "
    "Low, Medium, or High price range for New York City."
)


# ---------------------------------------------------------------------------
# Load schema
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_schema():
    resp = requests.get(f"{API_BASE}/schema", timeout=10)
    resp.raise_for_status()
    return resp.json()


try:
    schema = load_schema()
except Exception:
    st.error(
        "Could not reach the backend at "
        f"{API_BASE}. Make sure it's running "
        "(`uvicorn backend.app:app --reload`) and try again."
    )
    st.stop()

fields = schema["fields"]
neighbourhoods_by_borough = schema["neighbourhoods_by_borough"]

PRICE_ORDER = ["Low", "Medium", "High"]
BADGE_CLASS = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}
EXPLAIN_LINE = {
    "Low": "This listing is likely in the Low price range for NYC.",
    "Medium": "This listing is likely in the Medium price range for NYC.",
    "High": "This listing is likely in the High price range for NYC.",
}


# ---------------------------------------------------------------------------
# Numeric input helper with tooltip support
# ---------------------------------------------------------------------------
def _num_input(container, label, key, help_text=None):
    meta = fields[key]
    lo = float(meta["min"])
    hi = float(meta["max"])

    if lo.is_integer() and hi.is_integer():
        return container.number_input(
            label,
            min_value=int(lo),
            max_value=int(hi),
            value=int(lo),
            step=1,
            help=help_text,
        )
    else:
        return container.number_input(
            label,
            min_value=lo,
            max_value=hi,
            value=lo,
            step=0.5,
            help=help_text,
        )


# ---------------------------------------------------------------------------
# Location inputs (reactive updates outside form)
# ---------------------------------------------------------------------------
st.subheader("Location")
col1, col2 = st.columns(2)

with col1:
    borough = st.selectbox(
        "Borough",
        options=fields["neighbourhood_group_cleansed"]["allowed"],
    )

neighbourhood_options = neighbourhoods_by_borough.get(borough, [])
with col2:
    neighbourhood = st.selectbox("Neighbourhood", options=neighbourhood_options)


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("listing_form"):
    st.subheader("Property")
    col3, col4 = st.columns(2)

    with col3:
        room_type = st.selectbox(
            "Room type", options=fields["room_type"]["allowed"]
        )
    with col4:
        property_type = st.selectbox(
            "Property type", options=fields["property_type"]["allowed"]
        )

    st.subheader("Capacity")
    st.caption(
        "💡 **Note:** Bathrooms accept decimal values (`0.5`, `1.5`) where `0.5` represents a half-bathroom (powder room with toilet & sink only). Bedrooms can be `0` for Studio apartments."
    )
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        accommodates = _num_input(col5, "Guests", "accommodates", help_text="Total guest capacity.")
    with col6:
        bathrooms = _num_input(
            col6,
            "Bathrooms",
            "bathrooms",
            help_text="Decimals allowed: 0.5 represents a half-bath (toilet + sink only, no shower/tub).",
        )
    with col7:
        bedrooms = _num_input(
            col7,
            "Bedrooms",
            "bedrooms",
            help_text="Set to 0 for Studio apartments (combined living and sleeping area).",
        )
    with col8:
        beds = _num_input(col8, "Beds", "beds", help_text="Total number of beds available.")

    submitted = st.form_submit_button("Predict price category", use_container_width=True)


# ---------------------------------------------------------------------------
# Execution and Results
# ---------------------------------------------------------------------------
if submitted:
    if not neighbourhood:
        st.warning("No neighbourhoods are available for this borough. Please pick another borough.")
        st.stop()

    payload = {
        "features": {
            "neighbourhood_group_cleansed": borough,
            "neighbourhood_cleansed": neighbourhood,
            "room_type": room_type,
            "property_type": property_type,
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,
        }
    }

    with st.spinner("Scoring listing..."):
        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=15)
        except Exception:
            st.error(
                "Could not reach the backend. Make sure it's running at "
                f"{API_BASE} and try again."
            )
            st.stop()

    if resp.status_code == 200:
        result = resp.json()
        category = result["price_category"]
        probabilities = result["probabilities"]
        confidence = result["confidence"]

        badge_class = BADGE_CLASS.get(category, "badge-medium")
        st.markdown(
            f'<div class="price-badge {badge_class}">{category} price</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="explain-line">{EXPLAIN_LINE.get(category, "")} '
            f"Confidence: {confidence * 100:.0f}%.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        st.subheader("Probability by category")
        for label in PRICE_ORDER:
            pct = probabilities.get(label, 0.0) * 100
            st.write(f"{label}: {pct:.1f}%")
            st.progress(min(max(pct / 100, 0.0), 1.0))

    elif resp.status_code == 422:
        error_body = resp.json()
        st.error(error_body.get("message", "Invalid input."))
        for err in error_body.get("errors", []):
            st.write(f"- {err}")

    else:
        st.error(f"Unexpected error from backend (status {resp.status_code}).")
        st.code(resp.text)

