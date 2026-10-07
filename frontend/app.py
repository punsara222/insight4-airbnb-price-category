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
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; }
    .price-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 6px;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }
    .badge-low { background-color: #1f6f54; color: #eafff5; }
    .badge-medium { background-color: #9a6b1c; color: #fff6e6; }
    .badge-high { background-color: #8c2f39; color: #fff0f1; }
    .explain-line { font-size: 1.0rem; color: #c8d0d8; margin-top: 4px; }
    .field-hint { font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px; }
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