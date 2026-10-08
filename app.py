
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss

from pathlib import Path
from PIL import Image
from sklearn.preprocessing import normalize

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FashionAI",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(120, 80, 255, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(0, 200, 255, 0.07),
                transparent 30%
            ),
            #09090b;
        color: #f5f5f5;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 38px 42px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.08),
                rgba(255,255,255,0.025)
            );
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 30px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 8px;
    }

    .hero-title span {
        background: linear-gradient(
            90deg,
            #ffffff,
            #b9a7ff,
            #78d8ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #a1a1aa;
        font-size: 17px;
        line-height: 1.6;
        max-width: 760px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .upload-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .product-info {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 20px;
        padding: 22px;
        margin-top: 15px;
    }

    .product-name {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .recommendation-card {
        background: linear-gradient(
            145deg,
            rgba(255,255,255,0.065),
            rgba(255,255,255,0.025)
        );
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .rank {
        font-size: 13px;
        font-weight: 700;
        color: #c4b5fd;
        margin-bottom: 8px;
    }

    .rec-name {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        color: #d4d4d8;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 5px;
        margin-bottom: 7px;
    }

    .similarity {
        margin-top: 12px;
        color: #a7f3d0;
        font-weight: 700;
        font-size: 14px;
    }

    .empty-state {
        text-align: center;
        padding: 70px 20px;
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 22px;
        background: rgba(255,255,255,0.025);
    }

    .empty-icon {
        font-size: 55px;
        margin-bottom: 15px;
    }

    .empty-title {
        font-size: 22px;
        font-weight: 700;
    }

    .empty-text {
        color: #a1a1aa;
        margin-top: 8px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        background: linear-gradient(
            135deg,
            #7c3aed,
            #4f46e5
        );
        color: white;
        font-weight: 700;
        padding: 12px 20px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.025);
        border-radius: 16px;
    }

    [data-testid="stSidebar"] {
        background: #0d0d10;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sidebar-text {
        color: #a1a1aa;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 25px;
    }

    .info-box {
        background: rgba(124,58,237,0.10);
        border: 1px solid rgba(124,58,237,0.25);
        border-radius: 14px;
        padding: 14px;
        margin-top: 15px;
        color: #d4d4d8;
        font-size: 13px;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD PRODUCTS
# ============================================================

@st.cache_data
def load_products():

    return pd.read_csv(
        MODEL_DIR / "products_processed.csv"
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    pca = joblib.load(
        MODEL_DIR / "pca.pkl"
    )

    index = faiss.read_index(
        str(MODEL_DIR / "faiss_index.index")
    )

    cnn_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    return pca, index, cnn_model


products = load_products()

pca, index, cnn_model = load_models()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(uploaded_image):

    img = uploaded_image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    features = cnn_model.predict(
        img_array,
        verbose=0
    )

    reduced = pca.transform(
        features
    )

    normalized = normalize(
        reduced
    )

    return normalized.astype(
        "float32"
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_products(
    uploaded_image,
    top_n=8
):

    query_features = extract_features(
        uploaded_image
    )

    # Search a larger pool first.
    # This gives us enough candidates to filter.
    search_k = min(
        100,
        len(products)
    )

    similarities, indices = index.search(
        query_features,
        search_k
    )

    results = products.iloc[
        indices[0]
    ].copy()

    results["visual_similarity"] = similarities[0]

    return results.reset_index(
        drop=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">👗 FashionAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        AI-powered visual fashion discovery.
        Upload a product image and discover
        similar products from the catalog.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Search Settings")

    top_n = st.slider(
        "Recommendations",
        min_value=4,
        max_value=10,
        value=8
    )

    st.divider()

    st.markdown("### 🧠 AI Pipeline")

    st.caption("MobileNetV2")
    st.caption("↓")
    st.caption("PCA Feature Reduction")
    st.caption("↓")
    st.caption("FAISS Similarity Search")

    st.divider()

    st.markdown(
        """
        <div class="info-box">
        <b>Tip:</b> For better results, upload
        a clear image containing one main
        fashion product.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-title">
            Find products that
            <span>match your style.</span>
        </div>

        <div class="hero-subtitle">
            Upload a fashion product image and discover
            similar products using AI-powered visual search
            and intelligent recommendation technology.
        </div>

    </div>
    """
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📸 Visual Search</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your fashion product",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a clear image of a fashion product."
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# EMPTY STATE
# ============================================================

if not uploaded_file:

    st.html(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🛍️
            </div>

            <div class="empty-title">
                Start your visual search
            </div>

            <div class="empty-text">
                Upload a shirt, dress, shoe, accessory,
                or other fashion product to begin.
            </div>

        </div>
        """
    )


# ============================================================
# IMAGE UPLOADED
# ============================================================

if uploaded_file:

    uploaded_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # PRODUCT PREVIEW
    # ========================================================

    left, right = st.columns(
        [1, 2],
        gap="large"
    )


    with left:

        st.markdown(
            '<div class="section-title">Your Product</div>',
            unsafe_allow_html=True
        )

        st.image(
            uploaded_image,
            use_container_width=True
        )

        st.html(
            f"""
            <div class="product-info">

                <div class="product-name">
                    📷 Uploaded Product
                </div>

                <span class="badge">
                    {uploaded_file.name}
                </span>

                <span class="badge">
                    Visual Search
                </span>

            </div>
            """
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    with right:

        st.markdown(
            '<div class="section-title">✨ Recommendations</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Discover visually similar products from the catalog."
        )

        search_button = st.button(
            "🔍 Find Similar Products"
        )


        if search_button:

            with st.spinner(
                "AI is analyzing your product..."
            ):

                recommendations = recommend_products(
                    uploaded_image,
                    top_n=top_n
                )


            st.success(
                f"Found {len(recommendations)} visually similar products."
            )


            # =================================================
            # RECOMMENDATION CARDS
            # =================================================

            for i, row in recommendations.iterrows():

                similarity = float(
                    row["visual_similarity"]
                )

                similarity_percent = max(
                    0,
                    min(
                        100,
                        similarity * 100
                    )
                )


                product_name = str(
                    row["productDisplayName"]
                )

                gender = str(
                    row["gender"]
                )

                category = str(
                    row["masterCategory"]
                )

                subcategory = str(
                    row["subCategory"]
                )

                article_type = str(
                    row["articleType"]
                )

                colour = str(
                    row["baseColour"]
                )


                st.html(
                    f"""
                    <div class="recommendation-card">

                        <div class="rank">
                            #{i + 1} RECOMMENDATION
                        </div>

                        <div class="rec-name">
                            {product_name}
                        </div>

                        <span class="badge">
                            {gender}
                        </span>

                        <span class="badge">
                            {category}
                        </span>

                        <span class="badge">
                            {subcategory}
                        </span>

                        <span class="badge">
                            {article_type}
                        </span>

                        <span class="badge">
                            {colour}
                        </span>

                        <div class="similarity">
                            Visual Similarity:
                            {similarity_percent:.1f}%
                        </div>

                    </div>
                    """
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.html(
    """
    <div style="
        text-align:center;
        color:#71717a;
        font-size:13px;
        padding:15px;
    ">
        FashionAI • Deep Learning Powered Visual Product Discovery
    </div>
    """
)

