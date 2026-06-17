import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from src.inference import predict

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Eye Disease Detection",
    page_icon="👁️",
    layout="wide"
)

# ======================================
# HEADER
# ======================================

st.title(
    "👁️ Eye Disease Classification System"
)

st.markdown(
"""
Deep Learning Based Retinal Disease Detection

Models:
- ResNet50
- DenseNet121
- EfficientNet-B0

Technique:
- Weighted Ensemble
- Confidence Scoring
- Grad-CAM Explainability
"""
)

# ======================================
# FILE UPLOAD
# ======================================

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["png", "jpg", "jpeg"]
)

# ======================================
# INFERENCE
# ======================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with st.spinner("Analyzing Image..."):
        # Resizing here to avoid the Grad-CAM shape mismatch error observed previously
        img_resized = image.resize((224, 224))
        result = predict(img_resized)

    prediction = result["prediction"]
    confidence = result["confidence"]
    heatmap = result["heatmap"]
    probs = result["probabilities"]

    # ==============================
    # RESULT COLUMN
    # ==============================

    with col2:

        st.subheader("Prediction")

        st.success(
            prediction
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )

        if confidence < 0.60:

            st.warning(
                "Low confidence prediction. Review recommended."
            )

    # ==============================
    # HEATMAP
    # ==============================

    st.subheader(
        "Grad-CAM Explainability"
    )

    st.image(
        heatmap,
        use_container_width=True
    )

    # ==============================
    # PROBABILITIES
    # ==============================

    st.subheader(
        "Class Probabilities"
    )

    class_names = [
        "No DR",
        "Mild DR",
        "Moderate DR",
        "Severe DR",
        "Proliferative DR"
    ]

    df_probs = pd.DataFrame({
        "Disease": class_names,
        "Probability": probs
    })

    st.dataframe(
        df_probs,
        use_container_width=True
    )

    st.bar_chart(
        df_probs.set_index(
            "Disease"
        )
    )

    # ==============================
    # RECOMMENDATION
    # ==============================

    st.subheader(
        "Clinical Recommendation"
    )

    recommendations = {
        "No DR": "Routine annual retinal screening recommended.",
        "Mild DR": "Schedule follow-up eye examination.",
        "Moderate DR": "Consult an ophthalmologist for further evaluation.",
        "Severe DR": "Urgent specialist review advised.",
        "Proliferative DR": "Immediate ophthalmic intervention recommended."
    }

    st.info(
        recommendations[prediction]
    )

    # ==============================
    # DISCLAIMER
    # ==============================

    st.warning(
        """
        This application is intended for educational
        and research purposes only.

        It is NOT a substitute for professional
        medical diagnosis.
        """
    )