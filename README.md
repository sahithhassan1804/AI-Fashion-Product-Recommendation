# 👗 FashionAI — Visual Product Recommendation System

An AI-powered fashion product recommendation system that uses computer vision and visual similarity to recommend fashion products based on an uploaded product image.

## 📌 Project Overview

FashionAI helps users discover visually similar fashion products by uploading an image of a clothing or fashion item.

The system extracts visual features from the uploaded image using **MobileNetV2**, reduces the feature dimensions using **PCA**, and performs similarity search using **FAISS**.

The application provides an interactive web interface built with **Streamlit**.

---

## 🎯 Objectives

- Recommend visually similar fashion products from an uploaded image.
- Use deep learning-based image features for visual search.
- Reduce high-dimensional image features using PCA.
- Perform fast similarity search using FAISS.
- Provide an easy-to-use web application for users.
- Demonstrate the practical application of computer vision in fashion e-commerce.

---

## ✨ Features

- 🖼️ Upload a fashion product image
- 🤖 AI-based visual feature extraction
- 🔍 Visual similarity search
- 👗 Fashion product recommendations
- 📊 Similarity scores for recommendations
- ⚡ Fast search using FAISS
- 💻 Interactive Streamlit interface
- 🎨 Modern fashion-focused UI

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning feature extraction |
| MobileNetV2 | Image feature extraction |
| Scikit-learn | PCA and feature normalization |
| FAISS | Similarity search |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| Joblib | Model serialization |
| Streamlit | Web application |

---

## 🔄 System Workflow

```text
User uploads image
        ↓
Image preprocessing
        ↓
MobileNetV2
        ↓
Visual feature extraction
        ↓
PCA dimensionality reduction
        ↓
Feature normalization
        ↓
FAISS similarity search
        ↓
Retrieve visually similar products
        ↓
Display recommendations