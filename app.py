import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Preprocessing function for text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Custom stop words list (retain technical terms)
    stop_words = set([
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "what", "which", "this", "that",
        "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "doing", "a", "an", "the",
        "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
        "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
        "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "now"
    ])
    # Tokenize text and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Preprocess job description and resumes
    job_description = preprocess_text(job_description)
    resumes = [preprocess_text(resume) for resume in resumes]

    # Combine job description with resumes
    documents = [job_description] + resumes

    # Use bi-grams and tri-grams for better context understanding
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english').fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    # Normalize similarity scores for better scaling
    max_similarity = np.max(cosine_similarities) if np.max(cosine_similarities) > 0 else 1
    scaled_scores = [int((score / max_similarity) * 100) for score in cosine_similarities]

    return scaled_scores

# Streamlit app
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    h1, h2, h3 {
        color: #33475b;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #999;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Resume Screening & Candidate Ranking System")
st.subheader("Make your hiring process efficient and accurate!")

# Job description input
st.header("📄 Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# If both job description and resumes are provided
if uploaded_files and job_description:
    st.header("📊 Ranking Resumes")

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores as numerical values between 1 and 100
    results = pd.DataFrame({
        "Index": range(1, len(uploaded_files) + 1),
        "Resume": [file.name for file in uploaded_files],
        "Score (1-100)": scores,
    })

    results = results.sort_values(by="Score (1-100)", ascending=False)

    st.dataframe(results, use_container_width=True)

# Footer with your name
st.markdown(
    """
    <div class="footer">
        Created by <b>Kshithij Sangam</b>
    </div>
    """,
    unsafe_allow_html=True,
)
