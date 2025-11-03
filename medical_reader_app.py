# Filename: medical_reader_app.py
import streamlit as st
import fitz  # PyMuPDF

from PIL import Image
import io
import re
from transformers import pipeline
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print(pytesseract.get_tesseract_version())

# ---- Setup ----
st.set_page_config(page_title="🩺 Medical Report Summarizer", layout="wide")
st.title("🩺 Medical AI Assistant (Offline)")

st.write("Upload your **medical reports or scan images** to get a summarized interpretation. No API required, runs locally!")

# Initialize summarization model (offline)
@st.cache_resource
def load_summarizer():
    # Using a small but accurate local model
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ---- File Upload ----
uploaded_file = st.file_uploader("📄 Upload a medical report (PDF, JPG, PNG, TXT)", type=["pdf", "jpg", "jpeg", "png", "txt"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    extracted_text = ""

    # ---- PDF ----
    if file_type == "pdf":
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf:
            extracted_text += page.get_text("text")

    # ---- Image ----
    elif file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        extracted_text = text

    # ---- Text ----
    elif file_type == "txt":
        extracted_text = uploaded_file.read().decode("utf-8")

    # ---- Clean extracted text ----
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

    if len(extracted_text) < 20:
        st.warning("⚠️ Could not extract readable medical text. Try another file.")
    else:
        st.subheader("🧾 Extracted Report Content")
        with st.expander("Show Extracted Text"):
            st.write(extracted_text)

        # ---- Summarization ----
        st.subheader("🩸 AI Summary / Interpretation")
        with st.spinner("Analyzing report..."):
            try:
                summary = summarizer(extracted_text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
                st.success("✅ Summary generated successfully!")
                st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")

        # ---- Optional Highlighted Findings ----
        st.subheader("🧠 Possible Key Findings")
        findings = []
        for keyword in ["glucose", "bilirubin", "bp", "blood", "infection", "liver", "kidney", "cholesterol", "rbc", "wbc"]:
            if keyword.lower() in extracted_text.lower():
                findings.append(f"🔹 {keyword.capitalize()} levels mentioned")
        if findings:
            st.write("\n".join(findings))
        else:
            st.write("No specific clinical terms detected.")
else:
    st.info("Please upload a medical report to begin.")

st.caption("⚙️ Runs entirely offline using local pretrained models and OCR.")
