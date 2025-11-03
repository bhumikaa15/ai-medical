# ai-medical
Medical AI Assistant(Streamlit), extracts and summarizes text from uploaded medical reports (PDF, image, or text files). It uses PyMuPDF for PDFs, Tesseract OCR for image text extraction, and a BART summarization model for generating concise interpretations,also highlights key medical findings.
# Libraries and Tools Used
Streamlit – For building the interactive web interface.
PyMuPDF (fitz) – For extracting text content from PDF files.
PIL (Pillow) – For handling image uploads and preprocessing.
OpenCV (cv2) – For image enhancement and grayscale conversion before OCR.
NumPy – For efficient numerical and image array handling.
Pytesseract – For Optical Character Recognition (OCR) to extract text from images.
Transformers (Hugging Face) – For natural language summarization using the pretrained BART model (facebook/bart-large-cnn).
re (Regular Expressions) – For cleaning and normalizing extracted text.Tech Stack
# Tech Stack
Frontend UI-Streamlit
Document Processing-PyMuPDF (fitz), re
Image Processing-OpenCV, Pillow, NumPy
OCR-Tesseract OCR (via pytesseract)
NLP / Summarization-Hugging Face Transformers (facebook/bart-large-cnn)
Language-Python 3.11
# GIT
# Installation & Setup
git clone https://github.com/<your-username>/medical-ai-assistant.git
cd medical-ai-assistant
# Create a Virtual Environment(optional)
conda create -n medai python=3.11
conda activate medai
# Install Dependencies
pip install -r requirements.txt
Install Tesseract OCR
Download from UB Mannheim Builds
Default path:
C:\Program Files\Tesseract-OCR\tesseract.exe
# Run
streamlit run medical_reader_app.py







