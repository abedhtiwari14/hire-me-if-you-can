# 🧠 Hire Me If You Can – ATS Resume Analyzer

An intelligent tool that compares your resume against a job description using NLP-based keyword extraction, TF-IDF similarity scoring, and interactive Streamlit UI.
It provides a similarity score, highlights missing skills, and helps users optimize resumes for Applicant Tracking Systems (ATS).

---

## 🚀 Features
- Upload resume (PDF/Text) and paste job description
- Extract keywords using TF-IDF vectorization
- Compute similarity score (cosine similarity)
- Display visual feedback for matching & missing keywords
- GPT-based improvement suggestions *(coming soon)*

---

## 🛠️ Tech Stack
- **Python 3.11+**
- **Streamlit** for UI
- **Scikit-learn** for NLP (TF-IDF & Cosine Similarity)
- **PyPDF2 / pdfminer** for text extraction
- **Pandas**, **Regex**, **Matplotlib** for analytics

---

## ⚙️ Installation
```bash
git clone https://github.com/abedhtiwari14/hire-me-if-you-can.git
cd hire-me-if-you-can
pip install -r requirements.txt
streamlit run app.py