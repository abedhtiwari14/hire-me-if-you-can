import streamlit as st
import docx
from pdfminer.high_level import extract_text
import spacy
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # loads the .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


load_dotenv()

nlp = spacy.load("en_core_web_sm")

SKILLS_DB = [
        'python', 'java', 'c++', 'javascript', 'react', 'nodejs', 'django', 'flask',
    'machine learning', 'deep learning', 'nlp', 'sql', 'nosql', 'postgresql',
    'mongodb', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'tensorflow',
    'pytorch', 'data analysis', 'data visualization', 'power bi', 'tableau',
    'streamlit', 'pandas', 'numpy', 'matplotlib', 'scikit-learn', 'seaborn',
    'git', 'github', 'jenkins', 'agile', 'scrum', 'unit testing', 'oop',
    'data structures', 'algorithms', 'problem solving', 'communication',
    'teamwork', 'collaboration'
]

st.set_page_config(page_title = "Hire Me If You Can - ATS Analyzer", layout = "centered")
st.title("Hire Me If You Can - Resume ATS Analyzer")

resume_file = st.file_uploader("Upload Your Resume (.pdf or .docx)", type = ['pdf', 'docx'])
jd_text_input = st.text_area("Paste the Job Description Here!!!")

def extract_text_from_resume (uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        return extract_text(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

def extract_skills(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    found_skills = set([skill for skill in SKILLS_DB if skill in tokens])
    return found_skills

def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Returns cosine similarity between two texts using TF-IDF features.
    Range: 0.0 (no similarity) to 1.0 (identical in TF-IDF space).
    """
    vect = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1, 2))
    X = vect.fit_transform([text_a or "", text_b or ""])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def generate_resume_bullets(missing_skills, jd_text_input):
    prompt = f"""
    You are an expert resume coach. A job Description is given below, followed by some missing skills from the candidate's resume.

    Job Description: {jd_text_input}

    Missing Skills: {', '.join(missing_skills)}

    Suggest 2-3 resume bullet points that the candidate could add to naturally reflect these skills (assume they have experience in these).
    Keep the tone professional and bullet-style.
    """

    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
)
        reply = response.choices[0].message.content

        return reply.strip()
    except Exception as e:
        return f" ERROR GENERATING SUGGESTIONS: {str(e)}"

if resume_file and jd_text_input:
    with st.spinner("Analyzing Your Resume!!!"):
        resume_text = extract_text_from_resume(resume_file)
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text_input)
        
        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills

        score = round((len(matched_skills) / len(jd_skills)) * 100,2) if jd_skills else 0

        # --- Clean up text and compute TF-IDF similarity ---
        def _clean(s: str) -> str:
            return " ".join((s or "").split())

        jd_clean     = _clean(jd_text_input)
        resume_clean = _clean(resume_text)

        tfidf_score = tfidf_cosine_similarity(jd_clean, resume_clean)


        c1, c2 = st.columns(2)
        with c1:
            st.success(f"Match Score (Skills): {score}%")
        with c2:
            st.info(f"Text Similarity (TF-IDF): {tfidf_score:.3f}")

        
        st.markdown("### Skills Matched:")
        st.write(", ".join(matched_skills) if matched_skills else "None")

        st.markdown("### Skills Missing From Resume:")
        st.write(", ".join(missing_skills) if missing_skills else "None")

        st.markdown("*Tip: Add missing skills in your resume if you have experience with them!!!")

        if missing_skills:
            st.markdown("### RESUME SUGGESTIONS (AI-GENERATED!!!): ")
            with st.spinner("GENERATING SUGGESTIONS WITH GPT..."):
                suggestions = generate_resume_bullets(missing_skills, jd_text_input)
                st.markdown(suggestions)
