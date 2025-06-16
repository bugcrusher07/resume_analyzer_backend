from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from typing import Dict, List, Tuple
import os
from pathlib import Path
import shutil
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re
import docx2txt
import numpy as np
from collections import Counter
import time

nlp = spacy.load("en_core_web_sm")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

TECH_SKILLS = {
    'programming': ['python', 'javascript', 'java', 'c++', 'ruby', 'php', 'swift', 'kotlin', 'typescript'],
    'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node.js', 'laravel'],
    'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server'],
    'tools': ['git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'jira'],
    'concepts': ['agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'rest api', 'graphql']
}

STRONG_ACTION_VERBS = [
    'developed', 'implemented', 'created', 'managed', 'led', 'improved', 'increased', 'decreased',
    'optimized', 'designed', 'architected', 'engineered', 'delivered', 'achieved', 'transformed',
    'streamlined', 'automated', 'integrated', 'deployed', 'maintained', 'enhanced', 'resolved'
]

INDUSTRY_KEYWORDS = {
    'software_development': ['software', 'development', 'programming', 'coding', 'application'],
    'data_science': ['data', 'analysis', 'machine learning', 'ai', 'statistics'],
    'cloud_computing': ['cloud', 'aws', 'azure', 'gcp', 'infrastructure'],
    'devops': ['devops', 'ci/cd', 'automation', 'deployment', 'infrastructure'],
    'web_development': ['web', 'frontend', 'backend', 'full-stack', 'responsive']
}

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Resume Analyzer API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]
)

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

RATE_LIMIT = "10/minute"

request_tracker = {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX file."""
    return docx2txt.process(str(file_path))

def extract_skills(text: str) -> List[str]:
    """Extract skills from text using spaCy and keyword matching."""
    doc = nlp(text.lower())

    skills = set()
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        for category, skill_list in TECH_SKILLS.items():
            if any(skill in chunk_text for skill in skill_list):
                skills.add(chunk_text)

    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT']:
            skills.add(ent.text.lower())

    return list(skills)

def analyze_sentence_structure(text: str) -> Tuple[float, List[str]]:
    """Analyze sentence structure and complexity using spaCy."""
    doc = nlp(text)
    sentences = list(doc.sents)

    avg_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0

    complex_sentences = [sent.text for sent in sentences if len(sent) > 20]

    return avg_length, complex_sentences

def analyze_action_verbs(text: str) -> Tuple[List[str], List[str]]:
    """Analyze the usage of action verbs in the text."""
    doc = nlp(text.lower())

    found_verbs = [token.text for token in doc if token.pos_ == "VERB"]

    strong_verbs = [verb for verb in found_verbs if verb in STRONG_ACTION_VERBS]

    weak_verbs = [verb for verb in found_verbs if verb not in STRONG_ACTION_VERBS]

    return strong_verbs, weak_verbs

def identify_industry_focus(text: str) -> List[Tuple[str, float]]:
    """Identify the main industry focus areas using TF-IDF and keyword matching."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])

    feature_names = vectorizer.get_feature_names_out()

    industry_scores = []
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        score = sum(tfidf_matrix[0, vectorizer.vocabulary_.get(kw, 0)]
                   for kw in keywords if kw in vectorizer.vocabulary_)
        industry_scores.append((industry, float(score)))

    return sorted(industry_scores, key=lambda x: x[1], reverse=True)

def generate_ml_based_recommendations(text: str, skills: List[str]) -> List[Dict]:
    """Generate ML-based recommendations for resume improvement."""
    recommendations = []

    avg_length, complex_sentences = analyze_sentence_structure(text)
    if avg_length > 15:
        recommendations.append({
            "type": "structure",
            "severity": "medium",
            "message": "Your sentences are quite long (average {:.1f} words). Consider breaking down complex sentences into shorter, more impactful ones.".format(avg_length),
            "example": complex_sentences[0] if complex_sentences else None
        })

    strong_verbs, weak_verbs = analyze_action_verbs(text)
    if len(strong_verbs) < 5:
        recommendations.append({
            "type": "content",
            "severity": "high",
            "message": "Use more strong action verbs to describe your achievements. Consider replacing weak verbs with more impactful ones.",
            "suggestion": f"Instead of using verbs like '{', '.join(weak_verbs[:3])}', try using strong action verbs like 'developed', 'implemented', 'optimized'."
        })

    industry_focus = identify_industry_focus(text)
    if industry_focus:
        top_industry = industry_focus[0]
        if top_industry[1] < 0.1:
            recommendations.append({
                "type": "content",
                "severity": "medium",
                "message": "Your resume lacks strong industry-specific keywords. Consider adding more relevant industry terminology.",
                "suggestion": f"Add more keywords related to {top_industry[0].replace('_', ' ')} to better target your desired industry."
            })

    if len(skills) < 5:
        recommendations.append({
            "type": "skills",
            "severity": "high",
            "message": "Your resume shows limited technical skills. Consider adding more relevant technical skills.",
            "suggestion": "Add specific technical skills you've used in your projects and work experience."
        })

    if not re.search(r'\d+%|\$\d+|\d+\s*(?:million|billion)?', text):
        recommendations.append({
            "type": "content",
            "severity": "high",
            "message": "Your resume lacks quantifiable achievements. Add specific numbers and metrics to make your experience more impactful.",
            "suggestion": "Instead of 'increased sales', try 'increased sales by 25%' or 'reduced costs by $50,000'."
        })

    if not re.search(r'\b(?:bachelor|master|phd|degree)\b', text.lower()):
        recommendations.append({
            "type": "structure",
            "severity": "high",
            "message": "Your resume is missing a clear education section. Make sure to list your educational background.",
            "suggestion": "Add a dedicated education section with your degree, university, and graduation date."
        })

    return recommendations

def calculate_ats_score(text: str) -> float:
    """Calculate ATS score based on various factors."""
    score = 0.0
    max_score = 100.0

    sections = ['experience', 'education', 'skills', 'projects', 'summary']
    section_score = 0
    for section in sections:
        if re.search(rf'\b{section}\b', text.lower()):
            section_score += 1
    score += (section_score / len(sections)) * 20

    skills = extract_skills(text)
    skill_score = min(len(skills) / 10, 1.0)
    score += skill_score * 30

    experience_indicators = ['years', 'experience', 'worked', 'developed', 'implemented', 'managed']
    exp_score = sum(1 for word in experience_indicators if word in text.lower()) / len(experience_indicators)
    score += exp_score * 25

    education_indicators = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
    edu_score = sum(1 for word in education_indicators if word in text.lower()) / len(education_indicators)
    score += edu_score * 25

    return min(score, max_score)

def is_valid_file_size(file_size: int) -> bool:
    """Check if file size is within limits."""
    return file_size <= MAX_FILE_SIZE

def is_valid_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def track_request(ip: str) -> bool:
    """Track requests per IP with a time window."""
    current_time = time.time()
    window = 60  # 1 minute window

    if ip not in request_tracker:
        request_tracker[ip] = []

    # Remove old requests
    request_tracker[ip] = [t for t in request_tracker[ip] if current_time - t < window]

    # Check if too many requests
    if len(request_tracker[ip]) >= 10:  # 10 requests per minute
        return False

    # Add new request
    request_tracker[ip].append(current_time)
    return True

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    return response

@app.middleware("http")
async def validate_request(request: Request, call_next):
    """Validate request size and track requests."""
    # Get client IP
    client_ip = request.client.host

    # Check request tracking
    if not track_request(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )

    # Check content length for POST requests
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
            )

    response = await call_next(request)
    return response

@app.get("/")
@limiter.limit(RATE_LIMIT)
async def read_root(request: Request):
    return {"message": "Welcome to Resume Analyzer API"}

@app.post("/analyze-resume")
@limiter.limit(RATE_LIMIT)
async def analyze_resume(request: Request, file: UploadFile = File(...)) -> Dict:
    """
    Analyze a resume file and return insights.
    Accepts PDF and DOCX files.
    """
    # Validate file size
    if not is_valid_file_size(file.size):
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
        )

    # Validate file extension
    if not is_valid_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Please upload a PDF or DOCX file."
        )

    allowed_types = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Please upload a PDF or DOCX file."
        )

    try:
        file_extension = allowed_types[file.content_type]
        unique_filename = f"{file.filename.split('.')[0]}_{os.urandom(4).hex()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_docx(file_path)

        skills = extract_skills(text)
        ats_score = calculate_ats_score(text)
        recommendations = generate_ml_based_recommendations(text, skills)

        industry_focus = identify_industry_focus(text)
        top_industries = [ind[0].replace('_', ' ').title() for ind in industry_focus[:3] if ind[1] > 0]

        return {
            "status": "success",
            "message": "Resume analyzed successfully",
            "filename": file.filename,
            "analysis": {
                "ats_score": round(ats_score, 1),
                "skills": skills,
                "recommendations": recommendations,
                "industry_focus": top_industries,
                "summary": f"Your resume has an ATS score of {round(ats_score, 1)}/100. " +
                          f"Found {len(skills)} relevant skills. " +
                          f"Primary industry focus: {', '.join(top_industries) if top_industries else 'Not clearly defined'}. "
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )
    finally:
        if file_path.exists():
            file_path.unlink()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)