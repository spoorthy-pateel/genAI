import pdfplumber
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer, util

# Step 1: Extract text from the PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Clean extracted text
def clean_text(text):
    # Replace newline artifacts (n) with spaces
    text = text.replace('\n', ' ')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove any Unicode artifacts (if present)
    text = re.sub(r'u{[0-9a-fA-F]+}', '', text)
    return text.strip()

# Step 3: Remove personal details
def remove_personal_details(text):
    # Remove the entire "PERSONAL DETAILS" section if it exists
    text = re.sub(r'PERSONAL DETAILS.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    
    # Remove phone numbers (e.g., 10-digit numbers, international formats)
    text = re.sub(r'\b\d{10}\b', '', text)  # Matches 10-digit numbers
    text = re.sub(r'\+?\d{1,4}[-.\s]?(?:\(\d{1,4}\))?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)  # Matches international formats
    
    # Remove dates of birth (e.g., "Date of Birth: 31-08-2000")
    text = re.sub(r'Date of Birth:? *\d{2}-\d{2}-\d{4}', '', text, flags=re.IGNORECASE)
    
    # Remove other personal identifiers (e.g., marital status, nationality, gender)
    text = re.sub(r'\b(Marital status|Nationality|Sex):\s*\w+\b', '', text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Step 4: Extract certificates
def extract_certificates(text):
    # List of possible certifications
    certificate_keywords = [
        "AWS Certified Solutions Architect",
        "AWS Certified Cloud Practitioner",
        "Google Data Analytics Certificate",
        "Microsoft Azure Fundamentals Certification",
        "Docker Certified Associate",
        "Kubernetes Certified Administrator"
    ]
    # Normalize text for comparison
    normalized_text = text.lower()
    certificates = [cert for cert in certificate_keywords if cert.lower() in normalized_text]
    return certificates

# Step 5: Extract skills, experience sentences, and date ranges
def extract_skills_and_experience(text):
    skills_list = [
        "Python", "Machine Learning", "Data Analysis", "SQL", "Java", "Deep Learning",
        "Natural Language Processing", "TensorFlow", "PyTorch", "AWS", "Docker", "Kubernetes",
        "Spring Boot", "React", "Angular", "RESTful Services", "Microservices", "JMS", "Hibernate"
    ]
    skills = [skill for skill in skills_list if skill.lower() in text.lower()]
    experience_keywords = ["experience", "worked", "responsible", "developed", "managed", "designed", "built"]
    sentences = text.split(".")
    experience_sentences = [
        sentence.strip() for sentence in sentences if any(keyword in sentence.lower() for keyword in experience_keywords)
    ]
    date_ranges = re.findall(
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{4} - (Present|\d{4})",
        text,
        re.IGNORECASE,
    )
    date_ranges = [" - ".join(date_range) for date_range in date_ranges]
    return skills, experience_sentences, date_ranges

# Step 6: Extract number of years of experience
def extract_years_of_experience(text):
    # Correct regex pattern to match phrases like "1 year of experience"
    years_pattern = r"(?i)(\d+)\s+year(?:s)?\s+of\s+experience"
    matches = re.findall(years_pattern, text)
    if matches:
        return max(map(int, matches))  # Convert matches to integers and return the maximum value
    else:
        return 0  # Default to 0 if no match is found

# Step 7: Calculate skill match percentage
def calculate_skill_match(job_description, extracted_skills):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    skills_text = ", ".join(extracted_skills)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    skills_embedding = model.encode(skills_text, convert_to_tensor=True)
    similarity_score = util.cos_sim(job_embedding, skills_embedding).item()
    return similarity_score * 100

# Step 8: Recommend courses
def recommend_courses(employee_skills, employee_certificates, trending_courses):
    # Combine skills and certificates into a single text
    employee_text = ", ".join(employee_skills + employee_certificates)
    
    # Load the sentence-transformers model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Encode the employee's skills and certificates
    employee_embedding = model.encode(employee_text, convert_to_tensor=True)
    
    # Encode the trending courses
    course_embeddings = model.encode(trending_courses, convert_to_tensor=True)
    
    # Calculate similarity scores
    cosine_scores = util.cos_sim(employee_embedding, course_embeddings)
    
    # Rank the courses based on similarity
    top_matches = sorted(
        zip(trending_courses, cosine_scores[0].tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return the top recommendations
    return top_matches[:5]  # Top 5 recommendations

# Step 9: Process the resume
def process_resume_with_bert(pdf_path, job_description, trending_courses):
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    sanitized_text = remove_personal_details(cleaned_text)
    certificates = extract_certificates(sanitized_text)
    skills, experience_sentences, date_ranges = extract_skills_and_experience(sanitized_text)
    years_of_experience = extract_years_of_experience(sanitized_text)
    skill_match_percentage = calculate_skill_match(job_description, skills)
    recommended_courses = recommend_courses(skills, certificates, trending_courses)
    return {
        "skills": skills,
        "certificates": certificates,
        "experience_sentences": experience_sentences,
        "date_ranges": date_ranges,
        "years_of_experience": years_of_experience,
        "skill_match_percentage": skill_match_percentage,
        "recommended_courses": recommended_courses
    }

# Example usage
if __name__ == "__main__":
    pdf_path = "SpoorthyPateel_Resume.pdf"  # Replace with the path to your PDF file
    job_description = """
    We are looking for a Java Developer with experience in application development using React, AWS, and Spring Boot.
    """
    trending_courses = [
        "Advanced Machine Learning with TensorFlow",
        "Data Engineering on Google Cloud",
        "Docker and Kubernetes for DevOps",
        "Full Stack In Java Development",
        "Deep Learning Specialization by Andrew Ng",
        "AWS Certified Machine Learning Specialty",
        "Microsoft Azure Fundamentals Certification",
        "Natural Language Processing with Transformers",
        "Cybersecurity Fundamentals by IBM"
    ]
    
    extracted_info = process_resume_with_bert(pdf_path, job_description, trending_courses)
    print("Extracted Skills:", extracted_info["skills"])
    print("Extracted Certificates:", extracted_info["certificates"])
    print("Extracted Experience Sentences:", extracted_info["experience_sentences"])
    print("Extracted Date Ranges:", extracted_info["date_ranges"])
    print("Number of Years of Experience:", extracted_info["years_of_experience"])
    print(f"The skills match the job description by {extracted_info['skill_match_percentage']:.2f}%")
    print("Top Course Recommendations:")
    for course, score in extracted_info["recommended_courses"]:
        print(f"{course} (Similarity Score: {score:.2f})")
