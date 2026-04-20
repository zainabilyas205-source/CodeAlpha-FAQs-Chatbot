import nltk
import numpy as np
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import faqs

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ─── Synonym Map ─────────────────────────────────────────────────────────────
SYNONYMS = {
    "deadline"        : ["deadline", "last date", "due date", "submission date",
                         "last day", "final date", "end date", "submit by",
                         "time limit", "expiry", "closing date", "cutoff"],
    "duration"        : ["duration", "how long", "length", "time period",
                         "internship period", "weeks", "days", "months", "long"],
    "submit"          : ["submit", "submission", "upload", "send", "complete",
                         "finish", "turn in", "hand in", "submitting"],
    "certificate"     : ["certificate", "certification", "cert", "completion letter",
                         "internship letter"],
    "stipend"         : ["stipend", "paid", "payment", "salary", "money",
                         "compensation", "fee", "cost"],
    "extend"          : ["extend", "extension", "more time", "extra time",
                         "late submission", "delay"],
    "task"            : ["task", "project", "assignment", "work", "activity"],
    "github"          : ["github", "repository", "repo", "source code", "code upload"],
    "linkedin"        : ["linkedin", "post", "social media", "share", "video"],
    "offer letter"    : ["offer letter", "acceptance", "onboarding", "joining letter"],
    "apply"           : ["apply", "application", "register", "sign up", "enroll"],
}

def expand_synonyms(text):
    """Replace words with their canonical synonym group so TF-IDF matches better."""
    words = text.lower().split()
    expanded = list(words)
    for word in words:
        for key, synonyms in SYNONYMS.items():
            if word in synonyms:
                # add ALL synonyms so any variant matches
                expanded.extend(synonyms)
                break
    return " ".join(expanded)

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(text):
    text   = text.lower()
    text   = text.translate(str.maketrans('', '', string.punctuation))
    text   = expand_synonyms(text)          # <-- synonym expansion added here
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

faq_questions  = [f['question'] for f in faqs]
faq_answers    = [f['answer']   for f in faqs]
processed_faqs = [preprocess(q) for q in faq_questions]

vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_faqs)

# ─── Main Response Function ───────────────────────────────────────────────────
def get_response(user_input):
    processed_input = preprocess(user_input)
    user_vec        = vectorizer.transform([processed_input])
    similarities    = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx        = np.argmax(similarities)
    best_score      = similarities[best_idx]

    if best_score < 0.15:
        return {
            "answer"           : "Sorry, I could not find a matching answer. Please visit www.codealpha.tech or contact CodeAlpha support directly.",
            "matched_question" : None,
            "confidence"       : round(float(best_score) * 100, 1),
            "confidence_level" : "low"
        }
    elif best_score < 0.25:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    return {
        "answer"           : faq_answers[best_idx],
        "matched_question" : faq_questions[best_idx],
        "confidence"       : round(float(best_score) * 100, 1),
        "confidence_level" : confidence_level
    }

# ─── NLP Details (for UI display) ────────────────────────────────────────────
def get_nlp_details(user_input):
    raw_tokens = word_tokenize(
        user_input.lower().translate(str.maketrans('', '', string.punctuation))
    )
    no_stop    = [t for t in raw_tokens if t not in stop_words]
    lemmatized = [lemmatizer.lemmatize(t) for t in no_stop]
    return {
        "raw_tokens"   : raw_tokens,
        "no_stopwords" : no_stop,
        "lemmatized"   : lemmatized
    }

# ─── Greetings ────────────────────────────────────────────────────────────────
GREETINGS_INPUT = {
    'hi', 'hello', 'hey', 'hii', 'helo', 'hiiii', 'hiii',
    'good morning', 'good afternoon', 'good evening',
    'howdy', 'sup', 'whats up', "what's up", 'greetings',
    'yo', 'hiya', 'hai'
}

GREETINGS_RESPONSES = [
    "Hello! 👋 Welcome to CodeAlpha FAQ Assistant! How can I help you today?",
    "Hi there! 😊 I am your CodeAlpha Intern Assistant. Ask me anything about your internship!",
    "Hey! 👋 Great to see you! What would you like to know about CodeAlpha?",
    "Hello! Welcome aboard! 🚀 I am here to help you with all your CodeAlpha internship questions!",
    "Greetings! 😊 I am your CodeAlpha FAQ bot. What can I help you with today?",
    "Hey! 👋 So glad you are here! Ask me anything about tasks, GitHub, LinkedIn or submission!"
]

def is_greeting(text):
    cleaned = text.lower().strip().translate(
        str.maketrans('', '', string.punctuation)
    )
    return cleaned in GREETINGS_INPUT