# CodeAlpha — FAQ Chatbot 🤖

> **Task 2** of the CodeAlpha Artificial Intelligence Internship

A smart FAQ chatbot powered by **NLP (NLTK)**, **TF-IDF vectorization**, and **Cosine Similarity** — with a beautiful dark-themed web UI built using Flask.

---

## 📌 Features

- ✅ 100+ real-world FAQs covering internship, tasks, GitHub, LinkedIn, and technical queries
- ✅ Full NLP pipeline: Tokenization → Stopword Removal → Lemmatization
- ✅ TF-IDF vectorization using Scikit-learn
- ✅ Cosine Similarity matching to find best FAQ answer
- ✅ Confidence score display (High / Medium / Low)
- ✅ NLP Inspector panel — shows processing steps live
- ✅ Beautiful dark-themed responsive UI
- ✅ FAQ browser sidebar
- ✅ Typing animation effect

---

## 🧠 How It Works

```
User Question
      ↓
Preprocessing (NLTK)
  → Lowercase → Remove Punctuation → Tokenize → Remove Stopwords → Lemmatize
      ↓
TF-IDF Vectorization (Scikit-learn)
      ↓
Cosine Similarity against all FAQ questions
      ↓
Return Best Match + Confidence Score
```

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_ChatbotForFAQs.git
cd CodeAlpha_ChatbotForFAQs
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
CodeAlpha_ChatbotForFAQs/
│
├── app.py              → Flask backend (routes)
├── chatbot.py          → NLP engine (core logic)
├── faqs.py             → FAQ dataset (20 Q&A pairs)
├── requirements.txt    → Python dependencies
├── templates/
│   └── index.html      → Frontend UI
└── README.md
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core language |
| Flask | Web framework / backend |
| NLTK | NLP preprocessing |
| Scikit-learn | TF-IDF + Cosine Similarity |
| NumPy | Vector operations |
| HTML / CSS / JS | Frontend UI |

---

## 👤 Author

- **Intern at CodeAlpha**
- LinkedIn: [https://www.linkedin.com/in/zainab-ilyas-559109349/]
- GitHub: [https://github.com/zainabilyas205-source/CodeAlpha-FAQs-Chatbot]

---

## 📜 License

This project is part of the CodeAlpha Internship Program.  
Visit: [www.codealpha.tech](https://www.codealpha.tech)
