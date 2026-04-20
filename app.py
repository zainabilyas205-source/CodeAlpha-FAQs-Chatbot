import random
from flask import Flask, render_template, request, jsonify
from chatbot import (
    get_response,
    get_nlp_details,
    faq_questions,
    faq_answers,
    is_greeting,
    GREETINGS_RESPONSES
)
from faqs import faqs

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data         = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    if is_greeting(user_message):
        return jsonify({
            'answer'           : random.choice(GREETINGS_RESPONSES),
            'matched_question' : None,
            'confidence'       : 100.0,
            'confidence_level' : 'greeting',
            'nlp'              : get_nlp_details(user_message)
        })

    response    = get_response(user_message)
    nlp_details = get_nlp_details(user_message)

    return jsonify({
        'answer'           : response['answer'],
        'matched_question' : response['matched_question'],
        'confidence'       : response['confidence'],
        'confidence_level' : response['confidence_level'],
        'nlp'              : nlp_details
    })

@app.route('/faqs', methods=['GET'])
def get_faqs():
    return jsonify(faqs)

if __name__ == '__main__':
    print("=" * 50)
    print("  CodeAlpha FAQ Chatbot is running!")
    print("  Open browser: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)