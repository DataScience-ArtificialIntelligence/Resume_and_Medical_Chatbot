from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import markdown
import os
import sqlite3
import json
from datetime import datetime
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "medibot:latest"

GEMINI_API_KEY = "AIzaSyDVqbcDnKf-mJncF4efQMzLHboJ5iu-5V0"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

def get_db_connection():
    conn = sqlite3.connect('medical_chat.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        rouge1 REAL,
        rouge2 REAL,
        rougeL REAL,
        bleu REAL
    )
    ''')
    conn.commit()
    conn.close()

init_db()

def evaluate_response(query, response):
    """Use Gemini to evaluate the response with ROUGE and BLEU scores"""
    prompt = f"""
    Evaluate the following AI response to a medical query using ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores.
    
    User Query: {query}
    AI Response: {response}
    
    Calculate and provide only the numerical scores (between 0 and 1) in this exact format:
    ROUGE-1: [score]
    ROUGE-2: [score]
    ROUGE-L: [score]
    BLEU: [score]
    
    Do not include any other text or explanations and always give some score in way illustrated above.
    """
    
    try:
        result = gemini_model.generate_content(prompt)
        evaluation_text = result.text
        
        scores = {}
        for line in evaluation_text.strip().split('\n'):
            if ':' in line:
                metric, value = line.split(':', 1)
                metric = metric.strip()
                try:
                    value = float(value.strip())
                    scores[metric] = value
                except ValueError:
                    continue
        
        return {
            'rouge1': scores.get('ROUGE-1', None),
            'rouge2': scores.get('ROUGE-2', None),
            'rougeL': scores.get('ROUGE-L', None),
            'bleu': scores.get('BLEU', None)
        }
    except Exception as e:
        app.logger.error(f"Error evaluating response with Gemini: {str(e)}")
        return {
            'rouge1': None,
            'rouge2': None,
            'rougeL': None,
            'bleu': None
        }

@app.route('/')
def index():
    conn = get_db_connection()
    conversations = conn.execute('SELECT * FROM conversations ORDER BY timestamp DESC').fetchall()
    conn.close()
    return render_template('index.html', conversations=conversations)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    user_query = data['query']
    
    prompt = f"""You are a helpful medical assistant chatbot. You provide general medical information but always remind users that they should consult with a healthcare professional for specific medical advice or emergencies.

User query: {user_query}

Please provide a helpful, accurate response with medical information.
Format your response using Markdown for better readability."""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        generated_text = result.get('response', '')
        
        # Evaluate the response using Gemini
        evaluation = evaluate_response(user_query, generated_text)
        
        # Store in database
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO conversations (query, response, rouge1, rouge2, rougeL, bleu) VALUES (?, ?, ?, ?, ?, ?)',
            (user_query, generated_text, evaluation['rouge1'], evaluation['rouge2'], evaluation['rougeL'], evaluation['bleu'])
        )
        conn.commit()
        conn.close()
        
        return jsonify({
            'response': generated_text,
            'evaluation': evaluation,
            'status': 'success'
        })
        
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error communicating with Ollama: {str(e)}")
        return jsonify({
            'response': "I'm having trouble connecting to my knowledge base right now. Please try again later.",
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
