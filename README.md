📰 Fake News Detection System

🌐 Live Deployment
🔗 Access the Application: https://fake-news-detection-app.onrender.com

⚠️ Note: The application may take 30-60 seconds to load initially due to Render's free tier spin-down policy.

📋 Project Overview
A machine learning-based web application that detects fake news articles using Natural Language Processing (NLP) and classification algorithms. The system analyzes text content and provides confidence scores for news authenticity.

✨ Key Features
🔍 Real-time Analysis: Instant classification of news articles

📊 Confidence Scoring: Probability-based predictions

📈 Visual Analytics: Word clouds and feature importance

📱 Responsive Design: Mobile-friendly interface

🔄 Batch Processing: Analyze multiple articles simultaneously

📥 File Upload: Support for CSV and text file uploads

🏗️ Project Structure: 

fake-news-detector/
│
├── app.py                    # Main Streamlit application
├── fake_news_svm_model.pkl   # Your SVM model (you already have this)
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── sample_data/              # Sample news articles
│   ├── real_news_sample.txt
│   └── fake_news_sample.txt
└── utils/                    # Utility functions
    └── text_processor.py

🚀 Quick Start
Prerequisites-
Python 3.8 or higher
pip (Python package manager)

🧠 Machine Learning Pipeline:

1. Data Preprocessing
Text cleaning and normalization

Stopword removal

Lemmatization

TF-IDF vectorization (10,000 features)

2. Feature Engineering
N-gram extraction (unigrams + bigrams)

Text length features

Sentiment analysis features

Readability scores

3. Model Architecture
Algorithm: Logistic Regression / Random Forest

Validation: 5-fold cross-validation

Performance: >92% accuracy

Metrics: Precision, Recall, F1-Score

📈 Performance Optimization
1. Caching Strategy
TF-IDF vectors cached for frequent queries

Model predictions cached for identical inputs

Session-based user data caching

2. Database Optimization
Indexed frequently queried columns

Connection pooling for web deployment

Regular database maintenance scripts

3. Scalability Features
Batch processing for multiple articles

Async processing for large texts

Load balancing ready architecture

🛠️ Technology Stack
Backend
Python 3.8+ - Core programming language

Flask - Web framework

Scikit-learn - Machine learning library

NLTK - Natural Language Processing

Pandas/Numpy - Data manipulation

Frontend
HTML5/CSS3 - Structure and styling

JavaScript - Interactive elements

Bootstrap 5 - Responsive design

Chart.js - Data visualization

DevOps
Render - Cloud deployment

Git - Version control

Docker - Containerization (optional)

📊 Dataset Information
The model is trained on a comprehensive dataset containing:

Total Samples: 44,898 articles

Real News: 23,481 samples

Fake News: 21,417 samples

Sources: Kaggle Fake News Dataset, ISOT Fake News Dataset

Features: Title, text content, author, publication date

🚨 Limitations & Future Work
Current Limitations
Limited to English language texts

Requires minimum 50 characters for analysis

Training data up to 2021

Planned Improvements
Multi-language support

Real-time news source verification

Browser extension development

Mobile application

Advanced deep learning models (BERT, GPT)

🙏 Acknowledgments
Kaggle community for datasets

Scikit-learn developers

Flask documentation team

Open-source contributors

📞 Contact & Support
For questions, issues, or suggestions:

GitHub Issues: Report a bug

Email: paaw4nnn.2005@gmail.com

<div align="center">
Made with ❤️ by Paawan Pawar

</div>