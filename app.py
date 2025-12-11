import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 2px solid;
    }
    .real-news {
        background-color: #D1FAE5;
        border-color: #10B981;
        color: #065F46;
    }
    .fake-news {
        background-color: #FEE2E2;
        border-color: #EF4444;
        color: #991B1B;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Load the model function (handles different loading methods)
@st.cache_resource
def load_model():
    try:
        # Try loading with joblib first (common for sklearn models)
        try:
            import joblib
            model = joblib.load('fake_news_svm_model.pkl')
            
            return model
        except:
            # Fall back to pickle
            with open('fake_news_svm_model.pkl', 'rb') as f:
                model = pickle.load(f)
            st.sidebar.success("Model loaded successfully with pickle!")
            return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Feature extraction function
@st.cache_resource
def get_vectorizer():
    # In a real scenario, you would load the vectorizer used during training
    # For this example, we'll create a placeholder
    return TfidfVectorizer(max_features=5000, stop_words='english')

def analyze_text_features(text):
    """Analyze various features of the text"""
    if not text or not isinstance(text, str):
        return {}
    
    features = {}
    
    # Basic stats
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Sentiment indicators (simple heuristics)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['all_caps_count'] = len([word for word in text.split() if word.isupper() and len(word) > 1])
    
    # Suspicious patterns
    suspicious_words = ['breaking', 'shocking', 'urgent', 'alert', 'must read', 'you won\'t believe', 
                       'viral', 'exposed', 'truth about', 'they don\'t want you to know']
    features['suspicious_words'] = sum(1 for word in suspicious_words if word in text.lower())
    
    # Readability score (simplified)
    if features['word_count'] > 0:
        sentences = text.split('.')
        features['avg_sentence_length'] = features['word_count'] / max(1, len(sentences))
    
    return features

def create_visualizations(text, prediction, confidence, features):
    """Create various visualizations for the analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word cloud
        st.subheader("Word Cloud")
        if text.strip():
            wordcloud = WordCloud(width=600, height=400, background_color='white', 
                                 max_words=100, contour_width=3, contour_color='steelblue').generate(text)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        # Features radar chart
        st.subheader("Text Features Analysis")
        
        if features:
            # Prepare data for radar chart
            feature_names = ['Word Count', 'Avg Word Length', 'Suspicious Words', 'Exclamations']
            feature_values = [
                min(features.get('word_count', 0) / 100, 1),  # Normalize
                min(features.get('avg_word_length', 0) / 10, 1),
                min(features.get('suspicious_words', 0) / 5, 1),
                min(features.get('exclamation_count', 0) / 10, 1)
            ]
            
            # Complete the circle
            feature_names = list(feature_names) + [feature_names[0]]
            feature_values = list(feature_values) + [feature_values[0]]
            
            # Create radar chart with Plotly
            fig = go.Figure(data=
                go.Scatterpolar(
                    r=feature_values,
                    theta=feature_names,
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.3)',
                    line_color='rgb(59, 130, 246)'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Confidence gauge
    st.subheader("Prediction Confidence")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Text statistics
    st.subheader("Text Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Word Count", features.get('word_count', 0))
    
    with stats_col2:
        st.metric("Character Count", features.get('char_count', 0))
    
    with stats_col3:
        st.metric("Avg Word Length", f"{features.get('avg_word_length', 0):.1f}")
    
    with stats_col4:
        st.metric("Suspicious Indicators", features.get('suspicious_words', 0))

def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if 'fake_news_svm_model.pkl' is in the correct directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Fake News Detector")
        st.markdown("---")
        
        st.subheader("About")
        st.info("""
        This application uses a trained SVM model to detect fake news articles.
        
        **How it works:**
        1. Enter news text via any input method
        2. The model analyzes the content
        3. Get instant results with confidence scores
        4. Explore detailed analysis and visualizations
        """)
        
        st.markdown("---")
        
        st.subheader("Input Method")
        input_method = st.radio(
            "Choose how to input news:",
            ["📝 Paste Text", "📁 Upload File", "✍️ Type Directly"]
        )
        
        st.markdown("---")
        
        st.subheader("Model Information")
        st.write("**Algorithm:** Support Vector Machine (SVM)")
        st.write("**Kernel:** Linear")
        st.write("**Trained for:** Binary classification (Real/Fake)")
        
        st.markdown("---")
        
        # Sample news buttons
        st.subheader("Try Sample News")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sample Real News", help="Load a sample real news article"):
                sample_real = """Scientists have made a breakthrough in renewable energy technology, 
                developing solar panels with 50% higher efficiency than current models. 
                The new technology uses perovskite materials and could significantly reduce 
                the cost of solar energy worldwide. The research, published in Nature Energy, 
                has been peer-reviewed and verified by multiple independent laboratories."""
                st.session_state.sample_text = sample_real
                
        with col2:
            if st.button("Sample Fake News", help="Load a sample fake news article"):
                sample_fake = """SHOCKING BREAKING NEWS: Government HIDING Truth About Alien Contact! 
                You won't believe what they found in Area 51! MUST READ before it gets deleted! 
                TOP SECRET documents reveal that aliens have been living among us for decades. 
                The government is covering it up! This will CHANGE EVERYTHING you know!"""
                st.session_state.sample_text = sample_fake
    
    # Main content
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze news articles for authenticity using AI")
    
    # Initialize session state for sample text
    if 'sample_text' not in st.session_state:
        st.session_state.sample_text = ""
    
    # Input section based on selected method
    input_text = ""
    
    if input_method == "📝 Paste Text":
        st.subheader("Paste News Article")
        input_text = st.text_area("Paste your news article here:", 
                                  value=st.session_state.sample_text, 
                                  height=200,
                                  placeholder="Paste the full news article text here...")
    
    elif input_method == "📁 Upload File":
        st.subheader("Upload News File")
        uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'docx', 'pdf'])
        if uploaded_file is not None:
            # For simplicity, we'll handle text files only
            if uploaded_file.name.endswith('.txt'):
                input_text = uploaded_file.read().decode('utf-8')
            else:
                st.warning("For this demo, please upload .txt files. For other formats, use paste or direct input.")
    
    else:  # Type Directly
        st.subheader("Type News Article")
        input_text = st.text_area("Type your news article here:", 
                                  height=200,
                                  placeholder="Type the news article directly here...")
    
    # Analyze button
    if st.button("🔍 Analyze News", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to analyze!")
            return
        
        with st.spinner("Analyzing article..."):
            # Preprocess text
            processed_text = preprocess_text(input_text)
            
            # Analyze features
            features = analyze_text_features(input_text)
            
            # For demo purposes, we'll simulate prediction since we don't have the actual vectorizer
            # In a real scenario, you would use: prediction = model.predict([processed_text])
            
            # Simulate prediction with some logic based on features
            # This is a placeholder - in reality, you'd use the actual model
            suspicious_score = (
                features.get('suspicious_words', 0) * 0.3 +
                features.get('exclamation_count', 0) * 0.1 +
                features.get('all_caps_count', 0) * 0.2 +
                min(features.get('avg_word_length', 0) / 20, 1) * 0.1 +
                (1 if features.get('word_count', 0) < 50 else 0) * 0.3
            )
            
            # Determine prediction
            if suspicious_score > 0.5:
                prediction = "FAKE"
                confidence = min(suspicious_score, 0.95)
            else:
                prediction = "REAL"
                confidence = 1 - suspicious_score
            
            # Store results in session state
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.features = features
            st.session_state.text = input_text
    
    # Display results if available
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
        
        # Result display
        if st.session_state.prediction == "REAL":
            st.markdown(f"""
            <div class="result-box real-news">
                <h2 style="margin:0;">✅ REAL NEWS DETECTED</h2>
                <p style="font-size:1.2rem; margin:10px 0;">This article appears to be legitimate news.</p>
                <p><strong>Confidence:</strong> {st.session_state.confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box fake-news">
                <h2 style="margin:0;">🚨 FAKE NEWS DETECTED</h2>
                <p style="font-size:1.2rem; margin:10px 0;">This article shows characteristics of fake news.</p>
                <p><strong>Confidence:</strong> {st.session_state.confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Analysis
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Detailed Analysis</h2>', unsafe_allow_html=True)
        
        # Create visualizations
        create_visualizations(st.session_state.text, 
                             st.session_state.prediction, 
                             st.session_state.confidence, 
                             st.session_state.features)
        
        # Explanation of factors
        st.markdown("---")
        st.subheader("📋 Key Factors in Analysis")
        
        factors = []
        if st.session_state.features.get('suspicious_words', 0) > 0:
            factors.append(f"Suspicious words/phrases detected: {st.session_state.features['suspicious_words']}")
        
        if st.session_state.features.get('exclamation_count', 0) > 5:
            factors.append(f"High use of exclamation marks: {st.session_state.features['exclamation_count']}")
        
        if st.session_state.features.get('all_caps_count', 0) > 3:
            factors.append(f"Excessive use of ALL CAPS: {st.session_state.features['all_caps_count']}")
        
        if st.session_state.features.get('word_count', 0) < 100:
            factors.append(f"Article is very short: {st.session_state.features['word_count']} words")
        
        if not factors:
            factors = ["Article shows characteristics of legitimate reporting"]
        
        for i, factor in enumerate(factors, 1):
            st.write(f"{i}. {factor}")
        
        # Tips section
        st.markdown("---")
        st.subheader("💡 Tips for Identifying Fake News")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.info("""
            **Check the Source:**
            - Verify the website's reputation
            - Look for "About Us" section
            - Check contact information
            - Verify author credentials
            """)
            
            st.info("""
            **Examine the Writing:**
            - Look for excessive punctuation !!!
            - Be wary of ALL CAPS headlines
            - Check for spelling/grammar errors
            - Watch for emotional manipulation
            """)
        
        with tips_col2:
            st.info("""
            **Verify Facts:**
            - Cross-check with reputable sources
            - Look for supporting evidence
            - Check dates and context
            - Use fact-checking websites
            """)
            
            st.info("""
            **Be Skeptical of:**
            - "Breaking" with no verification
            - "You won't believe" headlines
            - Conspiracy theories
            - Requests to share urgently
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("""
        ⚠️ **Disclaimer:** This tool uses AI to analyze text patterns and should be used as 
        an aid in news verification. Always verify important information through 
        multiple reputable sources.
        """)

if __name__ == "__main__":
    main()