# ============================================================================
# PART 2: STREAMLIT DEPLOYMENT APP
# ============================================================================
# Save as: app.py
# Run with: streamlit run app.py

"""
Movie Review Sentiment Analysis - Streamlit App
Predicts sentiment (positive/negative) with supporting evidence
"""

import streamlit as st
import torch
import torch.nn as nn
import pickle
import json
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Set page config
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .positive-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .negative-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .evidence-item {
        background-color: #f8f9fa;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DEFINITIONS (Same as training)
# ============================================================================

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words -= {'not', 'no', 'nor', 'neither', 'never'}
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        return ' '.join(text.split())
    
    def lemmatize_text(self, text):
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words or word in ['not', 'no', 'never']]
        return ' '.join(words)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.lemmatize_text(text)
        return text

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=2, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden)
        x = self.relu(self.fc1(dropped))
        x = self.dropout(x)
        output = self.fc2(x)
        return output

# ============================================================================
# EVIDENCE EXTRACTION
# ============================================================================

class EvidenceExtractor:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                              'love', 'best', 'perfect', 'brilliant', 'outstanding',
                              'superb', 'incredible', 'awesome', 'beautiful', 'masterpiece']
        self.negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 
                              'hate', 'poor', 'disappointing', 'waste', 'boring',
                              'stupid', 'dull', 'mediocre', 'lame', 'pathetic']
        
    def extract_sentences(self, text, sentiment, top_k=3):
        """Extract sentences that support the predicted sentiment"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return [text]
        
        scored_sentences = []
        for sent in sentences:
            score = self.vader.polarity_scores(sent)['compound']
            
            if sentiment == 'positive' and score > 0:
                scored_sentences.append((sent, score))
            elif sentiment == 'negative' and score < 0:
                scored_sentences.append((sent, abs(score)))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        evidence = [sent for sent, _ in scored_sentences[:top_k]]
        return evidence if evidence else sentences[:top_k]
    
    def extract_keywords(self, text, sentiment):
        """Extract sentiment-bearing keywords"""
        words = text.lower().split()
        
        if sentiment == 'positive':
            found = [w for w in words if w in self.positive_words]
        else:
            found = [w for w in words if w in self.negative_words]
        
        return list(set(found))
    
    def get_sentiment_scores(self, text):
        """Get detailed VADER scores"""
        return self.vader.polarity_scores(text)

# ============================================================================
# SIMILAR REVIEWS FINDER
# ============================================================================

class SimilarReviewsFinder:
    def __init__(self, reviews_data):
        self.reviews_data = reviews_data
        
    def find_similar(self, sentiment, top_k=3):
        """Find similar reviews with same sentiment"""
        similar = [r for r in self.reviews_data if r['sentiment'] == sentiment]
        return np.random.choice(similar, min(top_k, len(similar)), replace=False).tolist()

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and data"""
    try:
        # Load preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load PyTorch model
        checkpoint = torch.load('sentiment_bilstm.pth', map_location=torch.device('cpu'))
        vocab = checkpoint['vocab']
        label_classes = checkpoint['label_encoder']
        
        model = BiLSTMClassifier(len(vocab), num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load sklearn model
        with open('sentiment_lr.pkl', 'rb') as f:
            lr_data = pickle.load(f)
            lr_model = lr_data['model']
            tfidf = lr_data['tfidf']
        
        # Load sample reviews
        with open('sample_reviews.json', 'r') as f:
            sample_reviews = json.load(f)
        
        return preprocessor, model, vocab, label_classes, lr_model, tfidf, sample_reviews
    
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please ensure all model files are in the same directory.")
        st.error(f"Missing file: {e.filename}")
        st.stop()

# Load models

def load_models():
    """Load models from ZIP archive"""
    import zipfile
    
    # Extract ZIP if needed
    if not os.path.exists('sentiment_bilstm.pth'):
        st.info("📦 Extracting models...")
        with zipfile.ZipFile('models.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        st.success("✅ Models extracted!")
    
    # Load models normally
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    checkpoint = torch.load('sentiment_bilstm.pth', map_location=torch.device('cpu'))
    # ... rest of loading code
preprocessor, bilstm_model, vocab, label_classes, lr_model, tfidf, sample_reviews = load_models()
evidence_extractor = EvidenceExtractor()
similar_finder = SimilarReviewsFinder(sample_reviews)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_bilstm(text, preprocessed_text):
    """Predict using BiLSTM model"""
    words = preprocessed_text.split()
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    if not indices:
        indices = [vocab['<UNK>']]
    
    input_tensor = torch.tensor([indices], dtype=torch.long)
    
    with torch.no_grad():
        output = bilstm_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    sentiment = label_classes[predicted_class]
    return sentiment, confidence

def predict_lr(preprocessed_text):
    """Predict using Logistic Regression model"""
    X = tfidf.transform([preprocessed_text])
    prediction = lr_model.predict(X)[0]
    probabilities = lr_model.predict_proba(X)[0]
    confidence = probabilities[prediction]
    sentiment = label_classes[prediction]
    return sentiment, confidence

def ensemble_predict(text):
    """Ensemble prediction from both models"""
    preprocessed = preprocessor.preprocess(text)
    
    # BiLSTM prediction
    bilstm_sent, bilstm_conf = predict_bilstm(text, preprocessed)
    
    # LR prediction
    lr_sent, lr_conf = predict_lr(preprocessed)
    
    # Ensemble (weighted average)
    if bilstm_sent == lr_sent:
        final_sentiment = bilstm_sent
        final_confidence = (bilstm_conf * 0.6 + lr_conf * 0.4)
    else:
        # Take the one with higher confidence
        if bilstm_conf > lr_conf:
            final_sentiment = bilstm_sent
            final_confidence = bilstm_conf * 0.9
        else:
            final_sentiment = lr_sent
            final_confidence = lr_conf * 0.9
    
    return final_sentiment, final_confidence, bilstm_sent, bilstm_conf, lr_sent, lr_conf

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("### Analyze movie reviews with AI-powered sentiment detection and evidence extraction")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_choice = st.radio(
        "Select Model:",
        ["Ensemble (Recommended)", "BiLSTM Only", "Logistic Regression Only"]
    )
    
    show_evidence = st.checkbox("Show Evidence", value=True)
    show_keywords = st.checkbox("Show Keywords", value=True)
    show_similar = st.checkbox("Show Similar Reviews", value=True)
    show_scores = st.checkbox("Show Detailed Scores", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("**BiLSTM**: Deep learning model with bidirectional LSTM layers")
    st.info("**Logistic Regression**: Traditional ML with TF-IDF features")
    st.info("**Ensemble**: Combines both models for best accuracy")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Review")
    review_text = st.text_area(
        "Type or paste a movie review:",
        height=200,
        placeholder="Example: This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout. Definitely recommend watching it!"
    )
    
    # Sample reviews
    if st.button("🎲 Try a Random Sample"):
        sample = np.random.choice(sample_reviews)
        review_text = sample['review']
        st.rerun()

with col2:
    st.subheader("🚀 Quick Actions")
    
    analyze_button = st.button("🔍 Analyze Review", type="primary")
    
    if st.button("🗑️ Clear"):
        review_text = ""
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Write at least 2-3 sentences")
    st.markdown("- Include specific opinions")
    st.markdown("- Mention movie aspects")

# Analysis
if analyze_button and review_text:
    with st.spinner("🤖 Analyzing review..."):
        
        # Get predictions
        if model_choice == "Ensemble (Recommended)":
            sentiment, confidence, bilstm_sent, bilstm_conf, lr_sent, lr_conf = ensemble_predict(review_text)
        elif model_choice == "BiLSTM Only":
            preprocessed = preprocessor.preprocess(review_text)
            sentiment, confidence = predict_bilstm(review_text, preprocessed)
            bilstm_sent, bilstm_conf = sentiment, confidence
            lr_sent, lr_conf = None, None
        else:  # LR Only
            preprocessed = preprocessor.preprocess(review_text)
            sentiment, confidence = predict_lr(preprocessed)
            bilstm_sent, bilstm_conf = None, None
            lr_sent, lr_conf = sentiment, confidence
        
        # Display result
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Result box
        if sentiment == 'positive':
            st.markdown(f"""
                <div class="positive-box">
                    <h2 style="color: #28a745; margin: 0;">✅ POSITIVE REVIEW</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                        Confidence: <strong>{confidence*100:.1f}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="negative-box">
                    <h2 style="color: #dc3545; margin: 0;">❌ NEGATIVE REVIEW</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                        Confidence: <strong>{confidence*100:.1f}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Level"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#28a745" if sentiment == 'positive' else "#dc3545"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "darkgray"}
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
        
        # Model comparison
        if model_choice == "Ensemble (Recommended)":
            col1, col2 = st.columns(2)
            with col1:
                st.metric("BiLSTM Prediction", bilstm_sent.upper(), f"{bilstm_conf*100:.1f}%")
            with col2:
                st.metric("Logistic Regression", lr_sent.upper(), f"{lr_conf*100:.1f}%")
        
        # Evidence Section
        if show_evidence:
            st.markdown("---")
            st.subheader("🔍 Supporting Evidence")
            
            evidence_sentences = evidence_extractor.extract_sentences(review_text, sentiment)
            
            if evidence_sentences:
                for i, sentence in enumerate(evidence_sentences, 1):
                    st.markdown(f"""
                        <div class="evidence-item">
                            <strong>Evidence {i}:</strong> {sentence}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific evidence sentences found. The overall tone supports the prediction.")
        
        # Keywords
        if show_keywords:
            st.markdown("---")
            st.subheader("🔑 Key Sentiment Words")
            
            keywords = evidence_extractor.extract_keywords(review_text, sentiment)
            
            if keywords:
                # Create word frequency for visualization
                word_freq = Counter(keywords)
                
                cols = st.columns(min(len(keywords), 5))
                for i, (word, freq) in enumerate(word_freq.most_common(5)):
                    with cols[i]:
                        st.markdown(f"""
                            <div style="background-color: {'#d4edda' if sentiment == 'positive' else '#f8d7da'}; 
                                        padding: 0.5rem; border-radius: 0.5rem; text-align: center;">
                                <strong>{word.upper()}</strong>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info(f"No explicit {sentiment} keywords detected. Sentiment inferred from context.")
        
        # Detailed scores
        if show_scores:
            st.markdown("---")
            st.subheader("📈 Detailed Sentiment Scores")
            
            scores = evidence_extractor.get_sentiment_scores(review_text)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Positive", f"{scores['pos']:.3f}")
            col2.metric("Negative", f"{scores['neg']:.3f}")
            col3.metric("Neutral", f"{scores['neu']:.3f}")
            col4.metric("Compound", f"{scores['compound']:.3f}")
            
            # Score distribution
            fig = px.bar(
                x=['Positive', 'Negative', 'Neutral'],
                y=[scores['pos'], scores['neg'], scores['neu']],
                labels={'x': 'Sentiment', 'y': 'Score'},
                title='Sentiment Component Scores'
            )
            fig.update_traces(marker_color=['#28a745', '#dc3545', '#6c757d'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Similar reviews
        if show_similar:
            st.markdown("---")
            st.subheader("📚 Similar Reviews from Database")
            
            similar_reviews = similar_finder.find_similar(sentiment, top_k=3)
            
            for i, review in enumerate(similar_reviews, 1):
                with st.expander(f"Similar Review {i}"):
                    st.write(review['review'][:300] + "..." if len(review['review']) > 300 else review['review'])
                    st.caption(f"Sentiment: {review['sentiment'].upper()}")

elif analyze_button and not review_text:
    st.warning("⚠️ Please enter a review to analyze!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d;">
        <p>Built with Streamlit | Powered by BiLSTM & Traditional ML</p>
        <p>🎬 Movie Review Sentiment Analysis System</p>
    </div>
""", unsafe_allow_html=True)
