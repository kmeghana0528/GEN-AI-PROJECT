"""
Hotel Guest Review Sentiment Analyzer - FastAPI Backend
========================================================
A production-ready API for analyzing hotel guest reviews using LSTM deep learning models.

Features:
- Sentiment Analysis (Positive/Negative/Neutral)
- Category Detection (Cleanliness, Food, Staff, Amenities, Overall)
- Batch processing support
- Detailed analytics and insights
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import json
import pickle
import re
import os
from datetime import datetime

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom model loading function to handle version compatibility
def load_keras_model(path):
    """Load Keras model with version compatibility handling"""
    try:
        # Try standard loading first
        from tensorflow.keras.models import load_model
        return load_model(path)
    except Exception as e1:
        try:
            # Try with compile=False
            from tensorflow.keras.models import load_model
            model = load_model(path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e2:
            try:
                # Try legacy Keras format
                import keras
                return keras.models.load_model(path)
            except Exception as e3:
                raise Exception(f"Could not load model: {e1}, {e2}, {e3}")

# NLTK imports
import nltk
from nltk.stem import WordNetLemmatizer

# Flag to track if NLTK is available
NLTK_AVAILABLE = False

def download_nltk_data():
    """Download NLTK data with error handling"""
    global NLTK_AVAILABLE
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        NLTK_AVAILABLE = True
        print("✅ NLTK data already available")
    except LookupError:
        print("📥 Downloading NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            NLTK_AVAILABLE = True
            print("✅ NLTK data downloaded successfully")
        except Exception as e:
            print(f"⚠️ Could not download NLTK data: {e}")
            print("⚠️ Using simple tokenization fallback")
            NLTK_AVAILABLE = False


# ============== Lifespan Event ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting Hotel Review Sentiment Analyzer API...")
    download_nltk_data()
    initialize_nlp_tools()
    load_models()
    print("✅ API ready to serve requests!")
    yield
    # Shutdown
    print("👋 Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="🏨 Hotel Guest Review Sentiment Analyzer",
    description="Analyze hotel guest reviews to automatically detect sentiments and understand recurring themes.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Model Loading ==============

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "hotel_sentiment_model")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.h5")
CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, "category_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Global variables for models
sentiment_model = None
category_model = None
tokenizer = None
config = None

# NLP tools
lemmatizer = None
stop_words = set()

def initialize_nlp_tools():
    """Initialize NLP tools after NLTK download"""
    global lemmatizer, stop_words
    
    if NLTK_AVAILABLE:
        try:
            from nltk.corpus import stopwords as nltk_stopwords
            lemmatizer = WordNetLemmatizer()
            stop_words = set(nltk_stopwords.words('english'))
            important_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                               'nowhere', 'hardly', 'barely', 'very', 'really', 'absolutely'}
            stop_words = stop_words - important_words
            print("✅ NLP tools initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize NLP tools: {e}")
    else:
        # Basic stopwords fallback
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'under', 'again', 'further', 'then', 'once',
                      'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'each', 'few',
                      'more', 'most', 'other', 'some', 'such', 'than', 'too', 'only',
                      'own', 'same', 'just', 'also', 'now', 'here', 'there', 'when',
                      'where', 'why', 'how', 'all', 'any', 'both', 'each', 'this', 'that'}
        print("⚠️ Using basic stopwords fallback")


def load_models():
    """Load trained models and tokenizer"""
    global sentiment_model, category_model, tokenizer, config
    
    try:
        # Check if model files exist
        if os.path.exists(SENTIMENT_MODEL_PATH):
            try:
                sentiment_model = load_keras_model(SENTIMENT_MODEL_PATH)
                print("✅ Sentiment model loaded!")
            except Exception as e:
                print(f"❌ Could not load sentiment model: {e}")
        else:
            print(f"⚠️ Sentiment model not found at {SENTIMENT_MODEL_PATH}")
            
        if os.path.exists(CATEGORY_MODEL_PATH):
            try:
                category_model = load_keras_model(CATEGORY_MODEL_PATH)
                print("✅ Category model loaded!")
            except Exception as e:
                print(f"❌ Could not load category model: {e}")
        else:
            print(f"⚠️ Category model not found at {CATEGORY_MODEL_PATH}")
            
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            print("✅ Tokenizer loaded!")
        else:
            print(f"⚠️ Tokenizer not found at {TOKENIZER_PATH}")
            
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            print("✅ Config loaded!")
        else:
            # Default config
            config = {
                'max_words': 5000,
                'max_len': 100,
                'sentiment_labels': {'0': 'negative', '1': 'neutral', '2': 'positive'},
                'category_labels': {'0': 'cleanliness', '1': 'food', '2': 'staff', '3': 'amenities', '4': 'overall'}
            }
            print("⚠️ Using default config")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")


# ============== Preprocessing ==============

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for model prediction"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    if NLTK_AVAILABLE:
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
    else:
        tokens = text.split()
    
    # Remove stopwords and lemmatize
    if lemmatizer and NLTK_AVAILABLE:
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                  if token not in stop_words and len(token) > 2]
    else:
        tokens = [token for token in tokens 
                  if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)


# ============== Pydantic Models ==============

class ReviewInput(BaseModel):
    """Single review input model"""
    review: str = Field(..., min_length=3, max_length=5000, description="Hotel guest review text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "review": "The room was absolutely amazing and the staff were incredibly helpful!"
            }
        }


class BatchReviewInput(BaseModel):
    """Batch review input model"""
    reviews: List[str] = Field(..., min_length=1, max_length=100, description="List of hotel guest reviews")
    
    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "The room was spotlessly clean and fresh!",
                    "Terrible food quality. Very disappointed.",
                    "Average experience, nothing special."
                ]
            }
        }


class SentimentScore(BaseModel):
    """Sentiment probability scores"""
    negative: float
    neutral: float
    positive: float


class ReviewAnalysis(BaseModel):
    """Complete review analysis result"""
    review: str
    cleaned_review: str
    sentiment: str
    sentiment_emoji: str
    sentiment_confidence: float
    sentiment_scores: SentimentScore
    category: str
    category_emoji: str
    category_confidence: float
    analyzed_at: str


class BatchAnalysisResult(BaseModel):
    """Batch analysis result with summary"""
    total_reviews: int
    analyses: List[ReviewAnalysis]
    summary: Dict[str, Any]


class HealthStatus(BaseModel):
    """API health status"""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


# ============== Helper Functions ==============

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    emojis = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emojis.get(sentiment, '❓')


def get_category_emoji(category: str) -> str:
    """Get emoji for category"""
    emojis = {
        'cleanliness': '🧹',
        'food': '🍽️',
        'staff': '👨‍💼',
        'amenities': '🏊',
        'overall': '🏨'
    }
    return emojis.get(category, '📝')


def analyze_single_review(review_text: str) -> ReviewAnalysis:
    """Analyze a single review and return detailed results"""
    
    # If models are not loaded, use TextBlob fallback
    if sentiment_model is None or tokenizer is None:
        return fallback_analysis(review_text)
    
    # Preprocess
    cleaned = preprocess_text(review_text)
    
    # Tokenize and pad
    max_len = config.get('max_len', 100)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict sentiment
    sentiment_pred = sentiment_model.predict(padded, verbose=0)
    sentiment_class = int(np.argmax(sentiment_pred[0]))
    sentiment_confidence = float(sentiment_pred[0][sentiment_class])
    
    # Get sentiment label
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_labels.get(sentiment_class, 'unknown')
    
    # Predict category if model exists
    if category_model is not None:
        category_pred = category_model.predict(padded, verbose=0)
        category_class = int(np.argmax(category_pred[0]))
        category_confidence = float(category_pred[0][category_class])
        category_labels = {0: 'cleanliness', 1: 'food', 2: 'staff', 3: 'amenities', 4: 'overall'}
        category = category_labels.get(category_class, 'overall')
    else:
        category = detect_category_keywords(review_text)
        category_confidence = 0.8
    
    return ReviewAnalysis(
        review=review_text,
        cleaned_review=cleaned,
        sentiment=sentiment,
        sentiment_emoji=get_sentiment_emoji(sentiment),
        sentiment_confidence=round(sentiment_confidence * 100, 2),
        sentiment_scores=SentimentScore(
            negative=round(float(sentiment_pred[0][0]) * 100, 2),
            neutral=round(float(sentiment_pred[0][1]) * 100, 2),
            positive=round(float(sentiment_pred[0][2]) * 100, 2)
        ),
        category=category,
        category_emoji=get_category_emoji(category),
        category_confidence=round(category_confidence * 100, 2),
        analyzed_at=datetime.now().isoformat()
    )


def fallback_analysis(review_text: str) -> ReviewAnalysis:
    """Fallback analysis using TextBlob when models are not loaded"""
    from textblob import TextBlob
    
    cleaned = preprocess_text(review_text)
    blob = TextBlob(review_text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = 'positive'
        scores = SentimentScore(negative=10.0, neutral=20.0, positive=70.0)
    elif polarity < -0.1:
        sentiment = 'negative'
        scores = SentimentScore(negative=70.0, neutral=20.0, positive=10.0)
    else:
        sentiment = 'neutral'
        scores = SentimentScore(negative=20.0, neutral=60.0, positive=20.0)
    
    category = detect_category_keywords(review_text)
    
    return ReviewAnalysis(
        review=review_text,
        cleaned_review=cleaned,
        sentiment=sentiment,
        sentiment_emoji=get_sentiment_emoji(sentiment),
        sentiment_confidence=75.0,
        sentiment_scores=scores,
        category=category,
        category_emoji=get_category_emoji(category),
        category_confidence=80.0,
        analyzed_at=datetime.now().isoformat()
    )


def detect_category_keywords(text: str) -> str:
    """Detect category based on keywords"""
    text_lower = text.lower()
    
    categories = {
        'cleanliness': ['clean', 'dirty', 'hygiene', 'dust', 'spotless', 'housekeeping', 'bathroom', 'sheets', 'towel', 'mold'],
        'food': ['food', 'breakfast', 'dinner', 'lunch', 'restaurant', 'buffet', 'meal', 'dining', 'chef', 'cuisine', 'delicious', 'taste'],
        'staff': ['staff', 'service', 'reception', 'employee', 'manager', 'concierge', 'helpful', 'rude', 'polite', 'friendly'],
        'amenities': ['pool', 'gym', 'wifi', 'spa', 'tv', 'air conditioning', 'elevator', 'parking', 'fitness', 'internet']
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category
    
    return 'overall'


def calculate_summary(analyses: List[ReviewAnalysis]) -> Dict[str, Any]:
    """Calculate summary statistics from analyses"""
    total = len(analyses)
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    category_counts = {'cleanliness': 0, 'food': 0, 'staff': 0, 'amenities': 0, 'overall': 0}
    
    avg_confidence = 0
    
    for analysis in analyses:
        sentiment_counts[analysis.sentiment] += 1
        category_counts[analysis.category] += 1
        avg_confidence += analysis.sentiment_confidence
    
    return {
        'sentiment_distribution': {
            'positive': {'count': sentiment_counts['positive'], 'percentage': round(sentiment_counts['positive'] / total * 100, 1)},
            'negative': {'count': sentiment_counts['negative'], 'percentage': round(sentiment_counts['negative'] / total * 100, 1)},
            'neutral': {'count': sentiment_counts['neutral'], 'percentage': round(sentiment_counts['neutral'] / total * 100, 1)}
        },
        'category_distribution': {k: {'count': v, 'percentage': round(v / total * 100, 1)} for k, v in category_counts.items() if v > 0},
        'average_confidence': round(avg_confidence / total, 2),
        'most_common_sentiment': max(sentiment_counts, key=sentiment_counts.get),
        'most_discussed_category': max(category_counts, key=category_counts.get)
    }


# ============== API Endpoints ==============


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "🏨 Welcome to Hotel Guest Review Sentiment Analyzer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthStatus, tags=["General"])
async def health_check():
    """Check API health and model status"""
    return HealthStatus(
        status="healthy",
        models_loaded={
            "sentiment_model": sentiment_model is not None,
            "category_model": category_model is not None,
            "tokenizer": tokenizer is not None
        },
        timestamp=datetime.now().isoformat()
    )


@app.post("/analyze", response_model=ReviewAnalysis, tags=["Analysis"])
async def analyze_review(input_data: ReviewInput):
    """
    Analyze a single hotel guest review.
    
    Returns detailed sentiment analysis including:
    - Sentiment classification (Positive/Negative/Neutral)
    - Confidence scores for each sentiment
    - Category detection (Cleanliness, Food, Staff, Amenities, Overall)
    """
    try:
        result = analyze_single_review(input_data.review)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalysisResult, tags=["Analysis"])
async def analyze_batch(input_data: BatchReviewInput):
    """
    Analyze multiple hotel guest reviews in batch.
    
    Returns individual analysis for each review plus summary statistics.
    """
    try:
        analyses = [analyze_single_review(review) for review in input_data.reviews]
        summary = calculate_summary(analyses)
        
        return BatchAnalysisResult(
            total_reviews=len(analyses),
            analyses=analyses,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.get("/analyze/quick", response_model=ReviewAnalysis, tags=["Analysis"])
async def quick_analyze(
    review: str = Query(..., min_length=3, max_length=5000, description="Hotel guest review text")
):
    """
    Quick analysis via GET request with review as query parameter.
    Useful for simple integrations and testing.
    """
    try:
        result = analyze_single_review(review)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/categories", tags=["Information"])
async def get_categories():
    """Get all supported review categories"""
    return {
        "categories": [
            {"id": "cleanliness", "name": "Cleanliness", "emoji": "🧹", "description": "Room cleanliness, hygiene, housekeeping"},
            {"id": "food", "name": "Food & Dining", "emoji": "🍽️", "description": "Restaurant, breakfast, room service"},
            {"id": "staff", "name": "Staff & Service", "emoji": "👨‍💼", "description": "Reception, customer service, hospitality"},
            {"id": "amenities", "name": "Amenities", "emoji": "🏊", "description": "Pool, gym, WiFi, facilities"},
            {"id": "overall", "name": "Overall Experience", "emoji": "🏨", "description": "General hotel experience"}
        ]
    }


@app.get("/sentiments", tags=["Information"])
async def get_sentiments():
    """Get all sentiment classifications"""
    return {
        "sentiments": [
            {"id": "positive", "name": "Positive", "emoji": "😊", "color": "#22c55e"},
            {"id": "neutral", "name": "Neutral", "emoji": "😐", "color": "#eab308"},
            {"id": "negative", "name": "Negative", "emoji": "😞", "color": "#ef4444"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
