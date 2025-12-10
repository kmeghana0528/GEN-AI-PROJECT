# 🏨 Hotel Guest Review Sentiment Analyzer

A complete solution for analyzing hotel guest reviews using LSTM deep learning. This project helps hospitality teams understand guest sentiments and identify areas for improvement.

## 📁 Project Structure

```
hotel-sentiment-analyzer/
├── Hotel_Review_Sentiment_LSTM.ipynb    # Google Colab training notebook
├── backend/
│   ├── main.py                          # FastAPI backend server
│   ├── requirements.txt                 # Python dependencies
│   └── model/
│       └── hotel_sentiment_model/       # Trained models (after training)
│           ├── sentiment_model.h5
│           ├── category_model.h5
│           ├── tokenizer.pickle
│           ├── config.json
│           └── preprocessing.py
└── frontend/
    ├── index.html                       # Main HTML file
    ├── styles.css                       # CSS styles
    └── app.js                           # JavaScript application
```

## 🚀 Quick Start Guide

### Step 1: Train the Model (Google Colab)

1. Open `Hotel_Review_Sentiment_LSTM.ipynb` in Google Colab
2. Run all cells to:
   - Install dependencies
   - Prepare training data
   - Train the LSTM models
   - Download the model package as `hotel_sentiment_model.zip`

3. Extract the downloaded ZIP file to `backend/model/`

### Step 2: Set Up the Backend

```bash
# Navigate to backend folder
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Run the server
s
```

The API will be available at: `http://localhost:8000`

### Step 3: Run the Frontend

Simply open `frontend/index.html` in your web browser, or use a local server:

```bash
# Using Python
cd frontend
python -m http.server 5500

# Then open http://localhost:5500 in your browser
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | API health status |
| `/analyze` | POST | Analyze single review |
| `/analyze/batch` | POST | Analyze multiple reviews |
| `/analyze/quick` | GET | Quick analysis via query param |
| `/categories` | GET | List all categories |
| `/sentiments` | GET | List all sentiments |

### Example API Request

```bash
# Single review analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"review": "The room was amazing and staff were very helpful!"}'

# Response
{
  "review": "The room was amazing and staff were very helpful!",
  "sentiment": "positive",
  "sentiment_emoji": "😊",
  "sentiment_confidence": 95.5,
  "sentiment_scores": {
    "negative": 2.1,
    "neutral": 2.4,
    "positive": 95.5
  },
  "category": "staff",
  "category_emoji": "👨‍💼",
  "category_confidence": 88.3,
  "analyzed_at": "2024-01-01T12:00:00"
}
```

## 🎯 Features

### Sentiment Detection
- **Positive** 😊 - Guest had a good experience
- **Neutral** 😐 - Average/mixed experience
- **Negative** 😞 - Guest had issues/complaints

### Category Detection
- **Cleanliness** 🧹 - Room hygiene, housekeeping
- **Food** 🍽️ - Restaurant, breakfast, dining
- **Staff** 👨‍💼 - Service, reception, hospitality
- **Amenities** 🏊 - Pool, gym, WiFi, facilities
- **Overall** 🏨 - General experience

## 🛠️ Technology Stack

- **Model Training**: TensorFlow, Keras, LSTM
- **Backend**: Python, FastAPI, NLTK
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **NLP**: NLTK, TextBlob (fallback)

## 📊 Model Architecture

```
Bidirectional LSTM Model
├── Embedding Layer (128 dimensions)
├── Bidirectional LSTM (128 units, return_sequences=True)
├── Dropout (0.3)
├── Bidirectional LSTM (64 units)
├── Dropout (0.3)
├── Dense (64 units, ReLU)
├── Dropout (0.2)
├── Dense (32 units, ReLU)
└── Dense (3 units, Softmax) - Output
```

## 📝 Sample Reviews for Testing

**Positive:**
- "The room was absolutely amazing and the staff were incredibly helpful!"
- "Best hotel experience ever! Clean rooms and delicious breakfast."

**Negative:**
- "Terrible experience. The bathroom was dirty and the food was cold."
- "Staff was rude and WiFi didn't work at all."

**Neutral:**
- "It was an okay stay. Nothing special but met basic expectations."
- "Average hotel, decent for the price."

## 🔧 Troubleshooting

### Backend Issues

1. **Models not loading**: Make sure you've extracted the model ZIP to `backend/model/hotel_sentiment_model/`

2. **NLTK data missing**: Run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt_tab')
   ```

3. **Port already in use**: Change the port in `main.py`:
   ```python
   uvicorn.run(app, host="0.0.0.0", port=8001)
   ```

### Frontend Issues

1. **CORS errors**: Make sure the backend is running on the correct port (8000)

2. **API not connecting**: The frontend has a fallback local analysis mode that works without the backend

## 📈 Future Improvements

- [ ] Add more training data from real hotel reviews
- [ ] Implement aspect-based sentiment analysis
- [ ] Add multi-language support
- [ ] Create Docker containerization
- [ ] Add database for storing analysis history
- [ ] Implement user authentication

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- FastAPI for the excellent Python web framework
- NLTK for natural language processing tools
