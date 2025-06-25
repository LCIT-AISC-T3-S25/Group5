# Flask Sentiment Analysis Application

This project is a Flask web application that provides sentiment analysis using a Bidirectional GRU model. It allows users to input text and receive predictions on the sentiment of that text.

## Project Structure

```
flask-app-container
├── app
│   ├── index.py          # Main Flask application
│   └── templates
│       └── index.html    # HTML interface for sentiment prediction
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd flask-app-container
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   python app/index.py
   ```

5. **Access the application:**
   Open your web browser and go to `http://localhost:5002` to access the sentiment prediction interface.

## Usage

- Enter the text you want to analyze in the input field.
- Click the "Predict" button to get the sentiment prediction.
- The predicted sentiment and confidence level will be displayed below the input field.

## Dependencies

- Flask
- Flask-CORS

## License

This project is licensed under the MIT License.