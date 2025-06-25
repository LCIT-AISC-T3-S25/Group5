# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# @app.route("/health", methods=["GET"])
# def health_check():
#     return jsonify({"status": "healthy", "model": "Bidirectional GRU Sentiment Analysis"})

# @app.route("/", methods=["GET"])
# def index():
#     return '''
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>GRU Sentiment Prediction</title>
#         <script>
#             async function sendPrediction() {
#                 const text = document.getElementById("text-input").value;
#                 const resultDiv = document.getElementById("result");

#                 const response = await fetch("http://localhost:5001/predict", {
#                     method: "POST",
#                     headers: {
#                         "Content-Type": "application/json"
#                     },
#                     body: JSON.stringify({ text: text })
#                 });

#                 const data = await response.json();
#                 if (response.ok) {
#                     resultDiv.innerHTML = `
#                         <p><strong>Input:</strong> ${data.input_text}</p>
#                         <p><strong>Predicted Sentiment:</strong> ${data.predicted_sentiment}</p>
#                         <p><strong>Confidence:</strong> ${data.confidence.toFixed(3)}</p>
#                         <p><strong>Model:</strong> ${data.model}</p>
#                     `;
#                 } else {
#                     resultDiv.innerHTML = `<p style="color:red;"><strong>Error:</strong> ${data.error}</p>`;
#                 }
#             }
#         </script>
#     </head>
#     <body>
#         <h2>Sentiment Prediction (Bidirectional GRU)</h2>
#         <input type="text" id="text-input" placeholder="Enter text" size="60">
#         <button onclick="sendPrediction()">Predict</button>
#         <div id="result" style="margin-top:20px;"></div>
#     </body>
#     </html>
#     '''

# if __name__ == "__main__":
#     serve(app, host="0.0.0.0", port=5002)
