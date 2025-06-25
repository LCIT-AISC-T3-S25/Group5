# 🧠 Sentiment & Image Classification Microservices

This project consists of multiple Dockerized microservices to perform sentiment prediction and image classification via deep learning models, accessible through a single frontend UI.

---

## 📥 Clone the Repository

```bash
git clone https://github.com/LCIT-AISC-T3-S25/Group5.git
cd Group5
```

---

## 🚀 Run the Entire Project

Use Docker Compose to build and run all services with a single command:

```bash
docker-compose up --build
```

This will spin up the following services:

| Service         | Port  | Description                        |
|-----------------|-------|------------------------------------|
| UI              | 5002  | HTML frontend (Flask)              |
| BiLSTM GRU      | 5004  | Sentiment prediction (GRU)         |
| Word2Vec LSTM   | 5003  | Sentiment prediction (Word2Vec)    |
| VGG16 FastAPI   | 5005  | Image classification using VGG16   |
| CNN FastAPI     | 5006  | Custom CNN model for image input   |

---

## 🛠️ Manual Build & Run (if not using Compose)

### BiLSTM GRU (Port 5004)
```bash
docker build -t bi_lstm ./GRU_Sentiment
docker run -v $(pwd):/app -p 5004:5004 bi_lstm
```

### Word2Vec LSTM (Port 5003)
```bash
docker build -t word2vec ./word2vec_lstm_Model
docker run -v $(pwd):/app -p 5003:5003 word2vec
```

### UI (Port 5002)
```bash
docker build -t models_html ./HTML/flask-app-container
docker run -v $(pwd):/app -p 5002:5002 models_html
```

### VGG16 FastAPI (Port 5005)
```bash
docker build -t vgg16_fast_api ./VGG16_FastAPI
docker run -v $(pwd):/app -p 5005:5005 vgg16_fast_api
```

### CNN FastAPI (Port 5006)
```bash
docker build -t cnn ./CNN_FastAPI
docker run -v $(pwd):/app -p 5006:5006 cnn
```

---

## 🌐 Access the Application

Once all containers are running, open your browser and navigate to:

```
http://localhost:5002
```

---

## 📂 Project Structure

```
Group5/
│
├── docker-compose.yml
├── HTML/flask-app-container # Flask UI (Port 5002)
├── GRU_Sentiment/           # BiLSTM GRU model (Port 5004)
├── word2vec_lstm_Model/     # Word2Vec LSTM model (Port 5003)
├── VGG16_FastAPI/           # VGG16 model in FastAPI (Port 5005)
└── CNN_FastAPI/             # CNN model in FastAPI (Port 5006)
```

---

## 🧪 Health Check Endpoints

You can check if services are running with:

- `http://localhost:5002/health`
- `http://localhost:5003/health`
- `http://localhost:5004/health`
- `http://localhost:5005/docs` (VGG16 FastAPI Swagger UI)
- `http://localhost:5006/docs` (CNN FastAPI Swagger UI)

---

## 📦 Requirements

- Docker
- Docker Compose

---

## 📜 License

MIT © 2025 Group5
