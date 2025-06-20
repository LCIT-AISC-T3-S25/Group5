from fastapi import FastAPI
from predictor import router

app = FastAPI(
    title="FastAPI",
    description="Predicts image class using image and metadata.",
    version="1.0"
)

app.include_router(router)