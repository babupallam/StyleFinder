from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import search
from fastapi.staticfiles import StaticFiles
import os
from backend.routes import models

app = FastAPI()

GALLERY_PATH = os.path.abspath("../data/processed/splits/gallery")
app.mount("/static", StaticFiles(directory=GALLERY_PATH), name="static")


app.include_router(models.router, prefix="/api")

# Allow frontend (React) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or "*" for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api")

