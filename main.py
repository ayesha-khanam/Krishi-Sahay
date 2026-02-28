from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import shutil
import os

from disease_model import predict_image  # your function name is predict_image

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# -----------------------------
# Home UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# RAG Components
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("agri_index.faiss")

with open("data/agriculture.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines() if line.strip()]


class Query(BaseModel):
    question: str
    language: str = "English"


def retrieve_docs(query: str, k: int = 3) -> str:
    qv = embedding_model.encode([query])
    qv = np.array(qv).astype("float32")

    distances, indices = index.search(qv, k)

    retrieved_lines = []
    for idx in indices[0]:
        if 0 <= idx < len(texts):
            retrieved_lines.append(texts[idx])

    return "\n".join(retrieved_lines)


# -----------------------------
# Government Schemes (Rule-based + clean)
# -----------------------------
SCHEMES = [
    {
        "name": "PM-KISAN",
        "who": "Small & marginal farmers",
        "benefit": "₹6,000 per year income support (3 installments)",
        "keywords": ["money", "income", "installment", "pm kisan", "kisan", "support"]
    },
    {
        "name": "Soil Health Card",
        "who": "All farmers",
        "benefit": "Soil testing + fertilizer recommendations",
        "keywords": ["soil", "ph", "fertilizer", "nutrient", "test", "health card"]
    },
    {
        "name": "PMFBY (Crop Insurance)",
        "who": "Farmers with insured crops",
        "benefit": "Insurance for crop loss due to natural calamities/pests",
        "keywords": ["insurance", "loss", "flood", "drought", "hail", "damage", "pest"]
    },
    {
        "name": "Kisan Credit Card (KCC)",
        "who": "Farmers needing short-term credit",
        "benefit": "Low-interest farm loan/credit",
        "keywords": ["loan", "credit", "kcc", "money for farming", "finance"]
    },
    {
        "name": "Pradhan Mantri Krishi Sinchai Yojana (PMKSY)",
        "who": "Farmers needing irrigation support",
        "benefit": "Support for irrigation, drip/sprinkler, water saving",
        "keywords": ["irrigation", "drip", "sprinkler", "water", "sinchai"]
    },
]

class SchemeRequest(BaseModel):
    question: str
    language: str = "English"

@app.post("/schemes")
def suggest_schemes(req: SchemeRequest):
    q = req.question.lower()
    matched = []
    for s in SCHEMES:
        score = sum(1 for k in s["keywords"] if k in q)
        if score > 0:
            matched.append((score, s))
    matched.sort(key=lambda x: x[0], reverse=True)

    top = [m[1] for m in matched[:3]]

    # If nothing matches, show the most generally useful ones
    if not top:
        top = [SCHEMES[0], SCHEMES[1], SCHEMES[2]]

    # Return clean list (frontend will show nicely)
    return {"language": req.language, "schemes": top}


# -----------------------------
# Weather API (Free, no key): Open-Meteo
# Requires Internet connection to fetch weather.
# -----------------------------
@app.get("/weather")
def weather(city: str):
    # 1) Geocode city -> lat/lon
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=30
    ).json()

    if "results" not in geo or not geo["results"]:
        return {"error": f"City not found: {city}"}

    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    # 2) Get forecast
    fc = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto"
        },
        timeout=30
    ).json()

    current = fc.get("current_weather", {})
    daily = fc.get("daily", {})

    result = {
        "city": place.get("name"),
        "country": place.get("country"),
        "current": {
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed"),
            "weathercode": current.get("weathercode"),
        },
        "daily": {
            "dates": daily.get("time", [])[:3],
            "tmax": daily.get("temperature_2m_max", [])[:3],
            "tmin": daily.get("temperature_2m_min", [])[:3],
            "rain": daily.get("precipitation_sum", [])[:3],
        }
    }
    return result


# -----------------------------
# Ask Endpoint (Offline AI via Ollama)
# Grounded + Multilingual
# -----------------------------
@app.post("/ask")
def ask_question(query: Query):
    context = retrieve_docs(query.question)

    prompt = f"""
You are KrishiSahay, an agriculture assistant.

IMPORTANT RULES:
- Answer ONLY using the given Context + the Question.
- If Context is not enough, say: "I don't have enough information in my data. Please ask with more details."
- Answer in: {query.language}
- If Hindi, use Devanagari (हिंदी). If Telugu, use Telugu script (తెలుగు).
- Keep it short and step-by-step.

Context:
{context}

Question:
{query.question}
"""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi", "prompt": prompt, "stream": False},
            timeout=180
        )
        r.raise_for_status()
        data = r.json()
        answer = data.get("response", "").strip()
        return {"question": query.question, "answer": answer, "context": context}
    except Exception as e:
        return {
            "question": query.question,
            "answer": "Ollama is not responding. Please start Ollama and try again.",
            "error": str(e)
        }


# -----------------------------
# Image Disease Detection + Advice
# -----------------------------
@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)

    try:
        os.remove(file_path)
    except Exception:
        pass

    # If it doesn't look like a plant/leaf image
    if not result["is_leaf"]:
        return {
            "error": "Invalid image",
            "message": "Please upload a clear crop/leaf image (not a human/animal/object).",
            "detected_as": result["imagenet_label"]
        }

    # Otherwise generate advice
    prediction = result["disease_label"]
    prompt = f"""
A crop/leaf image was classified as: {prediction}.
Give simple treatment steps and prevention tips.
"""

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi", "prompt": prompt, "stream": False},
        timeout=180
    )
    advice = r.json().get("response", "").strip()

    return {
        "prediction": prediction,
        "confidence": round(result["confidence"] * 100, 2),
        "advice": advice
    }