import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from .api import predict

app = FastAPI(
    title="GBV Predictive Tool API",
    description="An API to predict the risk of Gender-Based Violence.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files only if directory exists
if os.path.isdir("backend/static"):
    app.mount("/static", StaticFiles(directory="backend/static"), name="static")
elif os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# API routes
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])

# Serve frontend pages
@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/faq", response_class=HTMLResponse)
@app.get("/faq.html", response_class=HTMLResponse)
async def faq(request: Request):
    return templates.TemplateResponse("faq.html", {"request": request})

# Health check
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}

# Catch-all
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def catch_all(request: Request, full_path: str):
    return templates.TemplateResponse("index.html", {"request": request})