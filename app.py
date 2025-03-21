from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logging.info("App started successfully!")

@app.get("/")
def home():
    return {"message": "Welcome to Image Classification API 🚀"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")
        
        # Process image
        image = Image.open(io.BytesIO(await file.read())).convert("L")
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0

        return {"message": "Image received!", "shape": image_array.shape}
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return {"error": str(e)}

