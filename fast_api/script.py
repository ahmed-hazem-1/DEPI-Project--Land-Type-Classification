from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

app = FastAPI()

# نقطة البداية (اختياري)
@app.get("/")
def home():
    return {"message": "Welcome to Image Classification API 🚀"}

# رفع الصورة وتحويلها إلى مصفوفة
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # قراءة الصورة وتحويلها إلى مصفوفة NumPy
        image = Image.open(io.BytesIO(await file.read())).convert("L")  # تحويل الصورة إلى رمادي
        image = image.resize((128, 128))  # تغيير الحجم ليطابق بيانات التدريب
        image_array = np.array(image) / 255.0  # تطبيع الصورة

        # هنا ممكن تضيف تحميل الموديل لاحقًا وتطبيق التنبؤ
        return {"message": "Image received!", "shape": image_array.shape}
    
    except Exception as e:
        return {"error": str(e)}
