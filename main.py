# FastAPI 应用：自动识别主要人物并填充背景
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import cv2
import numpy as np
import io

app = FastAPI()

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        expand = 0.6
        x1 = max(int(x - w * expand), 0)
        y1 = max(int(y - h * expand), 0)
        x2 = min(int(x + w * (1 + expand)), cv_image.shape[1])
        y2 = min(int(y + h * (1 + expand)), cv_image.shape[0])
        mask[y1:y2, x1:x2] = 255

    bg_color = (255, 255, 255)
    alpha = cv_image[:, :, 3] / 255.0
    foreground = cv_image[:, :, :3]
    background = np.full(foreground.shape, bg_color, dtype=np.uint8)
    composed = np.uint8((1 - alpha[..., None]) * background + alpha[..., None] * foreground)
    composed_masked = np.where(mask[..., None] == 255, composed, background)

    result = Image.fromarray(composed_masked.astype(np.uint8))
    buf = io.BytesIO()
    result.save(buf, format="JPEG")
    buf.seek(0)
    return buf

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    processed = process_image(image_data)
    return StreamingResponse(processed, media_type="image/jpeg")