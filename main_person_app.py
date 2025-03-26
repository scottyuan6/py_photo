
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
import io
import cv2
import numpy as np

# === 图像处理函数 ===
def process_image(image_bytes):
    # 使用 rembg 去除背景
    removed = remove(image_bytes)
    person = Image.open(io.BytesIO(removed)).convert("RGBA")

    # 将 RGBA 转换为 OpenCV 格式
    cv_image = cv2.cvtColor(np.array(person), cv2.COLOR_RGBA2BGRA)

    # 使用 OpenCV 检测所有人脸
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 创建人物蒙版，仅保留检测到的所有人脸区域周围（扩大区域）
    mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        expand = 0.6
        x1 = max(int(x - w * expand), 0)
        y1 = max(int(y - h * expand), 0)
        x2 = min(int(x + w * (1 + expand)), cv_image.shape[1])
        y2 = min(int(y + h * (1 + expand)), cv_image.shape[0])
        mask[y1:y2, x1:x2] = 255

    # 对于 alpha 通道背景部分用背景色填充
    bg_color = (255, 255, 255)
    alpha = cv_image[:, :, 3] / 255.0
    foreground = cv_image[:, :, :3]
    background = np.full(foreground.shape, bg_color, dtype=np.uint8)
    composed = np.uint8((1 - alpha[..., None]) * background + alpha[..., None] * foreground)

    # 将非主要人物区域遮盖为背景色
    composed_masked = np.where(mask[..., None] == 255, composed, background)

    # 转换为 PIL 图像
    result = Image.fromarray(composed_masked.astype(np.uint8))
    return result

# === Streamlit 页面 ===
st.set_page_config(page_title="主要人物识别器", layout="centered")
st.title("🧠 自动识别主角并去除其他人物")

st.markdown("""
- 上传图片后，系统会自动识别图中主要人物（多个）
- 非主要人物将被删除，背景自动填充为白色
- 无需任何操作，上传即可自动处理完成
""")

uploaded_file = st.file_uploader("上传你的照片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_bytes = uploaded_file.read()
    final = process_image(img_bytes)

    st.image(final, caption="自动处理结果", use_column_width=True)

    buf = io.BytesIO()
    final.save(buf, format="JPEG")
    st.download_button("📥 下载图像", data=buf.getvalue(), file_name="main_person_only.jpg", mime="image/jpeg")
