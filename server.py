from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io, cv2, numpy as np
from inference import predict

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_largest_face(pil_img):
    arr = np.array(pil_img.convert("L"))
    h, w = arr.shape

    scale = 1.0
    if max(h, w) < 300:
        scale = 300 / max(h, w)
        arr_detect = cv2.resize(arr, (int(w * scale), int(h * scale)))
    else:
        arr_detect = arr

    faces = face_cascade.detectMultiScale(arr_detect, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        return pil_img

    x, y, w_, h_ = max(faces, key=lambda f: f[2] * f[3])
    x, y, w_, h_ = int(x / scale), int(y / scale), int(w_ / scale), int(h_ / scale)
    return pil_img.crop((x, y, x + w_, y + h_))

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    face = crop_largest_face(img)
    return {
        "cnn": predict(face, "cnn"),
        "resnet": predict(face, "resnet"),
    }