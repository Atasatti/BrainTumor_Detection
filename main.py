from fastapi import FastAPI, File, UploadFile, Request
# from fastapi import FastAPI, HTTPException, status, Depends, Request, Form
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.responses import JSONResponse
import base64
import io
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.mount("/images", StaticFiles(directory="images"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


# Load the YOLO model
model = YOLO(r"C:\\Users\\PMLS\\Desktop\\APIS\\Brain_Tumor\\model\\new_best.pt")

# Define a tumor type mapping
TUMOR_TYPE_MAPPING = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

def preprocess_image(image_bytes):
    image = np.array(Image.open(BytesIO(image_bytes)))
    resized_image = cv2.resize(image, (640, 640))
    normalized_image = resized_image / 255.0
    return (normalized_image * 255).astype(np.uint8)

def draw_boxes(image, boxes): 
    for box in boxes:
        x_min, y_min, x_max, y_max = box['x_min'], box['y_min'], box['x_max'], box['y_max']
        confidence = box['confidence']
        tumor_type = box['tumor_type']

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"{tumor_type} ({confidence:.2f})"
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})







@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    
    results = model.predict(source=image, imgsz=640, conf=0.5)
    predictions = []
    
    for result in results[0].boxes:
        xyxy = result.xyxy.tolist()
        conf = result.conf.tolist()
        class_id = result.cls.tolist()
        
        for i in range(len(xyxy)):
            x_min, y_min, x_max, y_max = map(int, xyxy[i])
            predictions.append({
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'confidence': conf[i],
                'tumor_type': TUMOR_TYPE_MAPPING.get(class_id[i], "Unknown")
            })
    
    image_with_boxes = draw_boxes(image, predictions)
    
    # Convert the image with boxes to base64
    _, image_with_boxes_bytes = cv2.imencode('.png', image_with_boxes)
    image_base64 = base64.b64encode(image_with_boxes_bytes).decode('utf-8')
    
    return JSONResponse(content={"predictions": predictions, "highlighted_image": image_base64})
