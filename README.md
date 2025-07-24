# NeuroVision AI - Brain Tumor Detection

This project is an advanced brain tumor detection web application using deep learning (YOLO) and FastAPI. It allows users to upload MRI scans and receive instant AI-powered analysis, including tumor type, confidence, and risk assessment.

## Features
- Upload MRI scans (JPG, PNG, DICOM)
- Instant tumor detection and classification (Glioma, Meningioma, Pituitary, No Tumor)
- Visual highlights of detected regions
- Downloadable analysis report
- Modern, responsive frontend

## Tech Stack
- **Backend:** FastAPI, Ultralytics YOLO, OpenCV, Pillow, NumPy
- **Frontend:** Bootstrap, HTML, JavaScript

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the YOLO model
- Put your trained YOLO model (e.g., `new_best.pt`) in the `model/` directory.

### 4. Run the app
```bash
uvicorn main:app --reload
```
- Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Deployment
- Deployable on Render, Heroku, or any cloud supporting FastAPI.
- Update the frontend JS to point to your deployed backend URL if needed.

## File Structure
- `main.py` - FastAPI backend
- `templates/` - HTML templates
- `images/` - Static images for frontend
- `model/` - YOLO model weights
- `dataset/` - (Optional) Training/validation data

## .gitignore
- See `.gitignore` for files/folders to exclude from version control (e.g., model weights, __pycache__, etc.)

## License
MIT 