# NeuroVision AI – Brain Tumor Detection Platform

## Overview
NeuroVision AI is a web-based application that leverages state-of-the-art deep learning (YOLOv8) to detect and classify brain tumors from MRI scans. The platform provides a seamless experience for users to upload MRI images, receive instant AI-powered analysis, and download comprehensive reports.

## Features
- **Automated Tumor Detection:** Upload MRI scans and get real-time predictions for Glioma, Meningioma, Pituitary tumors, or No Tumor.
- **Visual Explanations:** The AI highlights regions of interest directly on the MRI image.
- **Detailed Confidence Scores:** Each prediction includes a confidence level and risk assessment.
- **Downloadable Reports:** Users can download a full analysis report for further review or sharing.
- **Modern UI:** Built with Bootstrap and custom design for a professional look and feel.
- **Secure & Private:** All data is processed securely, and privacy is a top priority.

## Tech Stack
- **Backend:** Python, FastAPI, YOLOv8, OpenCV, PIL
- **Frontend:** HTML, CSS (Bootstrap), JavaScript
- **Model:** Custom-trained YOLOv8 on a labeled MRI dataset

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Atasatti/BrainTumor_Detection.git
   cd BrainTumor_Detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the YOLOv8 model weights:**
   - Place your trained YOLOv8 model (e.g., `new_best.pt`) in the `model/` directory.

4. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```
5. **Access the web interface:**
   - Open your browser and go to `http://127.0.0.1:8000/`

## Usage
- Upload an MRI scan via the web interface.
- The backend processes the image and runs inference using the YOLOv8 model.
- Detected tumor regions are highlighted, and predictions are returned with confidence scores.
- Download a detailed report for further review.

## Folder Structure
- `dataset/` – Contains training and validation MRI images and labels.
- `model/` – Contains YOLOv8 model weights.
- `templates/` – HTML templates for the web interface.
- `images/` – Static images for the frontend.
- `main.py` – Main FastAPI application.

## License
This project is for educational and research purposes only.

## Contact
For questions or collaboration, please contact [Your Name] or open an issue on GitHub. 