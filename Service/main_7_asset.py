from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

# Load the YOLO model (ensure the path is correct)
model = YOLO("/home/nic/workspace/Mlops/MHD_1/Models/runs_7/detect/train/weights/best.pt")

app = FastAPI(title="Military Hardware Detection API")

@app.post("/detect/")
async def detect_military_hardware(file: UploadFile = File(...)):
    """
    Detect military hardware in the uploaded image
    
    - Accepts a single image file
    - Returns detection results or 'no asset found' message
    """
    # Validate file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run inference
        results = model(image)
        
        # Process results
        if not results or len(results[0].boxes) == 0:
            return JSONResponse(content={
                "detected": False,
                "message": "No military hardware detected"
            })
        
        # Prepare detection details
        detections = []
        for result in results:
            for box in result.boxes:
                # Get class name and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bounding_box": box.xyxy[0].tolist()
                })
        
        return JSONResponse(content={
            "detected": True,
            "assets": detections
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a health check endpoint
@app.get("/")
async def health_check():
    return {"status": "API is running", "model": "Military Hardware Detection"}

# To run this API:
# 1. Save this script as `main_7_asset.py`
# 2. Install requirements: 
#    pip install fastapi uvicorn ultralytics pillow
# 3. Run with: 
#    uvicorn main_7_asset:app --reload