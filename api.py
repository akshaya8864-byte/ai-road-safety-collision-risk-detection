import os
import tempfile
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Road Safety Detection API")
model = YOLO("yolo11n.pt")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect/image")
def detect_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Please upload a valid image.")

    result = model.predict(source=image, verbose=False)[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "label": result.names[class_id],
                "confidence": round(float(box.conf[0].item()), 3),
                "box_xyxy": [
                    round(float(value), 1)
                    for value in box.xyxy[0].tolist()
                ]
            })

    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections
    }


@app.post("/detect/video")
def detect_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(
            status_code=400,
            detail="Upload an MP4, MOV, AVI, or MKV video."
        )

    temp_path = None
    results = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.file.read())
            temp_path = temp_file.name

        frames = []
        class_counts = {}

        results = model.track(
            source=temp_path,
            tracker="bytetrack.yaml",
            stream=True,
            imgsz=416,
            vid_stride=3,
            verbose=False
        )

        for frame_number, result in enumerate(results, start=1):
            objects = []

            if result.boxes is not None:
                track_ids = result.boxes.id

                for index, box in enumerate(result.boxes):
                    class_id = int(box.cls[0].item())
                    label = result.names[class_id]
                    track_id = (
                        int(track_ids[index].item())
                        if track_ids is not None
                        else None
                    )

                    objects.append({
                        "track_id": track_id,
                        "label": label,
                        "confidence": round(float(box.conf[0].item()), 3),
                        "box_xyxy": [
                            round(float(value), 1)
                            for value in box.xyxy[0].tolist()
                        ]
                    })
                    class_counts[label] = class_counts.get(label, 0) + 1

            frames.append({
                "frame": frame_number,
                "objects": objects
            })

            if frame_number >= 60:
                break

        return {
            "filename": file.filename,
            "frames_processed": len(frames),
            "limited_to_60_frames": len(frames) == 60,
            "detection_counts_by_class": class_counts,
            "frames": frames
        }

    except Exception:
        app.logger.exception("Video detection failed")
        raise HTTPException(
            status_code=500,
            detail="Video processing failed. Check the VS Code terminal."
        )

    finally:
        try:
            if results is not None:
                results.close()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)