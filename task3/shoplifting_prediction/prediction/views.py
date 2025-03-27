import os
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from .forms import VideoUploadForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tempfile import NamedTemporaryFile
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Load the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mobile_net.keras')
model = load_model(MODEL_PATH)

# Constants
FRAME_SIZE = (224, 224)
NUM_FRAMES = 20


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // NUM_FRAMES)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = img_to_array(frame) / 255.0
            frames.append(frame)
        count += 1

    cap.release()

    # Pad or truncate to exactly NUM_FRAMES
    if len(frames) < NUM_FRAMES:
        padding = [np.zeros((224, 224, 3), dtype=np.float32)] * (NUM_FRAMES - len(frames))
        frames.extend(padding)
    else:
        frames = frames[:NUM_FRAMES]

    return np.array(frames)


def predict_video(video):
    logger.info(f"Received video: {video.name}")
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        for chunk in video.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name
        logger.info(f"Saved video to: {temp_file_path}")

    try:
        frames = extract_frames(temp_file_path)
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        prediction = model.predict(frames)[0][0]
        logger.info(f"Model prediction: {prediction}")

        confidence = prediction if prediction >= 0.5 else 1 - prediction
        label = 'Non Shop Lifter' if prediction >= 0.5 else 'Shop Lifter'

        return label, confidence
    finally:
        os.remove(temp_file_path)


def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            try:
                result, confidence = predict_video(video)
                return JsonResponse({
                    'success': True,
                    'prediction': result,
                    'confidence': float(confidence)  # Ensure JSON serializable
                })
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return JsonResponse({'success': False, 'error': str(e)})

        return JsonResponse({'success': False, 'error': 'Invalid form submission'})

        # Render template for GET requests
    form = VideoUploadForm()
    return render(request, 'prediction/upload.html', {'form': form})