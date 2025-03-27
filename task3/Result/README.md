# 🛒 Shoplifting Prediction using Django and TensorFlow  

This project is a Django-based web application that predicts shoplifting behavior from video input using a pre-trained TensorFlow model. The model processes video data, extracts frames, and predicts whether the person in the video is a shoplifter or not.  

---

## 🚀 Features  
✅ Upload video through a web interface  
✅ Preprocess video frames to match model input requirements  
✅ Predict using a TensorFlow-based model (MobileNet)  
✅ Display prediction and confidence score  
✅ Handles edge cases like corrupted video files  
✅ Logs predictions and errors for debugging  
✅ Clean and responsive user interface  

---


---

## 🎯 Usage  
### 1. **Open the app**  
- Open your browser and go to:  
👉 `http://127.0.0.1:8000/upload/`  

### 2. **Upload a video**  
- Select an MP4 file  
- Click **Upload**  

### 3. **Get Prediction**  
- The app will display:  
  ✅ Prediction result ("Shop Lifter" or "Non Shop Lifter")  
  ✅ Confidence score  

---

## 📝 Code Overview  

### **views.py**  
Handles video upload, preprocessing, and prediction:  
- **extract_frames** – Extracts frames from video, resizes them, and normalizes them.  
- **predict_video** – Loads the video, processes frames, and predicts using the TensorFlow model.  
- **upload_video** – Handles the POST request and renders the result.  

### **forms.py**  
Defines the `VideoUploadForm` for handling video input.  

### **upload.html**  
HTML form for video upload and displaying results.  

### **styles.css**  
CSS for styling the web interface.  

---

## 🤖 Model Details  
| Parameter | Value |
|-----------|--------|
| **Model** | MobileNet |
| **Input Shape** | (20 frames, 224 x 224, 3 channels) |
| **Output** | Probability of shoplifting (binary classification) |
| **Activation** | Sigmoid |
| **Loss Function** | Binary Crossentropy |
| **Optimizer** | Adam |

---

## 🔥 Prediction Logic  
1. The uploaded video is saved to a temporary file.  
2. Frames are extracted, resized to 224x224, and normalized.  
3. Frames are padded or truncated to 20 frames.  
4. The processed frames are passed to the MobileNet model.  
5. Model outputs a probability between 0 and 1.  
   - If `prediction >= 0.5`, label = **Non Shop Lifter**  
   - If `prediction < 0.5`, label = **Shop Lifter**  
6. Prediction and confidence score are logged and displayed.  

---

## 📌 Environment  
| Dependency | Version |
|-----------|---------|
| Python     | 3.10+   |
| Django     | 4.x     |
| TensorFlow | 2.x     |
| OpenCV     | 4.x     |
| NumPy      | 1.x     |

---

## 🌟 Results  
Here’s an example of the web interface and prediction results:  

### 🖼️ 1. Upload Interface  
Screenshot of the video upload page:  
![Upload Interface](https://github.com/user-attachments/assets/78e5b280-4109-4500-b68b-a8458da0a7c7)
 

---

### ✅ 2. Prediction Result  
Example of a successful prediction (Non-Shop Lifting):  
![Prediction Result]((https://github.com/user-attachments/assets/be58b69d-7928-4e26-af6a-b3f53649fd5c)
)  
Example of a successful prediction (Shop Lifting):
![Prediction Result]((https://github.com/user-attachments/assets/d6f66ffe-33dd-4dfa-b1d1-f35408d10005)
)
---




