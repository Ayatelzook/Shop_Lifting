# 🛒 Shoplifting Detection Using Deep Learning

This project focuses on detecting shoplifting activities using video data. The goal is to create a system that processes video frames, applies deep learning models, and predicts whether shoplifting is occurring. The project explores different deep learning approaches to improve the accuracy and efficiency of shoplifting detection.

## 📌 Project Overview

### **Purpose**  
The project involves the following key steps:  
1. **Data Preprocessing** – Preparing video frames for input to the model.  
2. **Modeling** – Implementing different deep learning architectures to detect shoplifting.  
3. **Training** – Training the models on the preprocessed video data.  
4. **Evaluation** – Assessing the models’ performance using accuracy, precision, and recall.  
5. **Prediction** – Creating a pipeline for real-time shoplifting detection on new video data.  

---

## 📂 Project Structure

---

## 🎥 Data  

### **Input Data**  
The dataset consists of video files in MP4 format, categorized into two classes:  
- **Shoplifting** – Videos showing shoplifting activities.  
- **Non-shoplifting** – Videos showing normal shopping behavior.  

### **Data Preprocessing**  
The preprocessing steps include:  
- **Frame Extraction** – Videos are converted into frames at a fixed frame rate.  
- **Padding and Normalization** –  
    - Videos are padded to max frames if they are shorter.  
    - Pixel values are normalized to the range `[0, 1]`.  
- **Resizing** – Frames are resized to a fixed resolution for consistent input size.  

---

## 🏆 Models  

### **1. I3D Model (Inflated 3D Convolutional Neural Network)**  
The I3D model is designed to capture spatiotemporal patterns in video data. It expands 2D convolutional layers into 3D to process video sequences more effectively.  

#### **Architecture**  
- **3D Convolutional Layers** – Extract spatiotemporal features from video frames.  
- **Batch Normalization** – Normalizes activations to improve convergence.  
- **Global Average Pooling** – Reduces dimensionality while retaining important information.  
- **Fully Connected Layers** – Dense layers for final classification.  
- **Sigmoid Activation** – Outputs a binary prediction (shoplifting or not).  

#### **Highlights**  
✅ Strong spatiotemporal learning through 3D convolutions.  
✅ Effective for capturing complex video patterns.  

#### **Training**  
- Optimizer: Adam  
- Loss: Binary Cross-Entropy  
- Metrics: Accuracy, Precision, Recall  

---

### **2. CNN + Bidirectional LSTM Model**  
This model combines convolutional layers for spatial feature extraction and a BiLSTM layer for temporal feature learning.  

#### **Architecture**  
- **TimeDistributed CNN Layers** – Applies convolution to each frame individually.  
- **Batch Normalization and MaxPooling** – Improves learning efficiency.  
- **Global Average Pooling** – Reduces dimensionality.  
- **Bidirectional LSTM** – Captures sequential patterns and improves temporal understanding.  
- **Dense Layers** – Final classification layers with dropout for regularization.  

#### **Highlights**  
✅ Combines spatial and temporal learning.  
✅ BiLSTM captures long-range dependencies between frames.  

#### **Training**  
- Optimizer: Adam (learning rate = 0.0001)  
- Loss: Binary Cross-Entropy  
- Metrics: Accuracy, Recall  

---

### **3. Hybrid 3D CNN + LSTM Model**  
This model combines 3D convolutional layers for spatiotemporal feature extraction and LSTM layers for temporal learning.  

#### **Architecture**  
- **3D Convolutional Layers** – Extracts spatiotemporal patterns.  
- **Batch Normalization** – Normalizes activations to stabilize training.  
- **MaxPooling** – Reduces the spatial dimensions.  
- **TimeDistributed Flattening** – Preserves temporal features across frames.  
- **LSTM Layer** – Learns long-term dependencies between frames.  
- **Dense Layers** – Fully connected layers for classification.  
- **Softmax Activation** – Outputs class probabilities.  

#### **Highlights**  
✅ Effective combination of 3D CNN and LSTM for complex spatiotemporal learning.  
✅ Softmax activation for multi-class classification.  

#### **Training**  
- Optimizer: Adam (learning rate = LEARNING_RATE)  
- Loss: Sparse Categorical Cross-Entropy  
- Metrics: Accuracy  

---

## 📊 Evaluation  

### **Metrics**  
The models are evaluated using the following metrics:  
- **Accuracy** – Measures overall prediction correctness.  
- **Precision** – Measures the proportion of correctly predicted shoplifting cases.  
- **Recall** – Measures the proportion of actual shoplifting cases that were correctly predicted.  
- **F1-Score** – Harmonic mean of Precision and Recall.  


---

## 🚀 Prediction Pipeline  

### **Pipeline Overview**  
The prediction pipeline involves the following steps:  
1. **Load Model** – Load the trained model.  
2. **Extract Frames** – Convert the input video into frames.  
3. **Preprocess Frames** – Resize, pad, and normalize the frames.  
4. **Predict** – Pass the frames through the model for prediction.  
5. **Output** – Return the classification result (shoplifting or not).  

---

## 📌 Usage  

### **Prepare the Data**  
1. Place the video files in the `data/` directory.  
2. Run the preprocessing script to extract frames and normalize them.  

### **Train the Model**  
1. Load the training script.  
2. Adjust the hyperparameters if needed.  
3. Start training and monitor the performance.  

### **Evaluate the Model**  
1. Run the evaluation script.  
2. Review the output metrics (accuracy, precision, recall).  

### **Run Predictions**  
1. Load the trained model.  
2. Pass the video file to the prediction script.  
3. Get the output prediction.  

---

## ✅ Conclusion  
This project demonstrates how deep learning models, including I3D, CNN + BiLSTM, and Hybrid 3D CNN + LSTM, can be used to detect shoplifting activities from video data. By combining spatiotemporal feature extraction and sequential learning, the models achieve high accuracy in identifying suspicious activities.  

---

## 📎 Acknowledgments  
- TensorFlow  
- Keras  
- OpenCV  
