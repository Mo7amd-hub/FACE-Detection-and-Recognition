# 🧠 Face Recognition System

A real-time face recognition system built using deep learning.
The system detects faces from a live camera feed and identifies individuals by comparing facial embeddings.

---

## 🚀 Features

* Real-time face detection using MTCNN
* Face recognition using a pre-trained deep learning model (FaceNet)
* Supports multiple faces in a single frame
* Uses embedding comparison for accurate identification
* Improved accuracy using **mean embedding per person**

---

## 🧩 How It Works

1. **Face Detection**

   * Detects faces in the image using MTCNN

2. **Face Alignment & Cropping**

   * Extracts and aligns the face for consistency

3. **Feature Extraction**

   * Uses a deep learning model to convert each face into a **512-dimensional embedding**

4. **Database Creation**

   * For each person:

     * Multiple images are processed
     * Embeddings are generated
     * The **mean embedding** is calculated and stored

5. **Recognition**

   * For each detected face:

     * Generate embedding
     * Compare with stored embeddings using Euclidean distance
     * If distance < threshold → Recognized
     * Else → Unknown

---

## 📁 Project Structure

```
Face-Recognition/
│
├── images/
│   ├── Person1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │
│   ├── Person2/
│       ├── img1.jpg
│       ├── img2.jpg
│
├── main.py
├── README.md
```

---

## ⚙️ Installation

```bash
pip install torch torchvision
pip install facenet-pytorch
pip install opencv-python
```

---

## ▶️ Usage

1. Add images for each person inside the `images/` folder
2. Run the script:

```bash
python main.py
```

3. Press **Q** to exit

---

## 🎯 Key Concepts

* **Embedding**: A numerical representation of a face (512 values)
* **Euclidean Distance**: Measures similarity between faces
* **Threshold**: Determines if two faces belong to the same person
* **Batch Dimension**: Required input format for deep learning models
* **Mean Embedding**: Improves accuracy by averaging multiple images

---

## 📊 Improvements Implemented

* Used multiple images per person
* Computed **mean embedding** for better representation
* Reduced noise caused by lighting and pose variations

---

## 🧠 Future Improvements

* Use cosine similarity instead of Euclidean distance
* Save/load database instead of rebuilding every run
* Convert system into a web app (Flask / Streamlit)
* Add accuracy evaluation metrics

---

## ⭐ Notes

This project was developed as part of a university assignment and enhanced for better understanding of deep learning and computer vision concepts.
