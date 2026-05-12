# 🧠 Face Recognition System with GUI

A complete, real-time desktop application for face recognition built using deep learning and Python.
The system features a user-friendly Graphical User Interface (GUI) and can detect and identify individuals from static images, video files, and live webcam feeds.

---

## 🚀 Features

* **Interactive GUI**: Built with Tkinter for seamless user experience.
* **Multi-Media Support**: Recognize faces in **Images**, **Videos**, and **Webcam** feeds.
* **Smart Database Loading**: Calculates embeddings once and saves them to `database.pt` for zero-delay startup on future runs.
* **Prototypical Mean Embeddings**: Uses multiple images per person to calculate a robust "mean embedding," drastically improving accuracy against lighting and pose variations.
* **Optimized Video Playback**: Implements Frame Skipping techniques for real-time, smooth video processing without lag.
* **Real-time Face Detection & Recognition**: Powered by MTCNN and FaceNet (InceptionResnetV1).

---

## 🧩 How It Works

1. **Face Detection**
   * Detects faces and extracts bounding boxes using MTCNN.

2. **Feature Extraction**
   * Converts each cropped face into a **512-dimensional embedding** using a pre-trained ResNet model.

3. **Smart Database Creation (One-Time Setup)**
   * Iterates through the `images/` folder.
   * Calculates embeddings for all images of a person and computes their **Mean Embedding**.
   * Saves the database locally as `database.pt`. In future runs, this file is loaded instantly.

4. **Recognition & GUI Display**
   * Compares incoming faces with the stored database using **Euclidean distance**.
   * If distance < threshold (e.g., 0.87) → Draws a Green box with the person's name.
   * If distance > threshold → Draws a Red box labeled **Unknown**.
   * Converts OpenCV arrays to PIL images to display seamlessly on the Tkinter Canvas.

---

## 📁 Project Structure

```text
Face-Recognition/
│
├── images/                 # Dataset folder
│   ├── Member1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │
│   ├── Member2/
│       ├── img1.jpg
│
├── app.py                  # Main application script (GUI + Logic)
├── database.pt             # Auto-generated saved embeddings database
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

1. Add folders containing images for each team member inside the images/ directory.
2. Run the script:

```bash
python app.py
```

3. Use the GUI buttons to:

Load Image: Select an image from your device.

Load Video: Select a video file.

Start Webcam: Start live recognition.

Stop Media: Safely close the current media and clear the screen.

---

## 🎯 Key Concepts

* **Embedding**: A numerical representation of a face (512 values)
* **Euclidean Distance**: Measures similarity between faces
* **Threshold**: Determines if two faces belong to the same person
* **Batch Dimension**: Required input format for deep learning models
* **Mean Embedding**: Improves accuracy by averaging multiple images

---

## 📊 Improvements Implemented

* 🟢 Added Graphical User Interface (GUI) instead of command-line execution.
* 🟢 Saved Database State: System no longer rebuilds embeddings on every run (Implemented torch.save/load).
* 🟢 Mean Embedding: Improved accuracy by averaging multiple images per person.
* 🟢 Video Optimization: Handled inference latency in videos using frame-skipping.
* 🟢 Object-Oriented Design: Refactored the code into a clean FaceApp class structure.

---

## 🧠 Future Improvements

* Use Cosine Similarity instead of Euclidean distance for even more robust comparison.
* Add a "Register New Face" button directly inside the GUI using the webcam.
* Add accuracy evaluation metrics (e.g., Confusion Matrix) on a testing dataset.

---

## ⭐ Notes

This project was developed as part of a university assignment and progressively enhanced to include software engineering best practices, GUI development, and real-time computer vision optimizations.
