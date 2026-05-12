import cv2
import torch
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageTk


# ==========================================
# 1. loading the pre-trained models and building the database of mean embeddings if not already saved
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Running on device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

database = {}
db_file = 'database.pt'
dataset_path = 'images_recognition'

if os.path.exists(db_file):
    print("Loading pre-calculated database...")
    database = torch.load(db_file, map_location=device)
else:
    print("Building database with Prototypical Mean Embeddings...")
    if os.path.exists(dataset_path):
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            embeddings = [] 
            
            if os.path.isdir(person_folder):
                for image_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, image_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        faces = mtcnn(img_rgb)
                        if faces is not None:
                            
                            emb = resnet(faces[0].unsqueeze(0).to(device))
                            embeddings.append(emb.detach().cpu())
                
                
                if len(embeddings) > 0:
                    mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
                    database[person_name] = mean_embedding
                    print(f"Processed {person_name}: {len(embeddings)} images averaged.")
                    
        torch.save(database, db_file)
        print("Database built and saved to database.pt!")
    else:
        print("Warning: Dataset folder not found.")

# ==========================================
# 2. building the GUI and controlling the display
# ==========================================
class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Team Face Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e272e")

        self.vid = None
        self.is_running = False
        self.threshold = 0.87

        # buttons frame
        btn_frame = tk.Frame(root, bg="#1e272e")
        btn_frame.pack(pady=20)

        style = {"font": ("Arial", 12, "bold"), "bg": "#0fb9b1", "fg": "white", "width": 15, "padx": 10, "pady": 5}
        
        tk.Button(btn_frame, text="Load Image", command=self.open_image, **style).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Load Video", command=self.open_video, **style).grid(row=0, column=1, padx=10)
        tk.Button(btn_frame, text="Start Webcam", command=self.open_webcam, **style).grid(row=0, column=2, padx=10)
        tk.Button(btn_frame, text="🛑 Stop Media", command=self.stop_media, bg="#eb3b5a", fg="white", font=("Arial", 12, "bold"), width=15).grid(row=0, column=3, padx=10)

        # bottom part (display screen)
        self.canvas_label = tk.Label(root, bg="black")
        self.canvas_label.pack(expand=True, fill="both", padx=20, pady=10)

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        faces = mtcnn(img_rgb)

        if boxes is not None and faces is not None:
            for i, face in enumerate(faces):
                x, y, x2, y2 = boxes[i].astype(int)
                curr_emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
                
                min_dist = self.threshold
                name = "Unknown"

                for person, db_emb in database.items():
                    dist = (db_emb.to(device) - curr_emb.to(device)).norm().item()
                    if dist < min_dist:
                        min_dist = dist
                        name = person

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img_rgb, (x, y), (x2, y2), color, 2)
                cv2.putText(img_rgb, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img_rgb

    def show_frame_on_gui(self, img_rgb):

        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((900, 550), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.canvas_label.imgtk = imgtk
        self.canvas_label.configure(image=imgtk)

    def open_image(self):
        self.stop_media()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path: return
        
        frame = cv2.imread(file_path)
        if frame is not None:
            result_rgb = self.process_frame(frame)
            self.show_frame_on_gui(result_rgb)

    def open_video(self):
        self.stop_media()
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if not file_path: return
        
        self.vid = cv2.VideoCapture(file_path)
        self.is_running = True
        self.update_video_loop()

    def open_webcam(self):
        self.stop_media()
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        self.is_running = True
        self.update_video_loop()

    def update_video_loop(self):
        if self.is_running and self.vid is not None:

            skip_frames = 2 
            for _ in range(skip_frames):
                self.vid.read()
                
            ret, frame = self.vid.read()
            
            if ret:
                result_rgb = self.process_frame(frame)
                self.show_frame_on_gui(result_rgb)
                self.root.after(1, self.update_video_loop)
            else:
                self.stop_media()

    def stop_media(self):
        self.is_running = False
        if self.vid is not None:
            self.vid.release()
            self.vid = None
        self.canvas_label.configure(image='')


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()