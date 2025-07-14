import cv2
import os
import numpy as np

# Initialize Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to save faces
data_dir = "face_data"
os.makedirs(data_dir, exist_ok=True)

# Step 1: Capture face samples for a person
def collect_face_samples():
    person_name = input("Enter name to register: ").strip()
    person_path = os.path.join(data_dir, person_name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("[INFO] Capturing face samples. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Collected {count} images for '{person_name}'")

# Step 2: Train the recognizer
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(current_label)

        label_map[current_label] = person_name
        current_label += 1

    recognizer.train(faces, np.array(labels))
    print("[INFO] Training completed.")
    return recognizer, label_map

# Step 3: Live recognition
def recognize_faces(recognizer, label_map):
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting live face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)
            name = label_map.get(label, "Unknown")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====== RUN SEQUENCE ======
if __name__ == "__main__":

    print("\n1. Register new face")
    print("2. Start face recognition")
    choice = input("Select option (1 or 2): ")

    if choice == '1':
        collect_face_samples()
    elif choice == '2':
        recognizer, label_map = train_model()
        recognize_faces(recognizer, label_map)
    else:
        print("❌ Invalid choice.")