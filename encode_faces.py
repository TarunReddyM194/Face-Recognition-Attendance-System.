import os
import pickle
import face_recognition

IMAGE_FOLDER = "images"
OUTPUT_FILE = "encodings/encodings.pkl"

encodings_list = []
names = []

for file in os.listdir(IMAGE_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_FOLDER, file)
        name = os.path.splitext(file)[0]   # filename without extension

        print("Processing:", file)

        img = face_recognition.load_image_file(path)
        boxes = face_recognition.face_locations(img, model="hog")

        if len(boxes) == 0:
            print("No face found in:", file)
            continue

        enc = face_recognition.face_encodings(img, boxes)[0]
        encodings_list.append(enc)
        names.append(name)

data = {"encodings": encodings_list, "names": names}

os.makedirs("encodings", exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("Encoding completed!")
print("Saved encodings for:", names)
