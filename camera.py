import cv2
from mtcnn import MTCNN
import torch
import torch.nn as nn
from torch.serialization import safe_globals

# ---------------- SETUP ---------------- #

detector = MTCNN()
cap = cv2.VideoCapture(0)

labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# ---------------- MODEL ---------------- #

class Emotion_detection(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

# ---------------- LOAD MODEL (FINAL FIX) ---------------- #

with safe_globals({"Emotion_detection": Emotion_detection}):
    model = torch.load(
        r"E:\face emotion\emotion.pth",
        map_location=torch.device('cpu'),
        weights_only=False
    )

model.eval()

# ---------------- LOOP ---------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(frame_rgb)
    except:
        continue

    for face in faces:
        x, y, w, h = face['box']

        # -------- SAFE CROPPING -------- #
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        face_img = frame[y:y2, x:x2]

        if face_img.size == 0:
            continue

        # -------- PREPROCESS -------- #
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0
        face_img = (face_img - 0.5) / 0.5

        face_img = torch.tensor(face_img).float()
        face_img = face_img.permute(2, 0, 1)
        face_img = face_img.unsqueeze(0)

        # -------- PREDICTION -------- #
        with torch.no_grad():
            output = model(face_img)
            pred = torch.argmax(output, dim=1).item()

        emotion = labels[pred]

        # -------- DRAW -------- #
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ---------------- #

cap.release()
cv2.destroyAllWindows()