import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import speech_recognition as sr
import onnxruntime as ort
import typing

# === AnimeGAN Tanımı ===
class AnimeGAN:
    def __init__(self, model_path: str = '', downsize_ratio: float = 1.0):
        if not os.path.exists(model_path):
            raise Exception(f"Model bulunamadı: {model_path}")
        
        self.downsize_ratio = downsize_ratio
        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']
        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    def to_32s(self, x):
        return 256 if x < 256 else x - x % 32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]
        if x32:
            frame = cv2.resize(frame, (self.to_32s(int(w * self.downsize_ratio)), self.to_32s(int(h * self.downsize_ratio))))
        frame = frame.astype(np.float32) / 127.5 - 1.0
        return frame

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        frame = (frame.squeeze() + 1.) / 2 * 255
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (wh[0], wh[1]))
        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        image = self.process_frame(frame)
        outputs = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(image, axis=0)})
        return self.post_process(outputs[0], frame.shape[:2][::-1])


# === Veri ve Model Yolları ===
json_path = r"output_data_updated (1).json"
image_folder = r"OutputImages\OutputImages"
animegan_model_path = r"Hayao_64.onnx"  # AnimeGAN onnx model dosyası

# === Dataset Sınıfı ===
class SignLanguageDataset(Dataset):
    def __init__(self, json_path, image_size=(64, 64)):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.image_paths = []
        self.labels = []
        self.label_map = {word: idx for idx, word in enumerate(self.data.keys())}

        for word, frames in self.data.items():
            for frame in frames:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        img = img / 255.0
                        self.image_paths.append(img)
                        self.labels.append(self.label_map[word])
                    else:
                        print(f"Resim yüklenemedi: {img_path}")
                else:
                    print(f"Dosya yok: {img_path}")

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[idx]
        label = self.labels[idx]
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# === ANN Model Tanımı ===
class ANNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# === Eğitim ve Veri Yükleme ===
dataset = SignLanguageDataset(json_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_size = 3 * 64 * 64
num_classes = len(dataset.label_map)
model = ANNModel(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'ann_sign_language_model.pth')

# === Modeli Yükleme ===
model.load_state_dict(torch.load('ann_sign_language_model.pth'))
model.eval()

# === AnimeGAN Nesnesi ===
animegan = AnimeGAN(animegan_model_path)

# === Video Üretimi ===
def generate_video(sentence, model, label_map, data, output_video="output_video_anime_update2.mp4", fps=60):
    words = sentence.lower().split()
    frame_list = []
    resolution = (256, 256)

    for word in words:
        if word in data:
            for frame in data[word]:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, resolution)
                        img = animegan(img)
                        frame_list.append(img)

    if not frame_list:
        print("Hiçbir kare bulunamadı.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, resolution)

    for frame in frame_list:
        out.write(frame)
    out.release()
    print(f"Video olusturuldu: {output_video}")

# === Sesli Cümle Alma ===
def get_sentence_from_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Cumle bekleniyor...")
        audio = recognizer.listen(source)
        try:
            sentence = recognizer.recognize_google(audio, language="tr-TR").lower()
            print(f"Algılanan cümle: {sentence}")
            return sentence
        except sr.UnknownValueError:
            print("Anlaşılamadı.")
        except sr.RequestError as e:
            print(f"API hatası: {e}")
    return None

# === Çalıştırma ===
sentence = get_sentence_from_audio()
if sentence:
    generate_video(sentence, model, dataset.label_map, dataset.data)
