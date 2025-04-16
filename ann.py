import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import speech_recognition as sr  # SpeechRecognition kütüphanesi

# Veri yolları
json_path = r"output_data (1).json"  # JSON dosyasının yolu
image_folder = r"OutputImages\OutputImages"  # Görsellerin bulunduğu klasör yolu

# Veri setini yükleme
class SignLanguageDataset(Dataset):
    def __init__(self, json_path, image_size=(64, 64)):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.image_paths = []
        self.labels = []
        self.label_map = {word: idx for idx, word in enumerate(self.data.keys())}  # Kelime - ID eşlemesi

        for word, frames in self.data.items():
            for frame in frames:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, image_size)  # Görselleri aynı boyuta getir
                        img = img / 255.0  # Normalizasyon
                        self.image_paths.append(img)
                        self.labels.append(self.label_map[word])
                    else:
                        print(f"Resim yüklenemedi: {img_path}")
                else:
                    print(f"Dosya mevcut değil: {img_path}")

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[idx]
        label = self.labels[idx]
        img = np.transpose(img, (2, 0, 1))  # PyTorch tensor formatına çevirme (C, H, W)
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


# Veri yükleme
dataset = SignLanguageDataset(json_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# CNN Modeli
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Model oluşturma
num_classes = len(dataset.label_map)
model = CNNModel(num_classes)

# Eğitim için kriter ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model eğitme
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Modeli kaydetme
torch.save(model.state_dict(), 'sign_language_model.pth')


# Modeli yükleme
model = CNNModel(num_classes)
model.load_state_dict(torch.load('sign_language_model.pth'))
model.eval()


def predict_word(image_path, model, label_map, image_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Resim yüklenemedi: {image_path}")
        return None

    img = cv2.resize(img, image_size) / 255.0
    img = np.transpose(img, (2, 0, 1))  # PyTorch tensor formatına çevirme (C, H, W)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Batç alanı ekle

    with torch.no_grad():
        outputs = model(img)
        _, predicted_label = torch.max(outputs, 1)

    for word, idx in label_map.items():
        if idx == predicted_label.item():
            return word
    return None


def generate_video(sentence, model, label_map, data, output_video="output_video1.mp4", fps=60):
    words = sentence.lower().split()
    frame_list = []
    target_resolution = (256, 256)  # Daha yüksek çözünürlük

    for word in words:
        if word in data:
            for frame in data[word]:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_CUBIC)
                        frame_list.append(img)
                    else:
                        print(f"Resim yüklenemedi: {img_path}")
                else:
                    print(f"Dosya mevcut değil: {img_path}")

    if not frame_list:
        print("Hiçbir kare eklenemedi!")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Daha yüksek kalite için codec değiştirildi
    out = cv2.VideoWriter(output_video, fourcc, fps, target_resolution)

    for frame in frame_list:
        out.write(frame)

    out.release()
    print(f"Video kaydedildi: {output_video}")


# SpeechRecognition kullanarak sesle cümle almak
def get_sentence_from_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Sesinizi bekliyorum...")
        audio = recognizer.listen(source)
        try:
            sentence = recognizer.recognize_google(audio, language="tr-TR")
            sentence=sentence.lower()
            print(f"Sesli girilen cümle: {sentence}")
            return sentence
        except sr.UnknownValueError:
            print("Anlaşılamadı, lütfen tekrar edin.")
        except sr.RequestError as e:
            print(f"Google API hatası: {e}")
    return None


# Sesle alınan cümleyi işaret dili video formatına çevir
sentence = get_sentence_from_audio()
if sentence:
    generate_video(sentence, model, dataset.label_map, dataset.data)
