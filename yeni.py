# Gerekli kütüphaneler içe aktarılıyor
import os  # Dosya ve dizin işlemleri için
import json  # JSON dosyalarını okumak ve yazmak için
import cv2  # Görüntü işleme işlemleri için OpenCV
import numpy as np  # Sayısal işlemler ve diziler için NumPy
import torch  # PyTorch derin öğrenme kütüphanesi
import torch.nn as nn  # Sinir ağı modülleri için
import torch.optim as optim  # Optimizasyon algoritmaları için
from torch.utils.data import DataLoader, Dataset  # Veri yükleyici ve özel veri kümesi sınıfı
import speech_recognition as sr  # Ses tanıma işlemleri için
import onnxruntime as ort  # ONNX modelini çalıştırmak için
import typing  # Tip tanımlamaları için (örneğin Tuple)

# === AnimeGAN Tanımı ===
# Anime stilinde görüntü oluşturmak için kullanılan AnimeGAN sınıfı tanımlanıyor
class AnimeGAN:
    # Sınıfın kurucu fonksiyonu (init), model yolu ve boyut küçültme oranı alıyor
    def __init__(self, model_path: str = '', downsize_ratio: float = 1.0):
        # Model dosyasının var olup olmadığı kontrol ediliyor
        if not os.path.exists(model_path):
            raise Exception(f"Model bulunamadı: {model_path}")
        
        # Görüntüyü küçültmek için kullanılacak oran atanıyor
        self.downsize_ratio = downsize_ratio
        
        # Modelin çalışacağı cihaz belirleniyor (GPU varsa GPU, yoksa CPU)
        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']
        
        # ONNX model oturumu başlatılıyor
        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    # Görüntü boyutlarını 32'nin katına yuvarlayan yardımcı fonksiyon
    def to_32s(self, x):
        return 256 if x < 256 else x - x % 32

    # Girdi görüntüsünü model için işleyen fonksiyon
    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]  # Yükseklik ve genişlik alınır
        if x32:
            frame = cv2.resize(frame, (self.to_32s(int(w * self.downsize_ratio)), self.to_32s(int(h * self.downsize_ratio))))
        frame = frame.astype(np.float32) / 127.5 - 1.0  # Görüntü verisi -1 ile 1 aralığına ölçeklenir
        return frame

    # Model çıktılarını işleyerek tekrar orijinal formata dönüştürür
    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        frame = (frame.squeeze() + 1.) / 2 * 255  # -1 ile 1 aralığından 0-255'e normalize edilir
        frame = frame.astype(np.uint8)  # Görüntü verisi uint8 formatına çevrilir
        frame = cv2.resize(frame, (wh[0], wh[1]))  # Görüntü orijinal boyuta getirilir
        return frame

    # Sınıfın çağrıldığında çalışacak ana fonksiyonu
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        image = self.process_frame(frame)  # Girdi görüntüsü işlenir
        outputs = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(image, axis=0)})  # Model çalıştırılır
        return self.post_process(outputs[0], frame.shape[:2][::-1])  # Çıktı işlenip geri döndürülür

# === Veri ve Model Yolları ===
# JSON veri dosyasının ve resimlerin yolu tanımlanıyor
json_path = r"output_data_updated (1).json"
image_folder = r"OutputImages\OutputImages"
animegan_model_path = r"Hayao_64.onnx"  # ONNX model dosyasının yolu

# === Dataset Sınıfı ===
# İşaret dili veri kümesi için özel bir PyTorch Dataset sınıfı
class SignLanguageDataset(Dataset):
    # Kurucu fonksiyon; JSON dosyasını yükler ve verileri hazırlar
    def __init__(self, json_path, image_size=(64, 64)):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)  # JSON verisi yüklenir
        
        self.image_paths = []  # İşlenecek görüntülerin listesi
        self.labels = []  # Görüntülere karşılık gelen etiketler
        self.label_map = {word: idx for idx, word in enumerate(self.data.keys())}  # Kelimelere sayısal etiket ataması

        for word, frames in self.data.items():  # Her kelime ve ilgili kareleri döner
            for frame in frames:  # Her kareyi işler
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))  # Resim yolu oluşturulur
                if os.path.exists(img_path):  # Dosya mevcutsa
                    img = cv2.imread(img_path)  # Görüntü okunur
                    if img is not None:  # Görüntü başarıyla yüklendiyse
                        img = cv2.resize(img, image_size)  # Yeniden boyutlandırılır
                        img = img / 255.0  # 0-1 aralığına normalize edilir
                        self.image_paths.append(img)  # Görüntü listeye eklenir
                        self.labels.append(self.label_map[word])  # Etiket listeye eklenir
                    else:
                        print(f"Resim yüklenemedi: {img_path}")
                else:
                    print(f"Dosya yok: {img_path}")

        self.image_paths = np.array(self.image_paths)  # Görüntü listesi NumPy dizisine çevrilir
        self.labels = np.array(self.labels)  # Etiketler NumPy dizisine çevrilir

    def __len__(self):
        return len(self.image_paths)  # Veri kümesinin uzunluğu

    def __getitem__(self, idx):
        img = self.image_paths[idx]  # Belirtilen indeksdeki görüntü alınır
        label = self.labels[idx]  # Etiketi alınır
        img = np.transpose(img, (2, 0, 1))  # Kanal sırası PyTorch için (C, H, W) olacak şekilde düzenlenir
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  # Tensör olarak döndürülür

# === Model Tanımı ===
# CNN (Convolutional Neural Network) tabanlı model tanımlanıyor
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()  # Üst sınıfın init fonksiyonu çağrılır
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # İlk konvolüsyon katmanı (3 kanal giriş, 32 filtre)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # İkinci konvolüsyon katmanı
        self.pool = nn.MaxPool2d(2, 2)  # Havuzlama katmanı (2x2 boyutunda)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Tam bağlantılı katman
        self.fc2 = nn.Linear(128, num_classes)  # Çıkış katmanı, sınıf sayısı kadar nöron
        self.dropout = nn.Dropout(0.5)  # Aşırı öğrenmeyi engellemek için dropout uygulanır

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # İlk konvolüsyon ve aktivasyon + havuzlama
        x = self.pool(torch.relu(self.conv2(x)))  # İkinci konvolüsyon ve aktivasyon + havuzlama
        x = x.view(-1, 64 * 16 * 16)  # Tensör düzleştirilir (flatten)
        x = torch.relu(self.fc1(x))  # Tam bağlantılı katman + aktivasyon
        x = self.dropout(x)  # Dropout uygulanır
        return self.fc2(x)  # Sonuç döndürülür

# === Eğitim ve Veri Yükleme ===
dataset = SignLanguageDataset(json_path)  # Veri kümesi oluşturulur
train_size = int(0.8 * len(dataset))  # %80 eğitim, %20 test verisi olarak ayrılır
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # Veri kümesi bölünür
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Eğitim verisi için yükleyici
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Test verisi için yükleyici

num_classes = len(dataset.label_map)  # Sınıf sayısı veri kümesinden alınır
model = CNNModel(num_classes)  # Model oluşturulur

criterion = nn.CrossEntropyLoss()  # Kayıp fonksiyonu (çok sınıflı sınıflandırma için)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizasyon algoritması

# Eğitim döngüsü başlatılıyor
for epoch in range(10):  # 10 dönem boyunca model eğitilecek
    model.train()  # Eğitim moduna alınır
    total_loss = 0  # Toplam kayıp sıfırlanır
    for inputs, labels in train_loader:  # Tüm eğitim verileri döner
        optimizer.zero_grad()  # Gradyanlar sıfırlanır
        outputs = model(inputs)  # Model tahmin yapar
        loss = criterion(outputs, labels)  # Kayıp hesaplanır
        loss.backward()  # Geri yayılım yapılır
        optimizer.step()  # Ağırlıklar güncellenir
        total_loss += loss.item()  # Kayıp değeri toplanır
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")  # Epoch sonunda ortalama kayıp yazdırılır

torch.save(model.state_dict(), 'cnn_sign_language_model.pth')  # Model dosyası kaydedilir

# === Modeli Yükleme ===
model.load_state_dict(torch.load('cnn_sign_language_model.pth'))  # Eğitimli model geri yüklenir
model.eval()  # Değerlendirme moduna alınır

# === AnimeGAN Nesnesi ===
animegan = AnimeGAN(animegan_model_path)  # AnimeGAN modeli başlatılır

# === Video Üretimi ===
# Verilen bir cümleyi işaret dili kareleriyle videoya çeviren fonksiyon
def generate_video(sentence, model, label_map, data, output_video="output_video_anime_update3.mp4", fps=60):
    words = sentence.lower().split()  # Cümle küçük harfe çevrilip kelimelere ayrılır
    frame_list = []  # Video karelerini tutacak liste
    resolution = (256, 256)  # Video boyutu

    for word in words:  # Her kelime için
        if word in data:  # Veri içinde varsa
            for frame in data[word]:  # O kelimeye ait her kare için
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))  # Görüntü yolu
                if os.path.exists(img_path):  # Görüntü mevcutsa
                    img = cv2.imread(img_path)  # Görüntü yüklenir
                    if img is not None:  # Görüntü başarıyla yüklendiyse
                        img = cv2.resize(img, resolution)  # Yeniden boyutlandırılır
                        img = animegan(img)  # AnimeGAN filtresi uygulanır
                        frame_list.append(img)  # Kare listeye eklenir

    if not frame_list:  # Hiç kare yoksa uyarı verilir
        print("Hiçbir kare bulunamadı.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video formatı belirlenir
    out = cv2.VideoWriter(output_video, fourcc, fps, resolution)  # Video yazıcısı başlatılır

    for frame in frame_list:  # Her kare yazılır
        out.write(frame)
    out.release()  # Video dosyası kaydedilir ve kapatılır
    print(f"Video olusturuldu: {output_video}")  # Kullanıcı bilgilendirilir

# === Sesli Cümle Alma ===
# Mikrofon aracılığıyla Türkçe sesli cümle tanıma fonksiyonu
def get_sentence_from_audio():
    recognizer = sr.Recognizer()  # Ses tanıma nesnesi oluşturulur
    with sr.Microphone() as source:  # Mikrofon girişi başlatılır
        print("Cumle bekleniyor...")  # Kullanıcıya beklemesi söylenir
        audio = recognizer.listen(source)  # Ses kaydedilir
        try:
            sentence = recognizer.recognize_google(audio, language="tr-TR").lower()  # Google API ile ses tanınır
            print(f"Algılanan cümle: {sentence}")  # Tanınan cümle yazdırılır
            return sentence  # Cümle geri döndürülür
        except sr.UnknownValueError:
            print("Anlaşılamadı.")  # Ses tanınamazsa
        except sr.RequestError as e:
            print(f"API hatası: {e}")  # API hatası olursa
    return None  # Tanıma başarısızsa None döndürülür

# === Çalıştırma ===
sentence = get_sentence_from_audio()  # Sesli cümle alınır
if sentence:  # Cümle başarıyla tanındıysa
    generate_video(sentence, model, dataset.label_map, dataset.data)  # Video üretilir
