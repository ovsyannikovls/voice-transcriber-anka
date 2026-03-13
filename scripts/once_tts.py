import torch
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
save_dir = os.path.join(ROOT_DIR, "models", "base", "silero")
os.makedirs(save_dir, exist_ok=True)

print("Скачивание модели Silero с голосом Евгения...")

model_url = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
save_path = os.path.join(save_dir, "silero_v4_ru.pt")

# Скачиваем файл
response = requests.get(model_url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(save_path, 'wb') as f:
    downloaded = 0
    for data in response.iter_content(chunk_size=8192):
        f.write(data)
        downloaded += len(data)
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            print(f"Прогресс: {percent:.1f}%", end='\r')

size_mb = os.path.getsize(save_path) / (1024 * 1024)
print(f"\n✅ Модель сохранена в {save_path}")
print(f"📁 Размер файла: {size_mb:.2f} MB")