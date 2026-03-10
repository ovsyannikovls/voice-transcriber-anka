import json
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import queue
import os

model_path = "/models/base/vosk-model-small-ru-0.22"
q = queue.Queue()

def callback(indata, frames, time, status):
    """Это будет вызываться для каждого блока звука с микрофона"""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def main():
    if not os.path.exists(model_path):
        print("Модель не найдена по пути:", model_path)
        return

    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)  # 16000 Hz - частота дискретизации

    with sd.RawInputStream(samplerate=16000, blocksize=8000, device=None,
                           dtype='int16', channels=1, callback=callback):
        print("Слушаю... Нажмите Ctrl+C для остановки.")

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get('text', '')
                if text:
                    print(f"Распознано: {text}")
                    # Здесь можно дописать логику: запись в файл, отправка по сети и т.д.
                    # with open("/home/ваш_пользователь/transcript.txt", "a") as f:
                    #     f.write(text + "\n")
            else:
                pass

if main == "main":
    try:
        main()
    except KeyboardInterrupt:
        print("Остановлено.")