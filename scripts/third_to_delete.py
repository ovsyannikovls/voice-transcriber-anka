import os

import json
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import queue

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from fourth_to_delete import generate_speech


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
vosk_model_path = os.path.join(ROOT_DIR, "models", "base", "vosk-model-small-ru-0.22")
qwen_model_path = os.path.join(ROOT_DIR, "models", "base", "qwen2-2B")

tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    qwen_model_path,
    dtype=torch.float16,
    local_files_only=True
)

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

gen_kwargs = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": model.config.pad_token_id
}


def ai_question(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


q = queue.Queue()


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def clear_audio_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def main():
    if not os.path.exists(vosk_model_path):
        print("Модель не найдена по пути:", vosk_model_path)
        return

    model = Model(vosk_model_path)
    rec = KaldiRecognizer(model, 16000)
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
                    audio = generate_speech(ai_question(text))
                    sd.play(audio, 48000)
                    sd.wait()
                    print("AI говорит: ...")
                    clear_audio_queue(q)
            else:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Остановлено.")