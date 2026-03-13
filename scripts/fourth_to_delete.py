import torch
import os

def load_silero_model(model_path=None):
    if model_path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
        model_path = os.path.join(ROOT_DIR, "models", "base", "silero", "silero_v4_ru.pt")
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    
    device = torch.device('cpu')
    model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
    model.to(device)
    return model

_MODEL_CACHE = None

def generate_speech(text, speaker='eugene', sample_rate=48000, model_path=None):
    """
    Генерирует речь из текста и возвращает аудио
    
    Args:
        text (str): Текст на русском для озвучивания
        speaker (str): Голос ('eugene', 'aidar', 'baya', 'kseniya', 'xenia')
        sample_rate (int): Частота дискретизации
        model_path (str): Путь к файлу модели (если None, используется стандартный)
    
    Returns:
        torch.Tensor: Аудио массив
    """
    global _MODEL_CACHE
    
    if not text or not text.strip():
        raise ValueError("Текст не может быть пустым")
    
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_silero_model(model_path)
        print("Доступные голоса:", _MODEL_CACHE.speakers)
    
    audio = _MODEL_CACHE.apply_tts(
        text=text.strip(),
        speaker=speaker,
        sample_rate=sample_rate
    )
    
    return audio

if __name__ == "__main__":
    import sounddevice as sd
    
    # Просто вызываем функцию с текстом
    text = input("Введите текст на русском: ").strip()
    
    try:
        # Получаем аудио
        audio = generate_speech(text)
        
        # Воспроизводим (по желанию)
        print("▶️ Воспроизведение...")
        sd.play(audio, 48000)
        sd.wait()
        print("✅ Готово!")
        
        # Можно сохранить или обработать аудио дальше
        # import soundfile as sf
        # sf.write("output.wav", audio, 48000)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")