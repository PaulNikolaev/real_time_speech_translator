"""
Конфигурация приложения.
Загружает настройки из переменных окружения.
"""

import os
import torch
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Hugging Face токен (опционально, не обязателен для публичных моделей)
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Константы из .env или значения по умолчанию
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
RECORD_DURATION = int(os.getenv("RECORD_DURATION", "5"))
OUTPUT_DIR = "temp_audio"

# Папка для хранения моделей (кэш)
MODELS_DIR = os.getenv("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Подпапки для разных типов моделей
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "whisper")
HF_MODELS_DIR = os.path.join(MODELS_DIR, "huggingface")

# Создание директорий для моделей
os.makedirs(WHISPER_MODELS_DIR, exist_ok=True)
os.makedirs(HF_MODELS_DIR, exist_ok=True)

# Установка переменных окружения для кэширования моделей
os.environ["WHISPER_CACHE_DIR"] = WHISPER_MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_MODELS_DIR
os.environ["HF_HOME"] = HF_MODELS_DIR

# Определение устройства из .env или автоматически
DEVICE_ENV = os.getenv("DEVICE", "").lower()
if DEVICE_ENV in ["cuda", "cpu"]:
    DEVICE = DEVICE_ENV if (DEVICE_ENV == "cuda" and torch.cuda.is_available()) else "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Функция для вывода информации о GPU (вызывается в main.py)
def print_gpu_info():
    """Выводит информацию о GPU при запуске."""
    if torch.cuda.is_available():
        print(f"✓ CUDA доступна: {torch.version.cuda}")
        print(f"✓ PyTorch версия: {torch.__version__}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ VRAM: {vram_gb:.2f} GB")

        # Предупреждение о VRAM для карт с < 8 GB
        if vram_gb < 8:
            print(f"⚠ Внимание: {vram_gb:.1f} GB VRAM может быть недостаточно для одновременной загрузки всех моделей.")
            print("  Модели будут работать, но может потребоваться больше времени.")
        elif vram_gb < 10:
            print(f"ℹ {vram_gb:.1f} GB VRAM - достаточно для работы, но может быть тесно при одновременной загрузке всех моделей.")
    else:
        print("ℹ CUDA недоступна, будет использоваться CPU")
        print("  Приложение полностью работает на CPU, но обработка будет медленнее.")
        print("  Ожидаемая задержка: ~12-20 секунд (вместо ~5-8 секунд на GPU).")
        print("  Все компоненты (faster-whisper, NLLB, Bark-small) поддерживают CPU.")

# Создание директории для временных файлов
os.makedirs(OUTPUT_DIR, exist_ok=True)
