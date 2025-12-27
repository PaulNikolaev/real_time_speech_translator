"""
Модуль для распознавания речи с использованием faster-whisper.
"""

from faster_whisper import WhisperModel
from config import DEVICE, WHISPER_MODELS_DIR


class SpeechRecognizer:
    """Класс для распознавания речи на русском языке."""

    def __init__(self, model_size: str = "base"):
        """Инициализирует распознаватель речи.

        Args:
            model_size: Размер модели Whisper ('tiny', 'base', 'small', 'medium', 'large').
        """
        print("Загрузка модели faster-whisper для распознавания речи...")
        print(f"Модели Whisper будут сохранены в: {WHISPER_MODELS_DIR}")

        self.model_size = model_size
        device = "cuda" if DEVICE == "cuda" else "cpu"
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    compute_capability = torch.cuda.get_device_capability(0)
                    major_capability = compute_capability[0]
                    if major_capability >= 7:
                        compute_type = "float16"
                    else:
                        compute_type = "float32"
                        print(f"ℹ GPU имеет compute capability {major_capability}.{compute_capability[1]}, используется float32")
                else:
                    compute_type = "float32"
            except Exception as e:
                print(f"⚠ Не удалось определить compute capability GPU: {e}. Используется float32")
                compute_type = "float32"
        else:
            compute_type = "float32"

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=WHISPER_MODELS_DIR
            )
            print("✓ Модель faster-whisper загружена!")
        except Exception as e:
            error_msg = str(e).lower()
            if device == "cuda" and ("cublas" in error_msg or "cudnn" in error_msg or "dll" in error_msg or "cuda" in error_msg):
                print(f"⚠ Ошибка при загрузке модели на GPU: {e}")
                print("ℹ Переключаемся на CPU для faster-whisper")
                self.model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="float32",
                    download_root=WHISPER_MODELS_DIR
                )
                print("✓ Модель faster-whisper загружена на CPU!")
            else:
                raise

    def recognize(self, audio_path: str, language: str = "ru") -> str:
        """Распознает речь в аудиофайле.

        Args:
            audio_path: Путь к аудиофайлу.
            language: Код языка для распознавания (по умолчанию 'ru').

        Returns:
            str: Распознанный текст.
        """
        print("\nРаспознавание речи...")

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5
            )
            recognized_text = " ".join([segment.text for segment in segments]).strip()

            print(f"Распознанный язык: {info.language}")
            print(f"Распознанный текст: {recognized_text}")

            return recognized_text
        except Exception as e:
            error_msg = str(e).lower()
            if "cublas" in error_msg or "cudnn" in error_msg or "dll" in error_msg:
                print(f"⚠ Ошибка CUDA при распознавании: {e}")
                print("ℹ Перезагружаем модель на CPU...")
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="float32",
                    download_root=WHISPER_MODELS_DIR
                )
                segments, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5
                )
                recognized_text = " ".join([segment.text for segment in segments]).strip()
                print(f"Распознанный язык: {info.language}")
                print(f"Распознанный текст: {recognized_text}")
                return recognized_text
            else:
                raise
