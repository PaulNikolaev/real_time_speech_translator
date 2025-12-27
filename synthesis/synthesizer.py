"""
Модуль для синтеза речи через Bark.
"""

import os
import traceback
import numpy as np
import soundfile as sf
import torch
from scipy import signal
from config import OUTPUT_DIR, DEVICE, HF_MODELS_DIR


class SpeechSynthesizer:
    """Класс для синтеза речи через Bark."""

    def __init__(self):
        """Инициализирует синтезатор речи с Bark."""
        print("Инициализация синтезатора речи...")
        print("Загрузка модели Bark для синтеза речи...")

        try:
            from transformers import BarkModel, AutoProcessor

            cache_dir = os.path.join(HF_MODELS_DIR, "transformers")
            os.makedirs(cache_dir, exist_ok=True)

            model_name = "suno/bark"

            print(f"Загрузка модели {model_name}...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model = BarkModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            ).to(DEVICE)

            print("✓ Модель Bark загружена!")
            print(f"✓ Модели будут сохранены в: {HF_MODELS_DIR}")
        except ImportError:
            raise ImportError(
                "Библиотека transformers не установлена или версия слишком старая. "
                "Установите: pip install transformers>=4.35.0"
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели Bark: {e}")

        self.bark_languages = {
            "en": "en",
            "fr": "fr"
        }

        print("Синтезатор речи готов!")

    def synthesize(self, text: str, target_lang: str = "fr",
                   output_filename: str = None, max_length: int = 250,
                   reference_audio_path: str = None) -> str:
        """Синтезирует речь на указанном языке.

        Args:
            text: Текст для синтеза.
            target_lang: Целевой язык ('en' для английского, 'fr' для французского).
            output_filename: Имя файла для сохранения (автоматически генерируется если None).
            max_length: Максимальная длина текста для синтеза.
            reference_audio_path: Путь к референсному аудио (опционально, в текущей версии не используется).

        Returns:
            str: Путь к сохраненному аудиофайлу или None при ошибке.
        """
        if not text or len(text.strip()) == 0:
            return None

        if target_lang not in ["en", "fr"]:
            raise ValueError(f"Неподдерживаемый целевой язык: {target_lang}. Используйте 'en' или 'fr'")

        if reference_audio_path and not os.path.exists(reference_audio_path):
            print(f"⚠ Предупреждение: файл референсного аудио не найден: {reference_audio_path}")
            print("  Продолжаем без клонирования голоса...")

        lang_names = {"en": "английском", "fr": "французском"}
        print(f"\nСинтез речи на {lang_names[target_lang]} (Bark)...")

        if len(text) > max_length:
            text = text[:max_length]
            print(f"Предупреждение: текст обрезан до {max_length} символов")

        if output_filename is None:
            output_filename = f"synthesized_{target_lang}.wav"

        output_path = os.path.join(OUTPUT_DIR, output_filename)

        try:
            lang_code = self.bark_languages[target_lang]
            prompt = f"[{lang_code}] {text}"

            inputs = self.processor(
                text=[prompt],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                audio_array = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    semantic_temperature=0.7,
                    coarse_temperature=0.7,
                    fine_temperature=0.5,
                )

            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()

            if len(audio_array.shape) > 1:
                audio_array = audio_array.squeeze()

            max_abs = np.max(np.abs(audio_array))
            if max_abs > 0:
                audio_array = audio_array / max_abs * 0.9

                if len(audio_array) > 100:
                    b, a = signal.butter(3, 0.01, 'high')
                    audio_array = signal.filtfilt(b, a, audio_array)
            else:
                print("⚠ Предупреждение: сгенерированное аудио пустое или содержит только нули")
                return None

            sf.write(output_path, audio_array, 24000, subtype='PCM_24')
            print(f"✓ Синтезированное аудио сохранено: {output_path}")
            return output_path

        except Exception as e:
            print(f"Ошибка при синтезе речи: {e}")
            traceback.print_exc()
            return None
