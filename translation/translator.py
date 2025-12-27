"""
Модуль для перевода текста с использованием NLLB (No Language Left Behind).
"""

import os
import traceback
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import DEVICE, HF_MODELS_DIR


class TextTranslator:
    """Класс для перевода текста с русского на английский или французский через NLLB."""

    def __init__(self):
        """Инициализирует модель перевода NLLB."""
        print("Загрузка модели NLLB для перевода...")
        print(f"Модели Hugging Face будут сохранены в: {HF_MODELS_DIR}")

        cache_dir = os.path.join(HF_MODELS_DIR, "transformers")
        os.makedirs(cache_dir, exist_ok=True)

        model_name = "facebook/nllb-200-distilled-600M"

        try:
            print(f"Загрузка модели {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir
            ).to(DEVICE)

            print("✓ Модель NLLB загружена!")
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели NLLB: {e}")

        self.nllb_languages = {
            "en": "eng_Latn",
            "fr": "fra_Latn"
        }

        print("Модель перевода готова!")

    def translate(self, text: str, target_lang: str = "fr", max_input_length: int = 500) -> str:
        """Переводит текст с русского на указанный язык через NLLB.

        Args:
            text: Текст на русском языке.
            target_lang: Целевой язык ('en' для английского, 'fr' для французского).
            max_input_length: Максимальная длина входного текста.

        Returns:
            str: Переведенный текст на целевом языке.
        """
        if not text or len(text.strip()) == 0:
            return ""

        if target_lang not in ["en", "fr"]:
            raise ValueError(f"Неподдерживаемый целевой язык: {target_lang}. Используйте 'en' или 'fr'")

        print(f"\nПеревод текста с русского на {target_lang.upper()} (NLLB)...")

        if len(text) > max_input_length:
            text = text[:max_input_length]
            print(f"Предупреждение: текст обрезан до {max_input_length} символов")

        try:
            target_lang_code = self.nllb_languages[target_lang]

            try:
                forced_bos_token_id = self.tokenizer.lang_code_to_id[target_lang_code]
            except (AttributeError, KeyError):
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang_code)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=400
            ).to(DEVICE)

            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=400,
                    num_beams=4,
                    early_stopping=True
                )

            translated_text = self.tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )

            max_output_length = 300
            if len(translated_text) > max_output_length:
                translated_text = translated_text[:max_output_length]
                print(f"Предупреждение: переведенный текст обрезан до {max_output_length} символов для TTS")

            print(f"Переведенный текст: {translated_text}")
            return translated_text

        except Exception as e:
            print(f"Ошибка при переводе: {e}")
            traceback.print_exc()
            return ""
