import time
from audio import AudioHandler
from recognition import SpeechRecognizer
from translation import TextTranslator
from synthesis import SpeechSynthesizer
from config import RECORD_DURATION, DEVICE
from utils import print_memory_usage, clear_cache


class SpeechTranslator:
    """Класс для перевода речи с русского на английский или французский."""

    def __init__(self, target_lang: str = "fr"):
        """Инициализирует все компоненты системы перевода речи.

        Args:
            target_lang: Целевой язык ('en' для английского, 'fr' для французского).
        """
        if target_lang not in ["en", "fr"]:
            raise ValueError(f"Неподдерживаемый целевой язык: {target_lang}. Используйте 'en' или 'fr'")

        self.target_lang = target_lang
        lang_names = {"en": "английский", "fr": "французский"}
        print(f"Инициализация системы перевода: Русский -> {lang_names[target_lang].upper()}")

        self.audio_handler = AudioHandler()

        # Загрузка моделей с мониторингом памяти
        if DEVICE == "cuda":
            print_memory_usage()

        self.recognizer = SpeechRecognizer()

        if DEVICE == "cuda":
            print_memory_usage()
            # Очистка кэша после загрузки Whisper для освобождения памяти
            clear_cache()

        self.translator = TextTranslator()

        if DEVICE == "cuda":
            print_memory_usage()

        self.synthesizer = SpeechSynthesizer()

        if DEVICE == "cuda":
            print_memory_usage()

        print("Система готова к использованию!")

    def process(self):
        """Выполняет полный цикл: запись -> распознавание -> перевод -> синтез -> воспроизведение."""
        start_time = time.time()

        try:
            # Шаг 1: Запись с микрофона
            step_start = time.time()
            recorded_path = self.audio_handler.record_audio(RECORD_DURATION)
            record_time = time.time() - step_start
            print(f"⏱ Запись завершена за {record_time:.2f} сек")

            # Шаг 2: Распознавание речи
            step_start = time.time()
            recognized_text = self.recognizer.recognize(recorded_path)
            recognition_time = time.time() - step_start
            print(f"⏱ Распознавание завершено за {recognition_time:.2f} сек")

            if not recognized_text or len(recognized_text.strip()) == 0:
                print("Не удалось распознать речь. Попробуйте еще раз.")
                return

            # Шаг 3: Перевод
            step_start = time.time()
            translated_text = self.translator.translate(recognized_text, target_lang=self.target_lang)
            translation_time = time.time() - step_start
            print(f"⏱ Перевод завершен за {translation_time:.2f} сек")

            if not translated_text or len(translated_text.strip()) == 0:
                print("Не удалось перевести текст.")
                return

            # Шаг 4: Синтез речи с клонированием голоса (используем записанный аудио как референс)
            step_start = time.time()
            result_path = self.synthesizer.synthesize(
                translated_text,
                target_lang=self.target_lang,
                output_filename=f"synthesized_{self.target_lang}.wav",
                reference_audio_path=recorded_path  # Используем записанный аудио для клонирования голоса
            )
            synthesis_time = time.time() - step_start
            print(f"⏱ Синтез завершен за {synthesis_time:.2f} сек")

            if result_path:
                # Шаг 5: Воспроизведение
                step_start = time.time()
                self.audio_handler.play_audio(result_path)
                playback_time = time.time() - step_start
                print(f"⏱ Воспроизведение завершено за {playback_time:.2f} сек")

                # Итоговая статистика
                total_time = time.time() - start_time
                processing_time = total_time - record_time - playback_time

                print("\n" + "=" * 60)
                print("СТАТИСТИКА ОБРАБОТКИ")
                print("=" * 60)
                print(f"Запись аудио:        {record_time:.2f} сек")
                print(f"Распознавание речи:  {recognition_time:.2f} сек")
                print(f"Перевод текста:      {translation_time:.2f} сек")
                print(f"Синтез речи:         {synthesis_time:.2f} сек")
                print(f"Воспроизведение:     {playback_time:.2f} сек")
                print("-" * 60)
                print(f"Время обработки:      {processing_time:.2f} сек")
                print(f"Общее время:         {total_time:.2f} сек")
                print("=" * 60)
            else:
                print("Не удалось синтезировать речь.")

        except KeyboardInterrupt:
            print("\n\nПрограмма остановлена пользователем.")
        except Exception as e:
            print(f"\nОшибка: {e}")
