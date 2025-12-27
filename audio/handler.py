"""
Модуль для работы с аудио: запись, воспроизведение, сохранение.
"""

import os
import sounddevice as sd
import soundfile as sf
from config import SAMPLE_RATE, OUTPUT_DIR


class AudioHandler:
    """Класс для работы с аудио: запись с микрофона и воспроизведение."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """Инициализирует обработчик аудио.

        Args:
            sample_rate: Частота дискретизации для записи и воспроизведения.
        """
        self.sample_rate = sample_rate

    def record_audio(self, duration: float, output_filename: str = "recorded_audio.wav") -> str:
        """Записывает аудио с микрофона.

        Args:
            duration: Длительность записи в секундах.
            output_filename: Имя файла для сохранения.

        Returns:
            str: Путь к сохраненному аудиофайлу.

        Raises:
            RuntimeError: Если запись аудио не удалась.
        """
        print(f"\nЗапись аудио ({duration} секунд)...")
        print("Говорите на русском языке...")

        try:
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()

            output_path = os.path.join(OUTPUT_DIR, output_filename)
            sf.write(output_path, audio, self.sample_rate, subtype='PCM_24')
            print(f"Аудио сохранено: {output_path}")

            return output_path
        except Exception as e:
            raise RuntimeError(f"Ошибка при записи аудио: {e}")

    def play_audio(self, audio_path: str):
        """Воспроизводит аудиофайл через колонки.

        Args:
            audio_path: Путь к аудиофайлу.

        Raises:
            FileNotFoundError: Если файл не найден.
            RuntimeError: Если воспроизведение аудио не удалось.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Файл не найден: {audio_path}")

        try:
            print("\nВоспроизведение аудио...")
            data, sample_rate = sf.read(audio_path)
            sd.play(data, sample_rate)
            sd.wait()
            print("Воспроизведение завершено")
        except Exception as e:
            raise RuntimeError(f"Ошибка при воспроизведении аудио: {e}")
