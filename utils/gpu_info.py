"""
Утилиты для работы с GPU и мониторинга памяти.
"""

import torch
from config import DEVICE


def _get_memory_usage():
    """Получает текущее использование памяти GPU.

    Returns:
        dict: Информация об использовании памяти или None если GPU недоступна.
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved,
        'usage_percent': (reserved / total) * 100
    }


def print_memory_usage():
    """Выводит информацию об использовании памяти GPU."""
    if DEVICE != "cuda":
        print("GPU не используется")
        return

    memory = _get_memory_usage()
    if memory:
        print(f"\nИспользование VRAM:")
        print(f"  Выделено: {memory['allocated_gb']:.2f} GB")
        print(f"  Зарезервировано: {memory['reserved_gb']:.2f} GB")
        print(f"  Свободно: {memory['free_gb']:.2f} GB")
        print(f"  Использовано: {memory['usage_percent']:.1f}%")


def clear_cache():
    """Очищает кэш GPU для освобождения памяти."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Кэш GPU очищен")
