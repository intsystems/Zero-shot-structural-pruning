"""Systematic mask sampling and loss evaluation for the surrogate."""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm 

@torch.no_grad()
def build_systematic_mask_dataset(
    model: torch.nn.Module,
    n_groups: int,
    mask_state: torch.Tensor,
    data_loader: Iterable,
    criterion: Callable,
    device: str | torch.device,
) -> TensorDataset:
    """Build a (masks, losses) dataset by systematically toggling one group off at a time.

    Masks:
        - all-ones
        - all-zeros
        - one all-ones-with-position-k-zero for each k in [0, n_groups)

    For every mask the loss is averaged over the whole data_loader; the resulting
    vector is min-max normalized to [0, 1] and inverted (1 - x) so that a higher
    score means "this configuration hurts more when removed".

    Assumes that the caller has installed forward hooks on the model that read
    values from `mask_state` (a tensor of shape [n_groups] on `device`) — this
    function just writes into `mask_state` and runs forward passes.
    """
    model.eval()

    masks: List[torch.Tensor] = [torch.ones(n_groups), torch.zeros(n_groups)]
    for k in range(n_groups):
        m = torch.ones(n_groups)
        m[k] = 0.0
        masks.append(m)

    losses: List[float] = []
    for mask in tqdm(masks):
        mask_state.copy_(mask.to(mask_state.device))
        total_loss = 0.0
        total = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total += y.size(0)
        losses.append(total_loss / max(total, 1))

    losses_t = torch.tensor(losses)
    losses_t = (losses_t - losses_t.min()) / (1e-10 + losses_t.max() - losses_t.min())
    losses_t = 1.0 - losses_t

    return TensorDataset(torch.vstack(masks), losses_t)


import torch

def generate_hadamard_matrix(n: int) -> torch.Tensor:
    """Генерирует матрицу Адамара размера n x n (n должно быть степенью двойки)."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Размер матрицы Адамара n должен быть степенью 2 (1, 2, 4, 8, 16...).")
    
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H

import math
import torch
from tqdm import tqdm


def build_hadamard_log_dataset(
    model, 
    total_channels: int, 
    mask_state: torch.Tensor, 
    data_loader, 
    criterion, 
    device, 
    n_masks: int = 256  # Фиксированное число масок K = O(log C)
):
    """
    Генерирует строго K (например, 256) масок Адамара для ЛЮБОГО числа каналов (даже 8200+).
    Время работы: O(n_masks), а не O(total_channels).
    """
    # 1. Берем матрицу Адамара размера K x K (K должно быть степенью 2)
    K = 1 << (n_masks - 1).bit_length()
    K = max(16, K)
    H_k = generate_hadamard_matrix(K)  # Форма: [K, K]

    # 2. Переводим [-1, 1] -> [0, 1]
    masks_base = (H_k + 1.0) / 2.0

    # 3. Растягиваем K столбцов матрицы на total_channels (8200) с помощью цикловирования
    # или случайной проекции, чтобы у каждого канала был уникальный паттерн
    repeats = math.ceil(total_channels / K)
    
    # Перемешиваем индексы для каждого повтора, чтобы устранить дублирование групп каналов
    projected_cols = []
    g = torch.Generator().manual_seed(42) # Фиксируем seed для воспроизводимости
    for _ in range(repeats):
        perm = torch.randperm(K, generator=g)
        projected_cols.append(masks_base[:, perm])
        
    masks_bin = torch.cat(projected_cols, dim=1)[:, :total_channels]

    # 4. Добавляем базовую маску All-Ones (рабочая сеть)
    all_ones = torch.ones(1, total_channels)
    all_masks = torch.cat([all_ones, masks_bin], dim=0)

    # 5. Дедупликация и отправка на GPU
    all_masks = torch.unique(all_masks, dim=0).to(device)

    dataset = []

    # 6. Прогон всего ~128-256 масок вместо 8200!
    for mask in tqdm(all_masks, desc=f"Collecting Log-Hadamard Masks ({len(all_masks)})"):
        if mask.sum() == 0:
            continue

        mask_state.copy_(mask)

        total_loss = 0.0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        dataset.append((mask.cpu(), torch.tensor(avg_loss, dtype=torch.float32)))

    return dataset

import typing
from typing import Iterable, Callable, List
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def generate_hadamard_matrix(n: int) -> torch.Tensor:
    """Генерирует матрицу Адамара размера n x n (n — степень двойки)."""
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H


@torch.no_grad()
def build_hybrid_hadamard_dataset(
    model: torch.nn.Module,
    total_channels: int,
    group_channel_counts: List[int],
    mask_state: torch.Tensor,
    data_loader: Iterable,
    criterion: Callable,
    device: str | torch.device,
) -> TensorDataset:
    """
    Гибридное сэмплирование:
    - 1 маска All-Ones
    - 1 маска All-Zeros
    - Внутри каждой группы каналов генерируются маски Адамара размера O(log C_group),
      пока каналы остальных групп остаются включенными (1.0).
    """
    model.eval()

    masks: List[torch.Tensor] = [
        torch.ones(total_channels),   # All-Ones
        torch.zeros(total_channels)   # All-Zeros
    ]

    # Вычисляем offsets (границы каналов каждой группы)
    channel_offsets = []
    curr = 0
    for count in group_channel_counts:
        channel_offsets.append((curr, curr + count))
        curr += count

    # Генерация блок-Адамара для каждой группы отдельно
    for g_idx, c_count in enumerate(group_channel_counts):
        start, end = channel_offsets[g_idx]

        # Минимальный Адамар O(log2(C_group)), например для 64 каналов -> K=8 или 16 масок
        n_hadamard = 1 << (c_count - 1).bit_length()
        n_hadamard = max(4, n_hadamard)
        
        # Ограничиваем сверху число масок на группу (например, не больше 32),
        # чтобы даже на огромных слоях с 2048 каналами не делать много прогонов
        K = min(32, n_hadamard)

        H = generate_hadamard_matrix(K)
        # Скрэмблинг столбцов для устранения дубликатов при обрезке
        scramble_signs = torch.randint(0, 2, (K,), dtype=torch.float32) * 2.0 - 1.0
        H_scrambled = H * scramble_signs

        group_masks_bin = (H_scrambled + 1.0) / 2.0  # [K, K]

        # Растягиваем/обрезаем K столбцов под c_count каналов группы
        repeats = (c_count + K - 1) // K
        group_masks_bin = group_masks_bin.repeat(1, repeats)[:, :c_count]

        # Создаем глобальные маски: везде 1.0, а в диапазоне [start:end] — Адамар
        for k in range(group_masks_bin.shape[0]):
            m = torch.ones(total_channels)
            m[start:end] = group_masks_bin[k]
            masks.append(m)

    # Дедупликация на случай совпадений
    masks_tensor = torch.stack(masks)
    masks_tensor = torch.unique(masks_tensor, dim=0)

    # Оценка Loss на датасете
    losses: List[float] = []
    for mask in tqdm(masks_tensor, desc="Collecting Hybrid Hadamard Dataset"):
        mask_state.copy_(mask.to(mask_state.device))
        total_loss = 0.0
        total = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total += y.size(0)
        losses.append(total_loss / max(total, 1))

    losses_t = torch.tensor(losses)
    
    # Мин-макс нормализация [0, 1] и инверсия (сохраняем оригинальную логику)
    losses_t = (losses_t - losses_t.min()) / (1e-10 + losses_t.max() - losses_t.min())
    losses_t = 1.0 - losses_t

    return TensorDataset(masks_tensor, losses_t)