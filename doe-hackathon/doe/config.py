"""Параметры задачи из ТЗ (слайд «Условия – 1: исходные параметры»)."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    n: int = 128                       # размер ДОЭ и изображений, пикселей
    wavelength: float = 532e-9         # длина волны лазера, м
    pixel: float = 10e-6               # шаг пикселя ПВМС, м
    distances: tuple = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)  # расстояния до 4 плоскостей, м

    # Как восстановление превращается в 8-битную картинку.
    # ЭТО НУЖНО УТОЧНИТЬ У ОРГАНИЗАТОРОВ: от выбора сильно зависит SSIM.
    recon_quantity: str = "amplitude"  # "amplitude" (|K|) или "intensity" (|K|^2)
    recon_norm: str = "max"            # "max" (деление на максимум) или "p99" (на 99-й процентиль с обрезкой)

    @property
    def planes(self) -> int:
        return len(self.distances)

    def out_pixel(self, z: float) -> float:
        """Шаг сетки в плоскости восстановления для формулы с одним БПФ: λz / (NΔ)."""
        return self.wavelength * z / (self.n * self.pixel)

    @property
    def critical_distance(self) -> float:
        """NΔ²/λ: на этом расстоянии шаг на выходе равен шагу на входе (≈24,06 мм)."""
        return self.n * self.pixel ** 2 / self.wavelength


DEFAULT = Config()
