# Необязательный способ запуска (основной — без Docker, см. README).
# Образ проекта: всё окружение (Python 3.15, PyTorch, зависимости из requirements.txt).
#
# Сборка:      docker build -t doe .
# GPU NVIDIA:  docker build -t doe-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
# Запуск (PowerShell, macOS, Linux; из папки проекта):
#   docker run --rm -v "${PWD}:/app" doe python scripts/run_baseline.py --targets data/examples/letters
# Флаг -v подключает папку проекта: код после git pull подхватывается без пересборки,
# а результаты (outputs/, checkpoints/) остаются на компьютере.
FROM python:3.15-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Шрифт с кириллицей для синтетических картинок (doe/synth.py)
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем: пересобираются только при изменении requirements.txt
COPY requirements.txt .
RUN pip install torch --index-url "$TORCH_INDEX" \
    && pip install -r requirements.txt

# Код копируется в образ, чтобы он работал и без монтирования папки.
# При запуске с -v "${PWD}:/app" папка проекта подключается поверх, и пересобирать после правок кода не нужно.
COPY . .

CMD ["python", "scripts/check_env.py"]
