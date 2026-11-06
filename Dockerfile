# DOE hackathon: все зависимости проекта в одном образе.
#
# CPU (по умолчанию):
#   docker build -t doe-hackathon:cpu .
# GPU (NVIDIA, нужен nvidia-container-toolkit на хосте):
#   docker build -t doe-hackathon:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
#
# Запуск:
#   docker run --rm -v "$PWD/outputs:/app/outputs" doe-hackathon:cpu \
#       python scripts/run_baseline.py --targets data/examples/letters
#   docker run --rm --gpus all -v "$PWD/checkpoints:/app/checkpoints" doe-hackathon:gpu \
#       python scripts/train.py --steps 3000 --batch 16
FROM python:3.15-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Системные библиотеки для Pillow / scikit-image / matplotlib + шрифт с кириллицей
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 fonts-dejavu-core git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем: пересобираются только при изменении requirements.txt
COPY requirements.txt .
RUN pip install torch --index-url "$TORCH_INDEX" \
    && pip install -r requirements.txt

# Код проекта
COPY . .

# Примеры картинок и проверка физики прямо при сборке
RUN python scripts/make_examples.py && python -m pytest -q tests/

EXPOSE 8888
CMD ["python", "scripts/check_env.py"]
