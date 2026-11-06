#!/usr/bin/env bash
# Создаёт окружение проекта в .venv и ставит зависимости.
# Запуск из Терминала:  bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"

# Нужен Python 3.10–3.12 (PyTorch для 3.13+ может ещё не выйти под вашу систему).
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3 не найден. На Mac: brew install python@3.12  (или скачайте с python.org)"; exit 1
fi
echo ">> Использую $($PY --version) ($PY)"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Ядро для Jupyter, чтобы ноутбуки видели это окружение
python -m ipykernel install --user --name doe-hackathon --display-name "DOE hackathon" >/dev/null || true

python scripts/make_examples.py
python scripts/check_env.py

echo
echo "Готово. В новом окне Терминала перед работой:  cd \"$(pwd)\" && source .venv/bin/activate"
