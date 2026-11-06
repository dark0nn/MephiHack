#!/usr/bin/env bash
# Создаёт окружение проекта в .venv и ставит зависимости.
# Mac / Linux:            bash setup_env.sh
# Windows (Git Bash):     bash setup_env.sh
# Свой Python:            PYTHON="py -3.12" bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1   # чтобы русский текст печатался в консоли Windows

# Нужен Python 3.10–3.12. Проверяем, что кандидат реально запускается:
# на Windows "python3" часто оказывается заглушкой Microsoft Store.
ok() { $1 -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' >/dev/null 2>&1; }

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 "py -3.12" "py -3.11" "py -3.10" python3 python; do
    if ok "$c"; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ] || ! ok "$PY"; then
  echo "Не найден рабочий Python 3.10–3.12."
  echo "  Mac:     brew install python@3.12"
  echo "  Windows: winget install Python.Python.3.12   (потом закрыть и заново открыть терминал)"
  echo "  Linux:   uv python install 3.12  и  PYTHON=\"\$(uv python find 3.12)\" bash setup_env.sh"
  exit 1
fi
echo ">> Использую $($PY --version) ($PY)"

if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
# На Windows активатор лежит в Scripts, на Mac/Linux — в bin
if [ -f .venv/Scripts/activate ]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Ядро для Jupyter, чтобы ноутбуки видели это окружение
python -m ipykernel install --user --name doe-hackathon --display-name "DOE hackathon" >/dev/null || true

python scripts/make_examples.py
python scripts/check_env.py

echo
if [ -f .venv/Scripts/activate ]; then
  echo "Готово. В новом окне терминала перед работой:  cd \"$(pwd)\" && source .venv/Scripts/activate"
else
  echo "Готово. В новом окне терминала перед работой:  cd \"$(pwd)\" && source .venv/bin/activate"
fi
