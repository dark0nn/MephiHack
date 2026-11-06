#!/usr/bin/env bash
# Создаёт окружение проекта в .venv и ставит зависимости.
# Mac / Linux:            bash setup_env.sh
# Windows (Git Bash):     bash setup_env.sh
# Свой Python:            PYTHON="py -3.15" bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1   # чтобы русский текст печатался в консоли Windows

# Целевая версия — Python 3.15 (PyTorch >= 2.15 даёт под неё сборки).
# Допускаем 3.11–3.15. Проверяем, что кандидат реально запускается:
# на Windows "python3" часто оказывается заглушкой Microsoft Store.
ok() { $1 -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,15) else 1)' >/dev/null 2>&1; }

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.15 "py -3.15" python3.14 python3.13 python3.12 python3.11 "py -3.14" "py -3.13" "py -3.12" "py -3.11" python3 python; do
    if ok "$c"; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ] || ! ok "$PY"; then
  echo "Не найден рабочий Python 3.11–3.15 (рекомендуется 3.15)."
  echo "  Mac:     brew install python@3.15"
  echo "  Windows: winget install Python.Python.3.15   (потом закрыть и заново открыть терминал)"
  echo "  Linux:   uv python install 3.15  и  PYTHON=\"\$(uv python find 3.15)\" bash setup_env.sh"
  exit 1
fi
echo ">> Использую $($PY --version) ($PY)"
$PY -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,15) else 1)' || echo "   (рекомендуется Python 3.15; работаем на этой версии)"

# Если .venv создан другой версией Python — пересоздаём
WANT="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
for v in .venv/bin/python .venv/Scripts/python.exe; do
  if [ -x "$v" ]; then
    HAVE="$("$v" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)"
    if [ "$HAVE" != "$WANT" ]; then echo ">> .venv на Python $HAVE, пересоздаю под $WANT"; rm -rf .venv; fi
    break
  fi
done
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
