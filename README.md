# Кейс: нейросетевой синтез многоплоскостного фазового ДОЭ

Сеть получает 4 картинки 128×128 и выдаёт фазовую маску ДОЭ 128×128, которая при освещении
лазером 532 нм восстанавливает каждую картинку на своём расстоянии: 24,1 / 25,3 / 26,5 / 27,7 мм.
Качество оценивается коэффициентом корреляции Пирсона и SSIM.

Ключевая идея: сеть учится **без ответов GS**. Модель Френеля дифференцируема, поэтому
цикл такой: цели → U-Net → фаза → 4 восстановления → лосс → обратное распространение ошибки.

## Установка на Mac (один раз)

Откройте Терминал и выполните:

```bash
cd ~/Documents/doe-hackathon
bash setup_env.sh
```

Скрипт создаст окружение `.venv`, поставит PyTorch и остальное, сгенерирует примеры
и проверит, что физика считается правильно. На Mac с чипом M1/M2/M3 PyTorch использует
видеоядро (устройство `mps`).

Перед каждой работой в новом окне Терминала:

```bash
cd ~/Documents/doe-hackathon
source .venv/bin/activate
```

Если `python3` не найден: `brew install python@3.12` (или установщик с python.org).

## Установка на Windows

1. Поставить Python 3.12 (в PowerShell): `winget install Python.Python.3.12`,
   потом **закрыть и заново открыть** терминал / VS Code. Заглушка `python3` из Microsoft Store
   не подходит (ошибка `Permission denied`): её можно выключить в «Параметры → Приложения →
   Псевдонимы выполнения приложений».
2. В терминале **Git Bash** (в VS Code: Terminal → bash):

```bash
cd ~/MephiHack
bash setup_env.sh
```

Перед каждой работой: `source .venv/Scripts/activate` (на Windows — `Scripts`, не `bin`).
Если у ноутбука видеокарта NVIDIA, после установки поставить PyTorch с CUDA:
`pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall`.

## Основные команды

| Что | Команда |
| --- | --- |
| Проверить окружение | `python scripts/check_env.py` |
| Тесты физики и формата | `python -m pytest -q` |
| Сгенерировать примеры наборов | `python scripts/make_examples.py` |
| Бейзлайны GS и Adam на наборе | `python scripts/run_baseline.py --targets data/examples/shapes` |
| Быстрая проверка обучения | `python scripts/train.py --steps 200 --batch 4 --base 16 --workers 0` |
| Полное обучение (лучше на GPU/Colab) | `python scripts/train.py --steps 3000 --batch 16` |
| Финальный результат по ТЗ | `python scripts/infer.py --weights checkpoints/<run>/best.pt --targets <папка с 4 PNG>` |
| То же + дотюнивание фазы | добавить `--refine 50` |

Результаты лежат в `outputs/`: `doe.png` (0 = −π, 255 = +π), `recon_1..4.png`, `metrics.json`.

## Docker

Образ со всеми зависимостями (Python 3.12, PyTorch, Jupyter). При сборке сразу генерируются примеры и прогоняются тесты.

| Что | Команда |
| --- | --- |
| Собрать образ (CPU) | `docker compose build doe` |
| Бейзлайны на буквах | `docker compose run --rm doe` |
| Тесты | `docker compose run --rm doe python -m pytest -q` |
| JupyterLab на http://localhost:8888 | `docker compose up jupyter` |
| Обучение на NVIDIA GPU | `docker compose run --rm train-gpu` |

Результаты и веса сохраняются в `outputs/` и `checkpoints/` на хосте. На Mac Docker не видит видеоядро (mps), поэтому
для обучения на Mac удобнее `.venv`, а Docker нужен для GPU-сервера и для одинакового окружения у всей команды.

## Структура

```
doe/                 # общий код, его импортируют все скрипты
  config.py          # параметры из ТЗ (λ, пиксель, расстояния) и правило перевода в 8 бит
  optics.py          # модель Френеля на PyTorch (дифференцируемая)
  optics_np.py       # эталонная модель на numpy — для проверки
  metrics.py         # КК и SSIM: для лосса (torch) и «как у жюри» (numpy)
  losses.py          # лосс = (1−КК) + (1−SSIM) + MSE
  baselines.py       # GS параллельный, GS последовательный, Adam по фазе
  data.py            # генератор обучающих четвёрок картинок
  model.py           # U-Net: 4 канала -> фаза
  io.py              # чтение/запись PNG в формате ТЗ
  pipeline.py        # единая финальная оценка любого метода
scripts/             # запускаемые скрипты (см. таблицу)
tests/               # проверки физики и формата
data/examples/       # тестовые наборы по 4 картинки
data/images/         # сюда можно положить свои картинки для обучения (MNIST, буквы, фото)
notebooks/           # ноутбуки, в т.ч. для Colab
checkpoints/         # веса сетей (не в git)
outputs/             # результаты (не в git)
```

## Кто что делает

| Роль | Файлы |
| --- | --- |
| Физика | `doe/optics.py`, `doe/optics_np.py`, `tests/` — сверить с формулой со слайда, TorchOptics/Odak |
| Бейзлайны | `doe/baselines.py`, `scripts/run_baseline.py` — таблица «метод / время / КК / SSIM» |
| Нейросеть | `doe/model.py`, `doe/data.py`, `doe/losses.py`, `scripts/train.py` |
| Сдача | `scripts/infer.py`, `outputs/` |

## Открытые вопросы к организаторам

Что сравнивают с картинкой — амплитуду |K| или интенсивность |K|², как переводят в 8 бит,
глобальный SSIM или оконный. Пока стоят `recon_quantity="amplitude"` и `recon_norm="max"`
в `doe/config.py`; после ответа поменять там, всё остальное подстроится само.
`metrics.json` содержит оба варианта SSIM.


1. Первый раз (один раз)

bash
cd ~/Documents/doe-hackathon
bash setup_env.sh

Создаст окружение, поставит PyTorch, сгенерирует тестовые картинки.

2. Каждый раз в новом окне терминала

bash
cd ~/Documents/doe-hackathon
source .venv/bin/activate

3. Основные команды

bash
# Проверить что всё работает
python scripts/check_env.py

# Запустить тесты
python -m pytest -q

# Бейзлайн GS на тестовом наборе (быстро, без GPU)
python scripts/run_baseline.py --targets data/examples/shapes

# Быстрая проверка обучения сети (2–3 мин, работает на mps)
python scripts/train.py --steps 200 --batch 4 --base 16 --workers 0

# Полное обучение
python scripts/train.py --steps 3000 --batch 16

# Финальный результат (doe.png + метрики)
python scripts/infer.py --weights checkpoints/<run>/best.pt --targets data/examples/shapes

4. Результаты лежат в outputs/:

doe.png — фазовая маска в формате организаторов
recon_1.png … recon_4.png — восстановленные изображения
metrics.json — КК и SSIM