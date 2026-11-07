# Нейросетевой синтез многоплоскостного фазового ДОЭ

Сеть получает 4 картинки 128×128 и выдаёт фазовую маску ДОЭ 128×128. При освещении лазером 532 нм маска
восстанавливает каждую картинку на своём расстоянии: 24,1 / 25,3 / 26,5 / 27,7 мм.
Качество оценивается коэффициентом корреляции Пирсона (КК) и SSIM.

Проект работает напрямую на Windows, macOS и Linux: нужен Python 3.15 и четыре пакета из `requirements.txt`
(PyTorch, NumPy, Pillow, pytest). Все они ставятся готовыми сборками, компилятор не нужен.

## Установка

### Windows

1. Скачать Python 3.15 с https://www.python.org/downloads/ и установить.
   **В первом окне установщика поставить галочку «Add python.exe to PATH».**
2. Открыть PowerShell и выполнить (если нет Git: https://git-scm.com/download/win, или скачать проект ZIP-архивом с GitHub):

```
git clone https://github.com/dark0nn/MephiHack.git
cd MephiHack
python -m pip install -r requirements.txt
python scripts/check_env.py
```

### macOS и Linux

Python 3.15: на Mac `brew install python@3.15`, на Linux — из пакетов дистрибутива или с python.org.

```
git clone https://github.com/dark0nn/MephiHack.git
cd MephiHack
python3.15 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_env.py
```

В новом окне терминала перед работой снова выполнить `source .venv/bin/activate`.

`check_env.py` печатает версию Python, устройство (`cuda`, `mps` или `cpu`) и в конце «Окружение готово.»

## Команды

Одинаковые на всех системах, запускаются из папки `MephiHack`.

| Что | Команда |
| --- | --- |
| Тесты | `python -m pytest -q` |
| Бейзлайны GS и Adam | `python scripts/run_baseline.py --targets data/examples/letters` |
| Скачать MNIST и Fashion-MNIST (один раз, ~110 МБ) | `python scripts/get_datasets.py` |
| Быстрая проверка обучения | `python scripts/train.py --steps 300 --batch 4 --base 16 --val-every 50` |
| Полное обучение | `python scripts/train.py --steps 3000 --batch 16` |
| Продолжить оборванное обучение | `python scripts/train.py --continue checkpoints/<запуск>/last.pt` |
| Результат по ТЗ (последняя обученная сеть) | `python scripts/infer.py --targets data/examples/shapes` |

На Mac долгое обучение лучше запускать через `caffeinate -i python scripts/train.py ...`, чтобы ноутбук не уснул.

Результаты:
- `outputs/` — `doe.png` (0 = −π, 255 = +π), `recon_1..4.png`, `metrics.json`;
- `checkpoints/<запуск>/` — веса `best.pt` и `last.pt`, лог `log.csv`.

## Видеокарта NVIDIA (Windows, Linux)

`pip install -r requirements.txt` ставит PyTorch без поддержки CUDA. Чтобы обучать на видеокарте NVIDIA,
после установки один раз переставить PyTorch:

```
python -m pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

После этого `python scripts/check_env.py` должен напечатать `устройство: cuda`.
На Mac с M1–M4 ничего делать не нужно: видеоядро (`mps`) подхватывается само.

## Если не работает

| Что видно | Что сделать |
| --- | --- |
| Windows: `python` не найден или открывается Microsoft Store | Python установлен без галочки «Add to PATH»: переустановить с ней, затем открыть новое окно PowerShell |
| `python --version` показывает не 3.15 | Стоит несколько версий. Windows: вместо `python` писать `py -3.15`. Mac/Linux: создать `.venv` через `python3.15`, как выше |
| `No matching distribution found for torch` | Python не 3.15 или 32-битный. Нужен 64-битный Python 3.15 |
| `ModuleNotFoundError: No module named 'torch'` | Mac/Linux: не активировано окружение, выполнить `source .venv/bin/activate` |
| `error: externally-managed-environment` | Mac/Linux: pip запущен вне `.venv`; создать и активировать `.venv`, как выше |
| Обучение на Windows зависает или падает при старте | Добавить к команде обучения `--workers 0` |

Ничего не подошло? Скопируйте весь текст из терминала, от команды до последней строки, и отправьте в чат команды.

## Docker (необязательно)

Для тех, у кого уже стоит Docker, есть `Dockerfile` с тем же окружением:

```
docker build -t doe .
docker run --rm -v "${PWD}:/app" doe python scripts/run_baseline.py --targets data/examples/letters
```

## Как устроен проект

```
requirements.txt  зависимости Python
doe/              общий код: физика, метрики, лосс, данные, модель
  config.py       параметры из ТЗ (λ, пиксель, расстояния) и правило перевода в 8 бит
  optics.py       дифференцируемая модель Френеля (PyTorch); optics_np.py — эталон на numpy
  metrics.py      КК и SSIM (глобальный и оконный 7×7)
  baselines.py    GS и Adam по фазе
  model.py        U-Net + физика внутри модели
scripts/          запускаемые скрипты (см. таблицу)
tests/            проверки физики и формата
data/examples/    тестовые наборы по 4 картинки
data/images/      картинки для обучения: MNIST/Fashion-MNIST и свои PNG (не в git)
checkpoints/      веса сетей (не в git)
outputs/          результаты (не в git)
```

Сеть учится без учителя: цели → U-Net → фаза ДОЭ → 4 восстановления через модель Френеля → лосс
(1 − КК) + (1 − SSIM) + MSE → обратное распространение. Ответы GS как разметка не используются.

Чекпоинты: `--continue <last.pt>` продолжает тот же запуск с того же шага;
`--resume <best.pt>` берёт веса как старт для новой версии модели (грузит слои с совпадающим именем и формой).
`infer.py` сам определяет тип модели по чекпоинту.

## Открытые вопросы к организаторам

Сравнивают ли картинку с амплитудой |K| или интенсивностью |K|², как переводят в 8 бит, какой SSIM —
глобальный или оконный. Сейчас в `doe/config.py` стоят `recon_quantity="amplitude"` и `recon_norm="max"`;
после ответа достаточно поменять их там. В `metrics.json` пишутся оба варианта SSIM.
