# Кейс: нейросетевой синтез многоплоскостного фазового ДОЭ

Сеть получает 4 картинки 128×128 и выдаёт фазовую маску ДОЭ 128×128, которая при освещении
лазером 532 нм восстанавливает каждую картинку на своём расстоянии: 24,1 / 25,3 / 26,5 / 27,7 мм.
Качество оценивается коэффициентом корреляции Пирсона и SSIM.

Ключевая идея: сеть учится **без ответов GS**. Модель Френеля дифференцируема, поэтому
цикл такой: цели → U-Net → фаза → 4 восстановления → лосс → обратное распространение ошибки.

## Установка Docker (один раз)

Всё окружение (Python, PyTorch, Jupyter, библиотеки) собирается внутри Docker-образа из двух файлов:
`Dockerfile` и `requirements.txt`. Ставить Python и пакеты на компьютер не нужно.

**Mac (Intel и M1/M2/M3)**

1. Скачать Docker Desktop: https://www.docker.com/products/docker-desktop/ (или `brew install --cask docker`).
2. Открыть приложение Docker из «Программ» и дождаться, пока значок кита в строке меню перестанет мигать.
3. Проверить в Терминале: `docker --version` и `docker run --rm hello-world`.

**Windows 10/11**

1. В PowerShell **от имени администратора**: `wsl --install`, затем перезагрузить компьютер.
2. Скачать и поставить Docker Desktop: https://www.docker.com/products/docker-desktop/
   (в установщике оставить галочку «Use WSL 2»).
3. Запустить Docker Desktop, дождаться статуса «Engine running».
4. Проверить в PowerShell: `docker --version` и `docker run --rm hello-world`.

**Linux (Ubuntu/Debian)**

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER     # чтобы не писать sudo перед каждой командой
# выйти из системы и зайти заново (или перезагрузиться)
docker run --rm hello-world
```

Для обучения на видеокарте NVIDIA дополнительно нужен NVIDIA Container Toolkit:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
(после установки: `sudo systemctl restart docker`, проверка: `docker run --rm --gpus all ubuntu nvidia-smi`).

## Запуск через Docker

Все команды выполняются из папки проекта (`cd ~/Documents/MephiHack`).

**1. Собрать образ** (первый раз 5–15 минут, дальше секунды, если не менялся `requirements.txt`).
При сборке сразу генерируются примеры картинок и прогоняются тесты физики.

```bash
# CPU (Mac, любой ноутбук)
docker build -t doe .

# GPU (Linux/Windows с видеокартой NVIDIA)
docker build -t doe-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
```

**2. Запускать:**

| Что | Команда |
| --- | --- |
| Проверить окружение | `docker run --rm doe` |
| Тесты | `docker run --rm doe python -m pytest -q` |
| Бейзлайны GS/Adam на буквах | `docker run --rm -v "$PWD/outputs:/app/outputs" doe python scripts/run_baseline.py --targets data/examples/letters` |
| Обучение на CPU (быстрая проверка) | `docker run --rm -v "$PWD/checkpoints:/app/checkpoints" doe python scripts/train.py --steps 200 --batch 4 --base 16 --workers 0` |
| Обучение на NVIDIA GPU | `docker run --rm --gpus all --shm-size=2g -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/data/images:/app/data/images" doe-gpu python scripts/train.py --steps 3000 --batch 16` |
| Финальный результат (берёт последнюю обученную сеть) | `docker run --rm -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/outputs:/app/outputs" doe python scripts/infer.py --weights "$(ls -td checkpoints/*/ \| head -1)best.pt" --targets data/examples/shapes` |
| JupyterLab на http://localhost:8888 | `docker run --rm -p 8888:8888 -v "$PWD/notebooks:/app/notebooks" -v "$PWD/outputs:/app/outputs" doe jupyter lab --ip=0.0.0.0 --no-browser --allow-root --ServerApp.token=''` |

Любую команду из таблицы «Основные команды» ниже можно запустить так же:
`docker run --rm doe` + команда, например `docker run --rm doe python scripts/check_env.py`.

Флаг `-v "$PWD/outputs:/app/outputs"` связывает папку на компьютере с папкой в контейнере. Без него
результаты останутся внутри контейнера и пропадут после его завершения.

- **Windows, PowerShell:** вместо `$PWD` писать `${PWD}`; в обычной командной строке (cmd) писать `%cd%`.
- **Код поменялся?** Пересоберите образ (`docker build -t doe .`), потому что код копируется в образ при сборке.
  Чтобы не пересобирать при каждой правке, можно смонтировать код целиком: `-v "$PWD:/app"`.
- **Mac:** Docker не видит видеоядро M1/M2/M3 (`mps`), поэтому в контейнере обучение идёт на CPU.
  Для обучения на Mac быстрее локальная установка (см. ниже), а Docker подходит для проверок, бейзлайнов и инференса.

## Без Docker (локально)

Нужен Python той же версии, что в первой строке `Dockerfile` (`FROM python:...`).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_examples.py
python scripts/check_env.py
```

## Если что-то не работает

### Четыре самых частых сценария

**1. Docker не запускается**

Ошибка: `docker: command not found`, `'docker' is not recognized` или `Cannot connect to the Docker daemon`.
Причина: Docker не установлен или не запущен.

```bash
docker --version             # нет ответа -> поставить Docker Desktop, перезапустить терминал
docker run --rm hello-world  # ошибка про daemon -> открыть Docker Desktop, дождаться «Engine running»
```

На Linux вместо Docker Desktop: `sudo systemctl start docker`.

**2. Сборка образа падает на установке пакетов**

Ошибка: `No matching distribution found for torch` или `manifest for python:... not found`.
Причина: под версию Python из `Dockerfile` ещё нет нужной сборки PyTorch или образа Python.
Что делать: в первой строке `Dockerfile` поставить версию постарше и собрать заново.

```bash
# в Dockerfile:  FROM python:3.13-slim
docker build -t doe .
```

**3. Результатов нет в `outputs/` или правки в коде не применяются**

Причина: контейнер работает со своей копией файлов.
- Нет результатов → в команде не хватает `-v "$PWD/outputs:/app/outputs"`. На Windows в PowerShell пишите `${PWD}`, в cmd — `%cd%`.
- Правки не видны → код копируется в образ при сборке. Пересоберите: `docker build -t doe .`

**4. Обучение обрывается**

Ошибка: `Killed`, код `137`, `Bus error` или `DataLoader worker ... killed`.
Причина: контейнеру не хватает памяти.
- Docker Desktop → Settings → Resources → Memory: 8 ГБ или больше.
- Добавить к `docker run` флаг `--shm-size=2g`.
- Или уменьшить нагрузку: `--batch 4 --workers 0`.

### Другие ошибки

| Ошибка | Что делать |
| --- | --- |
| `docker: command not found` / `'docker' is not recognized` | Docker не установлен или терминал открыт до установки. Поставить Docker Desktop, закрыть и заново открыть терминал. |
| `Cannot connect to the Docker daemon` / `error during connect` | Docker не запущен. Mac/Windows: открыть Docker Desktop и дождаться «Engine running». Linux: `sudo systemctl start docker`. |
| `permission denied while trying to connect to the Docker daemon socket` (Linux) | `sudo usermod -aG docker $USER`, затем выйти из системы и зайти снова. |
| Windows: `WSL 2 installation is incomplete` / `Virtualization support not detected` | В PowerShell от администратора: `wsl --update`, перезагрузиться. Если не помогло, включить виртуализацию (Intel VT-x / AMD SVM) в BIOS. |
| `manifest for python:... not found` при сборке | Такой версии Python нет в Docker Hub. Поменять первую строку `Dockerfile` на `FROM python:3.13-slim` и собрать заново. |
| `No matching distribution found for torch` | Под эту версию Python ещё нет сборки PyTorch. Взять версию Python постарше в `FROM` (например, 3.13) или ослабить ограничение `torch>=...` в `requirements.txt`. |
| Сборка падает на шаге `pytest` | Сломан код или физика: в выводе видно, какой тест упал. Исправить код и собрать заново. |
| `no space left on device` | Почистить старые образы: `docker system prune -a`. В Docker Desktop можно увеличить диск: Settings → Resources → Disk. |
| Процесс оборвался с `Killed` или кодом `137` | Контейнеру не хватило памяти. В Docker Desktop: Settings → Resources → Memory (поставить 8 ГБ и больше), либо уменьшить `--batch`. |
| `Bus error` / `DataLoader worker ... killed` | Мало общей памяти: добавить `--shm-size=2g` к `docker run` или запускать обучение с `--workers 0`. |
| `could not select device driver "" with capabilities: [[gpu]]` | Не установлен NVIDIA Container Toolkit (см. «Установка Docker»). На Mac GPU в Docker недоступен в принципе. |
| `port is already allocated` (Jupyter) | Порт 8888 занят. Запустить с `-p 8889:8888` и открыть http://localhost:8889 |
| После запуска пустая папка `outputs/` | Забыт флаг `-v "$PWD/outputs:/app/outputs"`. На Windows проверить, что вместо `$PWD` стоит `${PWD}` (PowerShell) или `%cd%` (cmd). |
| Правки в коде не применяются | Образ собран со старым кодом: `docker build -t doe .` или монтировать код через `-v "$PWD:/app"`. |
| Сборка очень долгая или скачивает гигабайты | Для CPU-образа PyTorch ставится с `download.pytorch.org/whl/cpu`, это нормально только в первый раз. Повторная сборка берёт кэш, пока не меняется `requirements.txt`. |

## Основные команды

| Что | Команда |
| --- | --- |
| Проверить окружение | `python scripts/check_env.py` |
| Тесты физики и формата | `python -m pytest -q` |
| Сгенерировать примеры наборов | `python scripts/make_examples.py` |
| Бейзлайны GS и Adam на наборе | `python scripts/run_baseline.py --targets data/examples/shapes` |
| Быстрая проверка обучения | `python scripts/train.py --steps 200 --batch 4 --base 16 --workers 0` |
| Полное обучение (лучше на GPU/Colab) | `python scripts/train.py --steps 3000 --batch 16` |
| Дообучить готовую сеть: штраф худшей плоскости + 5 итераций GS в модели | `python scripts/train.py --resume checkpoints/unet_20261106_2255/best.pt --steps 2000 --lr 3e-4 --w-worst 1 --gs-iters 5` |
| Финальный результат по ТЗ (последняя обученная сеть, папка с 4 PNG) | `python scripts/infer.py --weights "$(ls -td checkpoints/*/ \| head -1)best.pt" --targets data/examples/shapes` |
| То же + дотюнивание фазы | добавить `--refine 50` |

Результаты лежат в `outputs/`: `doe.png` (0 = −π, 255 = +π), `recon_1..4.png`, `metrics.json`.

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
