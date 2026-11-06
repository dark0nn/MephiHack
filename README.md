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

## Если установка упала: разбор всех сценариев

Найдите в таблице текст, который напечатал терминал (достаточно совпадения по нескольким словам),
и выполните шаги из правого столбца по порядку. Каждый шаг — отдельная команда или действие.

### Этап 1. Установка Docker

| Что напечатал терминал | Что это значит | Что делать |
| --- | --- | --- |
| `docker: command not found` (Mac/Linux) или `'docker' is not recognized as an internal or external command` (Windows) | Docker не установлен, или терминал открыт до установки | 1. Поставить Docker Desktop (раздел «Установка Docker» выше). 2. Закрыть терминал полностью и открыть заново. 3. Проверить: `docker --version` |
| `Cannot connect to the Docker daemon at unix://... Is the docker daemon running?` (Mac/Linux) | Docker установлен, но не запущен | Mac/Windows: открыть приложение Docker Desktop и подождать 1–2 минуты, пока внизу слева не появится «Engine running». Linux: `sudo systemctl start docker` |
| `error during connect` или `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` (Windows) | То же самое: Docker Desktop не запущен | Запустить Docker Desktop из меню «Пуск», дождаться «Engine running», повторить команду |
| `permission denied while trying to connect to the Docker daemon socket` (Linux) | Вашему пользователю не разрешено управлять Docker | 1. `sudo usermod -aG docker $USER` 2. Выйти из системы и войти заново (или перезагрузиться). 3. Проверить: `docker run --rm hello-world` |
| `WSL 2 installation is incomplete` или `WSL update required` (Windows) | Не установлена или устарела подсистема Linux для Windows | 1. Открыть PowerShell **от имени администратора**. 2. `wsl --install` 3. `wsl --update` 4. Перезагрузить компьютер. 5. Запустить Docker Desktop заново |
| `Virtualization support not detected` или `Hardware assisted virtualization ... must be enabled` (Windows) | В BIOS выключена виртуализация процессора | 1. Перезагрузиться и войти в BIOS (обычно клавиша F2, F10, Del или Esc при включении). 2. Включить пункт `Intel Virtualization Technology (VT-x)` или `SVM Mode` (AMD). 3. Сохранить (F10) и загрузиться. 4. Запустить Docker Desktop |
| Mac: «Docker.app не может быть открыто» или `incompatible CPU` | Скачан установщик не под тот процессор | Меню Apple → «Об этом Mac»: если «Чип Apple M…», скачать версию **Apple Silicon**; если «Процессор Intel», — версию **Intel Chip** |
| `docker run --rm hello-world` печатает `Unable to find image` и дальше ошибку сети | Docker не может скачивать образы из интернета | См. строку про `failed to resolve source metadata` в этапе 2 ниже |

### Этап 2. Сборка образа (`docker build -t doe .`)

| Что напечатал терминал | Что это значит | Что делать |
| --- | --- | --- |
| `failed to read dockerfile: open Dockerfile: no such file or directory` или `unable to prepare context` | Команда запущена не из папки проекта | 1. `cd ~/Documents/MephiHack` (Windows: `cd $HOME\Documents\MephiHack`). 2. Проверить, что там есть `Dockerfile`: `ls` (Windows: `dir`). 3. Повторить `docker build -t doe .` — **с точкой в конце** |
| `requires exactly 1 argument` (например, `"docker buildx build" requires exactly 1 argument`) | Забыта точка в конце команды | Писать ровно так: `docker build -t doe .` |
| `failed to resolve source metadata for docker.io/library/python...`, `TLS handshake timeout`, `i/o timeout`, `429 Too Many Requests` или `denied: ... access forbidden` | Docker не может скачать базовый образ Python из Docker Hub (часто бывает в РФ: Docker Hub ограничен) | 1. Docker Desktop → Settings (шестерёнка) → Docker Engine. 2. В окне с текстом сразу после первой `{` добавить строку `"registry-mirrors": ["https://mirror.gcr.io"],` (если в окне только `{}`, то запятая в конце не нужна). 3. Apply & Restart. 4. Повторить сборку. Linux: `echo '{"registry-mirrors": ["https://mirror.gcr.io"]}' \| sudo tee /etc/docker/daemon.json` (если файл уже есть, добавить строку в него), затем `sudo systemctl restart docker`. Если не помогло — включить VPN и повторить |
| `manifest for python:3.15-slim not found` | Такого образа Python нет | Открыть `Dockerfile`, в первой строке заменить на `FROM python:3.13-slim`, сохранить, повторить сборку |
| `No matching distribution found for torch` или `Could not find a version that satisfies the requirement torch` | Под эту версию Python ещё нет PyTorch | То же: в `Dockerfile` поставить `FROM python:3.13-slim` и собрать заново |
| `Read timed out`, `Connection broken`, `ProxyError` на шаге `pip install` | Обрывается скачивание пакетов (PyTorch весит сотни мегабайт) | 1. Просто повторить `docker build -t doe .` — скачанное сохраняется. 2. Если повторяется — включить или выключить VPN и попробовать снова |
| `no space left on device` | Закончилось место на диске для Docker | 1. `docker system prune -a` (удалит старые образы, на вопрос ответить `y`). 2. Docker Desktop → Settings → Resources → Disk: увеличить. 3. Освободить место на компьютере: нужно около 5 ГБ |
| `Killed` или `exit code: 137` на шаге `pip install` | Не хватило памяти при установке | Docker Desktop → Settings → Resources → Memory: поставить 4–8 ГБ, Apply & Restart, повторить сборку |
| Сборка падает на шаге `pytest` с `FAILED tests/...` | Код скачан не полностью или с ошибкой | 1. `git pull` — получить свежую версию. 2. Повторить сборку. 3. Если снова падает — прислать вывод разработчикам |
| Сборка идёт 10–20 минут и скачивает гигабайты | Это нормально в первый раз | Подождать. Следующие сборки занимают секунды |

### Этап 3. Запуск контейнера

| Что напечатал терминал | Что это значит | Что делать |
| --- | --- | --- |
| `Unable to find image 'doe:latest' locally` и затем `pull access denied` | Образ ещё не собран или собран под другим именем | Сначала `docker build -t doe .`, потом запуск |
| `invalid reference format` или `includes invalid characters for a local volume name` (Windows) | Терминал не подставил путь вместо `$PWD` | В PowerShell писать `${PWD}`, в обычной командной строке (cmd) — `%cd%`. Путь целиком в кавычках: `-v "${PWD}/outputs:/app/outputs"` |
| `The term 'head' is not recognized` или `$(ls ...)` не срабатывает (Windows) | Команда с `$(ls -td ... \| head -1)` работает только в bash (Mac, Linux, Git Bash) | Посмотреть папки: `dir checkpoints`, и вписать путь вручную: `--weights checkpoints/unet_20261106_2255/best.pt` |
| `port is already allocated` | Порт 8888 занят (например, уже запущен Jupyter) | Заменить `-p 8888:8888` на `-p 8889:8888` и открыть http://localhost:8889 |
| `could not select device driver "" with capabilities: [[gpu]]` | Нет поддержки видеокарты в Docker | Linux/Windows с NVIDIA: поставить NVIDIA Container Toolkit (ссылка в «Установка Docker»). Mac: видеокарта в Docker недоступна, запускать без `--gpus all` |
| `Killed`, код `137`, `Bus error` или `DataLoader worker ... killed` во время обучения | Контейнеру не хватает памяти | 1. Docker Desktop → Settings → Resources → Memory: 8 ГБ или больше. 2. Добавить в команду `--shm-size=2g`. 3. Или запускать обучение с `--batch 4 --workers 0` |
| Команда отработала, а папка `outputs/` пустая | Результаты остались внутри контейнера | Добавить в команду `-v "$PWD/outputs:/app/outputs"` (Windows PowerShell: `-v "${PWD}/outputs:/app/outputs"`) |
| Поменяли код, а результат прежний | Код копируется в образ при сборке | Пересобрать: `docker build -t doe .` |

Ничего не подошло? Скопируйте **весь** текст из терминала — от команды до последней строки — и отправьте разработчикам.

## Без Docker (локально)

Нужен Python той же версии, что в первой строке `Dockerfile` (`FROM python:...`).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_examples.py
python scripts/check_env.py
```

## Основные команды

| Что | Команда |
| --- | --- |
| Проверить окружение | `python scripts/check_env.py` |
| Тесты физики и формата | `python -m pytest -q` |
| Сгенерировать примеры наборов | `python scripts/make_examples.py` |
| Бейзлайны GS и Adam на наборе | `python scripts/run_baseline.py --targets data/examples/shapes` |
| Быстрая проверка обучения | `python scripts/train.py --steps 200 --batch 4 --base 16 --workers 0` |
| Полное обучение (лучше на GPU/Colab) | `python scripts/train.py --steps 3000 --batch 16` |
| Продолжить оборванное обучение с того же шага | `python scripts/train.py --continue checkpoints/unet_20261106_2255/last.pt` |
| Дообучить готовую сеть: штраф худшей плоскости + 5 итераций GS в модели | `python scripts/train.py --resume checkpoints/unet_20261106_2255/best.pt --steps 2000 --lr 3e-4 --w-worst 1 --gs-iters 5` |
| Новая версия сети поверх старой: совпадающие слои грузятся, новые учатся с нуля | `python scripts/train.py --resume checkpoints/unet_20261106_2255/best.pt --steps 1000 --lr 3e-4 --freeze-loaded 200` |
| Финальный результат по ТЗ (последняя обученная сеть, папка с 4 PNG) | `python scripts/infer.py --weights "$(ls -td checkpoints/*/ \| head -1)best.pt" --targets data/examples/shapes` |
| То же + дотюнивание фазы | добавить `--refine 50` |

Результаты лежат в `outputs/`: `doe.png` (0 = −π, 255 = +π), `recon_1..4.png`, `metrics.json`.

## Чекпоинты и дообучение

Каждые `--val-every` шагов в `checkpoints/<run>/` сохраняются `last.pt` (последний) и `best.pt` (лучший по val).
В них лежат веса, состояние оптимизатора, расписание lr, номер шага и параметры запуска.

- **`--continue checkpoints/<run>/last.pt`** продолжает тот же запуск: тот же шаг, тот же lr, лог дописывается.
  Подходит, если обучение оборвалось или ноутбук уснул. Параметры берутся из чекпоинта.
- **`--resume checkpoints/<run>/best.pt`** начинает новый запуск со старыми весами. Слои с тем же именем и формой
  загружаются, новые или изменённые слои начинают со случайных весов. В консоль выводится, что загрузилось, что новое и что
  пропущено. Подходит, если в `doe/model.py` добавили блок, поменяли голову или включили `--gs-iters`.
- **`--freeze-loaded N`** (вместе с `--resume`) первые N шагов учит только новые слои, чтобы они не испортили
  уже обученные. Потом размораживает всё.

Чтобы старые веса подхватились, не переименовывайте существующие слои в `UNet`: новые блоки добавляйте рядом
под новыми именами. Если поменять `--base` или `--depth`, совпадёт почти ничего, и сеть фактически учится заново.
Чекпоинты, сохранённые до этой правки, работают с `--resume`, но не с `--continue` (в них нет оптимизатора).

В Docker папку `checkpoints/` обязательно монтировать (`-v "$PWD/checkpoints:/app/checkpoints"`), иначе веса пропадут вместе с контейнером.

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
