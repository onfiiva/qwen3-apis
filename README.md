# Qwen3 APIs

## English

### Qwen3 Inference API

Based on [Qwen3-4B-VL-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

### Qwen3 TTS API

Based on [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)

#### Installation

**Firstly move your models into models/ path**

1. Clone this repository to any path
```bash
git clone https://github.com/onfiiva/qwen3-apis
```

2. Open your console and move to project directory

3. Install [Python 3.12](https://www.python.org/downloads/latest/python3.12/) (!!!)

4. Check your python installation
```bash
python --version
```
OR
```bash
python3 --version
```

4. Create venv
```bash
python -m venv venv
```
OR
```bash
python3 -m venv venv
```

5. Activate venv
Windows
```bash
venv\Scripts\activate
```
MacOS / Linux
```bash
source venv/bin/activate
```

6. Install requirements in venv
```bash
pip install -r requirements.txt
```

7. Launch with
TTS API
```bash
uvicorn api.tts.main:app --host 0.0.0.0 --port 9000 --reload
```
Inference API
```bash
uvicorn api.inference.main:app --host 0.0.0.0 --port 9001 --reload
```
Where:
api.tts.main - is the path to main.py file (choose between TTS and Inference)
--host - IP address to get access to the API
--port - Port to get access to the API
--reload - Uvicorn dynamic reload function if something happens

8. Your API available on:
TTS - http://localhost:9000
Inference - http://localhost:9001

9. Swagger UI available on:
TTS - http://localhost:9000/docs
Inference - http://localhost:9001/docs


## Русский

### Qwen3 Inference API

Основана на [Qwen3-4B-VL-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

### Qwen3 TTS API

Основана на [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)

#### Установка

**Сначала переместите свои модели в папку models/**

1. Клонируйте этот репозиторий в любую папку
```bash
git clone https://github.com/onfiiva/qwen3-apis
```

2. Откройте свою консоль и перейдите в ней на вашу директорию склонированного репозитория

3. Установите [Python 3.12](https://www.python.org/downloads/latest/python3.12/) (!!!)

4. Проверьте установку Python
```bash
python --version
```
ИЛИ
```bash
python3 --version
```

4. Создайте venv
```bash
python -m venv venv
```
ИЛИ
```bash
python3 -m venv venv
```

5. Активируйте venv
Windows
```bash
venv\Scripts\activate
```
MacOS / Linux
```bash
source venv/bin/activate
```

6. Установите необходимые зависимости в venv
```bash
pip install -r requirements.txt
```

7. Запуск
TTS API
```bash
uvicorn api.tts.main:app --host 0.0.0.0 --port 9000 --reload
```
Inference API
```bash
uvicorn api.inference.main:app --host 0.0.0.0 --port 9001 --reload
```
Где:
api.tts.main - путь к файлу main.py (выбирайте между TTS и Inference)
--host - IP адрес доступа к API
--port - Порт доступа к API
--reload - Динамическая перезагрузка Uvicorn на случай, если что-то случится

8. Ваше API доступно по:
TTS - http://localhost:9000
Inference - http://localhost:9001

9. Swagger UI доступен по:
TTS - http://localhost:9000/docs
Inference - http://localhost:9001/docs
