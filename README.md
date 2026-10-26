# LocalBrain

LocalBrain — RAG-ассистент для вопросов по локальным текстовым документам. Он находит подходящие фрагменты в ChromaDB и формирует ответ на русском языке через Ollama. Генерация и эмбеддинги выполняются на вашем компьютере.

Стек: Python 3.13, FastAPI, Pydantic, LangChain, ChromaDB, Ollama и httpx. По умолчанию используются `qwen2.5:1.5b` для ответов и `nomic-embed-text` для эмбеддингов.

## Архитектура

```text
CLI (main.py) → FastAPI /chat → RAGBot (rag_bot.py)
                               ├─ эмбеддинг вопроса → ChromaDB: top-3
                               └─ вопрос + контекст → Ollama → ответ
ingest.py / FastAPI /ingest → чанки → эмбеддинги Ollama → ChromaDB
```

`app/ingestion.py` читает и индексирует документы, `rag_bot.py` собирает промпт через `ChatPromptTemplate`, `app/main.py` обслуживает HTTP-запросы. Настройки находятся в `app/config.py`.

`ingest.py` читает UTF-8-файлы `.txt` и `.md` непосредственно из `knowledge_base/`, пропуская вложенные каталоги и символические ссылки. Размер чанка — до 500 символов, перекрытие — 100. Текст, векторы и метаданные источника сохраняются на диск. Повторный запуск обновляет изменённые файлы без дубликатов и удаляет их устаревшие чанки. В каталоге есть четыре демодокумента о Python Backend.

## Локальный запуск на Windows

Нужны Python 3.13 в `PATH`, Git и [Ollama для Windows](https://ollama.com/download/windows). Все команды ниже выполняются в PowerShell из корня репозитория; активация `.venv` не требуется.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Переменные окружения имеют приоритет над `.env`. Относительные пути считаются от корня проекта. Основные настройки по умолчанию:

| Переменная | Значение |
| --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` |
| `DOCUMENTS_DIR` | `knowledge_base` |
| `CHROMA_PERSIST_DIR` | `data/chroma` |
| `UPLOADS_DIR` | `documents/uploads` |
| `API_BASE_URL` | `http://127.0.0.1:8000` |

Установите Ollama и откройте новый терминал.

Отключите облачные возможности демона: добавьте `"disable_ollama_cloud": true` в `~/.ollama/server.json` (в Windows — `%USERPROFILE%\.ollama\server.json`) и перезапустите Ollama. Переменная окружения Python-приложения не меняет настройки уже работающего демона. Перед отправкой текста приложение проверяет обе модели через `/api/show`: облачные модели и неподтверждённые метаданные блокируются, включая облачные модели под локальными псевдонимами.

Если сервер не запущен, выполните `ollama serve` в отдельном окне. Загрузите модели и создайте индекс до запуска API:

```powershell
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe -X utf8 test_ollama.py
.\.venv\Scripts\python.exe -X utf8 ingest.py
```

Запустите API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 8
```

В другом окне PowerShell, из того же каталога, запустите CLI:

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py
```

CLI отправляет вопросы в API по `API_BASE_URL`. Выход — `exit`, Ctrl+C или EOF.

## Docker Compose

Нужны Docker с Linux-контейнерами и плагин Compose. Конфигурация подготовлена, но её локальные сборка и запуск пока не проверялись.

Compose запускает `api` и `ollama` на CPU. API доступен на `127.0.0.1:8000`, а внутри сети обращается к `http://ollama:11434`. Имена моделей можно переопределить через `.env`, пути к данным задаются внутри контейнера.

```powershell
docker compose config --quiet
docker compose build api
docker compose up -d --wait ollama
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull nomic-embed-text
docker compose run --rm --no-deps api python ingest.py
docker compose up -d --wait api
docker compose ps
docker compose exec api python main.py
```

Модели Windows Ollama не переносятся в контейнер автоматически. Команды `pull` явно скачивают их в отдельный volume; при сборке загрузки моделей нет. Healthcheck API подтверждает доступность HTTP-приложения, но не готовность всей RAG-цепочки: наличие моделей, поиск и генерация проверяются отдельно.

ChromaDB хранится в `chroma_data`, загрузки — в `uploads_data`, модели — в `ollama_models`. API работает от непривилегированного пользователя. Для остановки с сохранением данных:

```powershell
docker compose down
```

## API

### POST /chat

```powershell
$body = @{ question = "Для чего Redis используется в backend?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Пример ответа; формулировка зависит от модели:

```json
{"answer": "Redis используется для кеширования данных, счётчиков и краткоживущего состояния."}
```

Для пустой существующей коллекции ответ — `{"answer":"Я пока ничего не знаю"}`, без вызова Ollama. Если индекс ещё не создан, сначала выполните `ingest.py`.

### POST /ingest

Принимает непустой UTF-8-файл `.txt` или `.md` размером до 2 МиБ. Пример загрузки демодокумента:

```powershell
curl.exe -F "file=@.\knowledge_base\fastapi.md" http://127.0.0.1:8000/ingest
```

```json
{"chunks_added": 4}
```

Загрузки сохраняются под хешированными именами в `UPLOADS_DIR` и сразу доступны для вопросов. Повторная загрузка с тем же именем без учёта регистра обновляет этот источник, не затрагивая демодокументы. `chunks_added` — число актуальных чанков источника после обработки.

| HTTP | Причина ошибки |
| --- | --- |
| 400 | Небезопасное имя файла или некорректный UTF-8 |
| 413 | Превышен лимит файла или тела запроса |
| 415 | Неподдерживаемое расширение |
| 422 | Некорректный JSON, пустой вопрос или пустой документ |
| 503 | Ollama недоступна или нужная модель не готова |
| 500 | Ошибка индекса, поиска или некорректный ответ модели |

Схема API доступна по `/openapi.json`; Swagger UI и ReDoc отключены.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

Тесты используют моки Ollama и временные хранилища. Workflow `.github/workflows/ci.yaml` настроен на PR в `main`: эти проверки, валидация Compose, сборка API-образа и проверка импортов и прав записи в volumes. Модели и настоящая генерация в CI не используются.

## Ограничения

- API предназначен для локального запуска и не содержит аутентификации.
- Нужен один Uvicorn worker: общий лимит допускает одну тяжёлую операцию. Не запускайте `ingest.py` параллельно API-запросам.
- Локальным моделям нужны свободные RAM и место на диске. В Compose заданы лимиты 1 ГиБ для API и 3 ГиБ для Ollama; GPU автоматически не подключается.
- Небольшая генеративная модель может ошибаться и неточно пересказывать контекст. Промпт требует краткого ответа, но не гарантирует его достоверность.
- Облачных LLM API и fallback нет. LangSmith и телеметрия FastAPI отключены; для офлайн-работы зависимости и модели нужно скачать заранее.
- Записи удалённых с диска файлов автоматически не очищаются. При исключении записи восстанавливается прежнее состояние источника. Пока обновление или восстановление не завершено, поиск блокируется меткой в ChromaDB, сохраняющейся после перезапуска. Полной атомарности при аварийном завершении процесса нет: потребуется повторная индексация источника. Файл и ChromaDB не объединены общей транзакцией; после сбоя повторите загрузку.
- `.env`, локальная база и `documents/` исключены из Git и Docker-образа. Приватные файлы загружайте через API, не помещайте их рядом с демодокументами в `knowledge_base/`.
