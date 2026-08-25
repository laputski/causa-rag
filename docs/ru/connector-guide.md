# Подключение внешнего RAG через causa-rag-client

Это руководство — для разработчика своего собственного RAG (не для этого репозитория), который хочет проверять его метриками этой платформы без переписывания внутренностей RAG.

## Зачем это и для кого

Платформа умеет считать одни и те же метрики (recall/precision, semantic similarity, послойная диагностика retrieval/rerank/generation) как для встроенного референс-пайплайна, так и для любого внешнего RAG — если тот реализует HTTP-контракт (`answer` + `sources[]`, опц. `trace`).

`causa-rag-client` — тонкий Python-пакет (`clients/python/`), который делает это подключение минимальным: пара вызовов клиента + (опционально) пара функций вместо рукописного HTTP-сервера. Eval (сами метрики, послойная атрибуция, регрессия) всегда считается **на платформе** — клиент только драйвит уже существующий REST API, ничего не вычисляет сам.

Если ваш RAG не на Python, или вы предпочитаете добавить эндпоинты напрямую в уже существующий сервис без новой зависимости — см. [external-rag-contract.md](external-rag-contract.md): полный языконезависимый референс контракта (все поля запроса/ответа, `corpus_id`-маршрутизация, тайм-ауты, чек-лист для хардненинга).

## Две роли пакета

`causa-rag-client` устанавливается с двумя независимыми наборами зависимостей:

```bash
pip install causa-rag-client                 # только клиент (httpx)
pip install causa-rag-client[serve]          # + serve()-хелпер (fastapi, uvicorn)
```

До первого релиза на PyPI ставится из клона:

```bash
pip install -e 'path/to/causa-rag/clients/python[serve]'
```

- **Клиент** (`RagPlatformClient`) — зовёт уже существующие эндпоинты платформы (`/external-rags`, `/experiments`, `/external-rag-spec`). Используете его, даже если ваш RAG уже выставлен по HTTP вручную.
- **`serve()`** — нужен только если у вас ЕЩЁ нет HTTP-эндпоинта для своего RAG и не хочется писать FastAPI самостоятельно.

**Рекомендация платформы**: клиентскую часть (`register_rag`/`check_contract_version`) стоит использовать всегда, если ваш RAG на Python — независимо от того, используете вы `serve()` или у вас уже есть свой полноценный HTTP-сервер (LangGraph, свой FastAPI и т.п.). Регистрация вручную одним `curl`, как это делалось до сих пор, не документируется в самом RAG-репозитории и не воспроизводима — `register_rag()`/`check_contract_version()` делают это частью кода вашего сервиса, а не разовым действием, которое некому будет повторить после переустановки.

Если вызываете `register_rag()` из `lifespan()`/стартапа своего сервиса (а не разовым скриптом) — учтите, что метод **не делает dedup**: повторный вызов на каждом рестарте создаёт новую запись каждый раз. Проверяйте `client.list_rags(realm_id=...)` на совпадение по `url` перед регистрацией — именно так это сделано в реальных внешних RAG'ах, зарегистрированных на платформе.

## Двухфазный поток

Подключение внешнего RAG — всегда две раздельные фазы, и путать их — типичная ошибка:

1. **Выставить RAG наружу** (одно из двух):
   - у вас уже есть HTTP-эндпоинт, отвечающий `{"answer": "...", "sources": [...]}` — ничего делать не нужно, просто передайте его URL.
   - у вас есть только функции `retrieve(query, top_k)`/`generate(query, sources)` — `serve()` оборачивает их в этот контракт за вас.
2. **Зарегистрировать и прогнать** — `RagPlatformClient` зовёт платформу: регистрирует URL, заливает контрольные вопросы, запускает прогон, забирает результат.

Платформа должна дозвониться до вашего RAG по HTTP — для локальной отладки это просто означает «оба на localhost», платформа и RAG. Для прод-RAG нужна прод-версия платформы с публичным egress (см. `RAG_HTTP_ALLOWLIST`) — это вне рамок v0.

## Фаза 1а — у вас уже есть HTTP-эндпоинт

Ничего не нужно от `serve()`. Просто зарегистрируйте URL (см. Фазу 2). Убедитесь, что ваш эндпоинт отвечает на `POST /` телом:

```json
{"query": "вопрос пользователя", "top_k": 5}
```

и возвращает:

```json
{"answer": "текст ответа", "sources": [{"doc_id": "...", "chunk_text": "..."}]}
```

`doc_id` — единственное обязательное для retrieval-метрик поле в каждом `source`; без него recall/precision не считаются (платформа не выдумывает идентификатор, который RAG не вернул).

## Фаза 1б — у вас только функции, без HTTP

```python
# my_rag/serve_app.py
from causa_rag_client import serve, run_server

def retrieve(query: str, top_k: int) -> list[dict]:
    results = my_index.search(query, top_k)
    return [{"doc_id": r.id, "chunk_text": r.text} for r in results]

def generate(query: str, sources: list[dict]) -> str:
    return my_llm.answer(query, [s["chunk_text"] for s in sources])

app = serve(retrieve, generate)

if __name__ == "__main__":
    run_server(app, port=8800)
```

Обслуживаете несколько корпусов одним сервисом? Добавьте `corpus_id` третьим параметром — `serve()` сам определяет по сигнатуре `retrieve`, ждать его или нет (2-арг функция продолжает работать как раньше, без изменений):

```python
def retrieve(query: str, top_k: int, corpus_id: str | None) -> list[dict]:
    # ВАЖНО: явно матчите corpus_id на СВОЁ реальное имя индекса/коллекции —
    # не предполагайте, что платформенный corpus_id совпадает с ним буквально
    # (полное совпадение — совпадение, не гарантия). Так уже ловили реальный
    # баг: `corpus_id="handbook"` платформы никак не совпадал с фактическим
    # именем коллекции на стороне RAG, retrieval молча падал на каждый вопрос.
    collection = CORPUS_TO_COLLECTION.get(corpus_id, DEFAULT_COLLECTION)
    results = my_index.search(query, top_k, collection=collection)
    return [{"doc_id": r.id, "chunk_text": r.text} for r in results]

app = serve(retrieve, generate)
```

`corpus_id` может быть `None`, если платформа его не передала — обрабатывайте это как «нет мнения», а не как ошибку.

Если у вашего RAG есть отдельный шаг реранкинга — передайте `rerank_fn(query, sources, top_k) -> sources`; платформа тогда сможет отличить «retrieval не нашёл нужный чанк» от «нашёл, но реранкер выбросил» (`core/eval/funnel.py`), вместо грубой атрибуции:

```python
app = serve(retrieve, generate, rerank_fn=my_rerank)
```

## Фаза 2 — регистрация и прогон

```python
from causa_rag_client import RagPlatformClient

client = RagPlatformClient("http://localhost:8081")  # адрес gateway платформы

rag = client.register_rag(name="my_rag", url="http://localhost:8800/", realm_id="my-realm")

client.upload_dataset(
    rag["id"],
    filename="golden.jsonl",
    realm_id="my-realm",
    questions=[
        {"id": "q1", "question": "...", "article_refs": ["CODE/1"], "answerability": "answerable"},
        # ...
    ],
)

run = client.run_experiment(name="my_rag_run", dataset_name="golden.jsonl", rag_id=rag["id"])
result = client.wait_for_completion(run["run_id"], timeout=300)
print(result["aggregate_metrics"])
```

`run_experiment` принимает либо `rag_id` (зарегистрированный URL), либо `http_endpoint` (URL напрямую, без регистрации) — ровно один из двух.

### Свойства запуска (properties)

| Параметр | Что делает |
|---|---|
| `rag_id` / `http_endpoint` | куда платформа шлёт запросы — ровно один из двух |
| `corpus_id` | непрозрачная строка платформы — она НЕ обязана совпадать с реальным именем вашей коллекции/индекса; матчинг на своё хранилище — ответственность вашего `retrieve_fn` (см. пример выше), не платформы |
| `pipeline_id` / `reranker_id` | какую встроенную стратегию дог-фудить — применимо только к `services/reference_rag_server` |
| `top_k` | сколько источников запрашивать |
| `metric_embedder_id` | **измерительный** эмбеддер платформы для semantic-метрик — НЕ retrieval-эмбеддер внутри вашего RAG (см. ниже) |
| `params` | расширяемый блок произвольных knobs (fetch_k, temperature, ...), см. ниже |

### Эмбеддер метрик ≠ retrieval-эмбеддер вашего RAG

`metric_embedder_id` выбирает эмбеддер, которым платформа **сама** считает semantic-метрики (`answer_similarity`, `context_support`, `grounded_in_correct_source`) — сравнивая ваш ответ/источники с эталоном. Это измерительный инструмент платформы, фиксированная линейка для сравнимости между разными RAG. Платформе совершенно не нужен и не интересен эмбеддер, которым ваш RAG сам делает retrieval внутри — она никогда его не видит, только `sources`/`answer`, которые вы вернули.

### Расширяемые params + capabilities

Если у вашего RAG есть knobs, которые хочется варьировать из Прогонов платформы (глубина выборки, температура генерации, версия промпта и т.п.), не предусмотренные выделенным полем (`top_k`/`corpus_id`/`reranker_id`) — передавайте их через `params={...}` в `run_experiment`. Декларируйте, какие ключи вы реально читаете, при регистрации:

```python
client.register_rag(name="my_rag", url="...", supported_params=["fetch_k", "temperature"])
```

Платформа не предполагает, что недекларированный ключ что-то делает — она просто передаёт `params` вашему RAG как есть; интерпретация целиком на вашей стороне.

## Какой эмбеддер использует корпус, с которым вас тестируют

Найдено вживую: RAG корректно резолвил `corpus_id` в правильную коллекцию, но
всё равно падал — эмбеддинг запроса считался своей моделью, а корпус был
проиндексирован другой (платформенным BGE-M3). Название корпуса/коллекции —
не гарантия того, чем она заполнена.

```python
embedders = client.get_corpus_embedders(realm_id="demo", corpus_id="handbook")
# ["bge_m3"]
```

Оба параметра обязательны — `corpus_id` один не гарантированно уникален между
Realm'ами. Список, не строка: у одного корпуса может быть больше одного
эмбеддера сразу (например, отдельно для dense-поиска и отдельно для
графовых community-эмбеддингов).

Дальше, в ответе на `retrieve`/`generate`, верните `embedders` в `trace` —
платформа сама сравнит с тем, чем корпус реально проиндексирован, и покажет
предупреждение при несовпадении на странице Ресурсов (без дополнительного
кода с вашей стороны, кроме возврата поля) — см. `embedders` в
[external-rag-contract.md](external-rag-contract.md#embedders--сообщите-платформе-каким-эмбеддером-вы-отвечали).

## Версия либы = версия контракта

`RagPlatformClient` при создании (если не передан `check_version=False`) сверяет свою версию с тем, что платформа публикует в `GET /external-rag-spec`. Несовпадение MAJOR-версии — явная ошибка `ContractVersionMismatch` с инструкцией обновить пакет, а не тихая поломка где-то посередине прогона. Несовпадение только minor/patch (аддитивное расширение контракта, например появление `embedders` выше) — не падение, а предупреждение (`warnings.warn`) при создании клиента: платформа не форсит апдейт SDK на каждую обратно-совместимую добавку, просто напоминает, что она доступна.

## Как читать результат

`result["aggregate_metrics"]` — те же метрики, что показывает страница «Прогон» для встроенного пайплайна: `correct_refusal` для всех вопросов; `retrieval_recall_at_k`/`retrieval_precision_at_k`/`answer_similarity`/`context_support`/`grounded_in_correct_source`/`citation_number_coverage` только для `"answerable"` вопросов вашего датасета.

Если вы передали `rerank_fn` в `serve()`, в полном результате (`GET /experiments/{run_id}` через UI или `client.get_results`) для каждого вопроса будет посчитан `funnel`-вердикт — какой слой (retrieval/rerank/generation) подвёл, если подвёл.

## Тестирование вашей интеграции

Перед полноценным прогоном — `client.test_rag(rag["id"])`: один реальный запрос, отчёт `connection_tier` и превью распарсенных источников. Плохой `doc_id`/маппинг тогда виден сразу, а не маскируется под «плохое качество retrieval» после полного прогона на сотне вопросов.

## Какой моделью разрабатывать (для агентов/Claude Code)

Если развитие коннектора делает агент:
- Дизайн контракта, `funnel.py`-атрибуция, async job-model, golden-parity E2E — задачи, где тонкая корректность важнее скорости — **Opus**.
- Рутинный codegen клиента, методы-обёртки, скаффолд тестов, этот гайд — **Sonnet**.

## Полный рабочий пример

См. `examples/connector_quickstart/` — игрушечный RAG → `serve()` → `run_experiment()` → результаты, без `curl` и без единой строчки FastAPI.
