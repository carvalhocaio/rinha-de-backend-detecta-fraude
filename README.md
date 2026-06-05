# Rinha de Backend 2026 — Fraud Detector (Python)

Implementação Python para o desafio de detecção de fraude da Rinha de Backend 2026.

## Endpoints

- `GET /ready` — health check; retorna `503` enquanto o índice FAISS carrega e `200` quando a API está pronta
- `POST /fraud-score` — recebe os dados da transação e retorna `approved` e `fraud_score`

## Como funciona

Cada requisição é transformada em um vetor de 14 dimensões conforme a especificação da Rinha (normalização de amount, parcelas, hora, dia da semana, distâncias, MCC risk, etc.) e comparada com o dataset de referência oficial via busca aproximada de vizinhos mais próximos (FAISS IVF, k=5). O score é calculado como `fraudes_entre_os_5_vizinhos / 5`; a transação é aprovada se `fraud_score < 0.6`.

## Stack

| Camada | Tecnologia |
|---|---|
| Runtime | Python 3.13 |
| Aplicação | ASGI puro (sem framework) |
| ASGI server | Granian + uvloop |
| Serialização | msgspec |
| Busca vetorial | FAISS IVF (SQ8) |
| Load balancer | HAProxy 3 (TCP, Unix sockets) |
| Empacotamento | uv + Docker multi-stage |

Recursos: 1 CPU core total, 350 MB RAM total (conforme limite da Rinha).

O índice FAISS é carregado em uma thread de background durante a inicialização, de modo que o socket do Granian fica disponível imediatamente — `/ready` responde `503` até o índice estar pronto, evitando timeouts de health check em ambientes com CPU limitada.

## Como rodar localmente

```bash
docker compose up --build
curl http://localhost:9999/ready
```

## Como testar

```bash
# Validação de detecção (requer test/test-data.json)
.venv/bin/python validate_detection.py

# Load test (requer k6)
k6 run carga.js
```

## Compliance

O índice FAISS é construído exclusivamente a partir de `resources/references.json.gz` (dataset oficial da Rinha), processado em tempo de build da imagem Docker. Nenhum payload de teste, prévia ou resultado esperado é usado como dado de referência ou critério de decisão. O campo `id` da requisição é ignorado pela lógica de detecção.
