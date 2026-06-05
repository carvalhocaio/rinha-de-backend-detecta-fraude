# Rinha de Backend 2026 — Fraud Detector (Python)

Implementação Python para o desafio de detecção de fraude da Rinha de Backend 2026.

## Endpoints

- `GET /ready` — health check; retorna `503` enquanto o índice FAISS carrega e `200` quando a API está pronta
- `POST /fraud-score` — recebe os dados da transação e retorna `approved` e `fraud_score`

## Como funciona

Cada requisição é transformada em um vetor de 14 dimensões conforme a especificação da Rinha (normalização de amount, parcelas, hora, dia da semana, distâncias, MCC risk, etc.) e comparada com o dataset de referência oficial (3 milhões de vetores) via busca aproximada de vizinhos mais próximos (FAISS IVF, `nprobe=4`, k=5). O score é calculado como `fraudes_entre_os_5_vizinhos / 5`; a transação é aprovada se `fraud_score < 0.6`.

## Stack

| Camada | Tecnologia |
|---|---|
| Runtime | Python 3.13 |
| Aplicação | ASGI puro (sem framework) |
| ASGI server | Granian + uvloop |
| Serialização | msgspec |
| Busca vetorial | FAISS IVF (quantização fp16) |
| Load balancer | HAProxy 3 (TCP, Unix sockets) |
| Empacotamento | uv + Docker multi-stage |

Recursos: 1 CPU core total, 350 MB RAM total (conforme limite da Rinha).

O índice FAISS é carregado em uma thread de background durante a inicialização, de modo que o socket do Granian fica disponível imediatamente — `/ready` responde `503` até o índice estar pronto, evitando timeouts de health check em ambientes com CPU limitada.

## Otimização: quantização fp16

O índice IVF usa quantização escalar **fp16** (16 bits) em vez de **SQ8** (8 bits). A quantização de 8 bits distorcia as distâncias L2 o suficiente para limitar a detecção a ~+2317, mesmo com recall quase exato. Medindo o teto com busca exata float (`IndexFlatL2`), o limite real é **+2790** (FP=4, FN=0) — ou seja, os erros residuais eram quase todos de quantização, não de recall.

Trocar para fp16 preserva essa precisão a um custo de latência praticamente nulo (mesmo número de clusters varridos por `nprobe`):

| Quantização | nprobe | detection_score | p99 (infra oficial) | score final |
|---|---|---|---|---|
| SQ8 | 4 | +1896 | 1,56 ms | +4702 |
| **fp16** | **4** | **+2126** | **1,71 ms** | **+4894** |

O índice cresce de 66 MB para 108 MB em disco, mas o `mmap` mantém o RSS em ~44 MB, bem dentro do limite de 167 MB por instância. Não é possível subir o `nprobe` para capturar mais recall: a 0,45 CPU o sistema satura (`nprobe=8` levou o p99 a ~393 ms), e como o p99_score é logarítmico, o ganho de detecção não compensa. Um índice HNSW (que quebraria o tradeoff recall×latência) não cabe nos 350 MB com 3 milhões de vetores.

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
