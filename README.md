# BlurArroba

API em FastAPI que detecta texto contendo **@** (arroba) em vídeos e aplica **blur/pixelização** sobre essas regiões — útil para anonimizar @usuários/handles que aparecem na tela.

A detecção de texto usa [EasyOCR](https://github.com/JaidedAI/EasyOCR) (PyTorch), o processamento de quadros usa OpenCV, e o áudio original é remultiplexado com FFmpeg no resultado final.

## Como funciona

1. O vídeo é lido quadro a quadro com OpenCV.
2. A cada `sample_interval` quadros, o EasyOCR procura textos; caixas cujo texto contém `@` (acima de `min_conf`) entram na lista de regiões.
3. As caixas são rastreadas entre os quadros amostrados (IoU + TTL), com um padding ao redor, e recebem blur gaussiano ou pixelização.
4. O vídeo processado (sem áudio) é gerado e o áudio original é remultiplexado via FFmpeg (`libx264`/`aac`).

Arquivos principais:

| Arquivo | Responsabilidade |
|---|---|
| [`main.py`](main.py) | API HTTP (FastAPI) — upload, fila de jobs e download |
| [`detector.py`](detector.py) | Detecção de `@` com EasyOCR |
| [`processor.py`](processor.py) | Pipeline de vídeo (tracking, blur, remux) |
| [`modal_app.py`](modal_app.py) | Deploy serverless com GPU no Modal.com |
| [`frontend.html`](frontend.html) | Front web (upload, progresso, download) |
| [`test_run.py`](test_run.py) | Runner local para testar direto num arquivo |

## Requisitos

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) disponível no `PATH`
- (Opcional) GPU NVIDIA com CUDA para acelerar o OCR

## Instalação

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Acelerar com GPU (NVIDIA / CUDA)

O `pip install torch` padrão instala a build **somente-CPU**. Para usar a GPU, reinstale o PyTorch com a build CUDA correspondente ao seu driver (exemplo CUDA 12.4):

```bash
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verifique:

```bash
python -c "import torch; print(torch.cuda.is_available())"  # deve imprimir True
```

> Em testes locais (RTX 3050 Laptop), um vídeo de 766 quadros processou em **~64 s na GPU** contra **~559 s na CPU** (~8,8x mais rápido).

## Executando a API

```bash
# Habilita GPU (opcional)
# Windows PowerShell:  $env:USE_GPU="1"
# Linux/macOS:         export USE_GPU=1

uvicorn main:app --host 0.0.0.0 --port 8000
```

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `USE_GPU` | `0` | `1` para usar a GPU no OCR |
| `UPLOAD_DIR` | `/tmp/atblur/in` | Diretório de uploads |
| `OUTPUT_DIR` | `/tmp/atblur/out` | Diretório de saída |

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/jobs` | Envia o vídeo e cria um job |
| `GET` | `/jobs/{job_id}` | Consulta status/progresso |
| `GET` | `/jobs/{job_id}/download` | Baixa o vídeo processado |

Parâmetros do `POST /jobs` (multipart form):

| Campo | Padrão | Descrição |
|---|---|---|
| `file` | — | Vídeo (`.mp4 .mov .mkv .webm .avi`) |
| `mode` | `pixelate` | `pixelate` ou `gaussian` |
| `strength` | `14` | Intensidade do efeito |
| `detect_scale` | `1.0` | Escala da imagem antes do OCR (menor = mais rápido) |
| `min_conf` | `0.30` | Confiança mínima do OCR |
| `sample_interval` | `0` | Quadros entre detecções (`0` = automático: ~fps/4) |

### Exemplo (cURL)

```bash
# cria o job
curl -F "file=@meu_video.mp4" -F "mode=pixelate" http://localhost:8000/jobs
# {"job_id":"abc...","status":"queued"}

# consulta o status
curl http://localhost:8000/jobs/abc...

# baixa quando status == done
curl -OJ http://localhost:8000/jobs/abc.../download
```

## Deploy serverless com GPU (Modal.com)

[`modal_app.py`](modal_app.py) adapta o projeto para o [Modal](https://modal.com): a GPU é **serverless** — só "acorda" quando chega um vídeo, carrega o EasyOCR uma vez e fica quente por alguns minutos. O front [`frontend.html`](frontend.html) é servido pelo próprio app.

```bash
pip install modal
modal token new            # autentica (uma vez, abre o navegador)

modal serve modal_app.py   # dev: URL temporária com hot-reload
modal deploy modal_app.py  # produção: URL permanente
```

Após o `serve`/`deploy`, o Modal imprime a URL do web app (ex.: `https://<voce>--blur-arroba-web.modal.run`). Abra no navegador: o front faz upload, acompanha o progresso e baixa o resultado.

Fluxo:

1. Ao **selecionar** o vídeo, o front chama `POST /wake`, que dispara o cold-start do container GPU (a GPU acorda enquanto o upload acontece).
2. `POST /jobs` envia o vídeo; o processamento roda em background na GPU (`Blurrer.process.spawn`) e o progresso é gravado num `modal.Dict`.
3. O front faz polling em `GET /jobs/{id}` e, ao concluir, baixa via `GET /jobs/{id}/download` (arquivo vem de um `modal.Volume`).

> A GPU usada é `T4`. Ajuste em `@app.cls(gpu=...)` (ex.: `"A10G"`, `"L4"`) e o tempo que fica quente em `scaledown_window`. Testado com `modal==1.4.3`.

## Teste local rápido (sem subir a API)

```bash
python test_run.py meu_video.mp4
```

Gera `out_blurred_gpu.mp4` ao lado do arquivo de entrada.

## Notas

- O `ThreadPoolExecutor` usa `max_workers=1`, serializando os jobs (um modelo na memória, sem disputa de VRAM). Para produção, troque por uma fila (Celery/RQ + Redis) com N workers.
- O armazenamento de jobs é em memória — reiniciar a API perde o estado.
