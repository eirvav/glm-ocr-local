# GLM-OCR Batch Runner

This project runs the open-source `zai-org/GLM-OCR` model against PDF files and writes extracted markdown.

## What was optimized

- Single model load for many documents (critical for large runs).
- Batch directory mode (`--input-dir`) so one process handles thousands of PDFs.
- Adaptive page batching with automatic OOM downshift.
- Fast defaults tuned for throughput:
  - `--dpi 72`
  - `--max-new-tokens 256`
- Resume support by default (`skip existing outputs` unless `--overwrite` is set).
- Throughput metrics printed after every run.

## Requirements

- Python 3.11+ (arm64 strongly recommended on Apple Silicon).
- For `transformers>=5.2.0`, use PyTorch `>=2.4`.
- Enough RAM/VRAM for your chosen settings.

Install dependencies:

```bash
cd /Users/eirvav/PythonProjects/marketing-ai-project
./venv/bin/pip install -r requirements.txt
```

If you need to create a new venv:

```bash
cd /Users/eirvav/PythonProjects/marketing-ai-project
python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install -r requirements.txt
```

## Usage

### Single PDF

```bash
cd /Users/eirvav/PythonProjects/marketing-ai-project
./venv/bin/python /Users/eirvav/PythonProjects/marketing-ai-project/model.py \
  --input-pdf /Users/eirvav/PythonProjects/marketing-ai-project/data/CoverLetterMar042004drft.pdf \
  --output-md /Users/eirvav/PythonProjects/marketing-ai-project/data/CoverLetterMar042004drft.md \
  --dpi 72 \
  --max-new-tokens 256
```

### Batch mode (recommended for scale)

```bash
cd /Users/eirvav/PythonProjects/marketing-ai-project
./venv/bin/python /Users/eirvav/PythonProjects/marketing-ai-project/model.py \
  --input-dir /Users/eirvav/PythonProjects/marketing-ai-project/data \
  --recursive \
  --glob "*.pdf" \
  --output-dir /Users/eirvav/PythonProjects/marketing-ai-project/data \
  --dpi 72 \
  --max-new-tokens 256
```

## Key CLI options

- `--input-pdf`: process one PDF.
- `--input-dir`: process many PDFs in one run.
- `--recursive`: recursive scan for batch mode.
- `--glob`: file pattern inside `--input-dir` (default `*.pdf`).
- `--output-md`: single-file output path.
- `--output-dir`: root output directory for batch mode.
- `--overwrite`: reprocess outputs that already exist.
- `--dpi`: lower = faster and lower memory; higher = better fidelity.
- `--max-new-tokens`: lower = faster and lower memory.
- `--batch-size`: pages per inference call (`0` = auto).
- `--device`: `auto|cuda|mps|cpu`.
- `--dtype`: `auto|float16|bfloat16|float32`.
- `--attn-implementation`: `auto|sdpa|eager`.
- `--use-slow-processor`: disables fast image processor.

## Throughput tuning for thousands of docs

1. Keep one long-running batch invocation (avoid per-document startup).
2. Start with `--dpi 72 --max-new-tokens 256`.
3. Increase `--batch-size` only if memory allows.
4. Use CUDA GPUs for highest throughput.
5. Run multiple workers by sharding input directories across processes/machines.
6. Keep `--overwrite` off for resumable, restart-safe runs.

## Notes

- Default output is markdown text concatenated across pages.
- If you hit memory errors, reduce `--dpi`, `--max-new-tokens`, or `--batch-size`.
- If extracted text looks cut off, increase `--max-new-tokens` (for example to `512` or `1024`).
