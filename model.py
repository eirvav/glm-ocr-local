#!/usr/bin/env python3
"""High-throughput GLM-OCR runner for single PDFs or entire directories."""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import fitz
import torch
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

DEFAULT_MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_PROMPT = "Convert this page to markdown and preserve all visible text and structure."
OOM_MARKERS = (
    "out of memory",
    "invalid buffer size",
    "failed to allocate private mtlbuffer",
    "mps backend out of memory",
)


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    dtype: torch.dtype
    attn_implementation: str | None
    batch_size: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_pdf = repo_root / "data" / "CoverLetterMar042004drft.pdf"

    parser = argparse.ArgumentParser(
        description="Extract markdown from PDF(s) with zai-org/GLM-OCR."
    )
    parser.add_argument(
        "--input-pdf",
        type=Path,
        default=None,
        help="Single PDF path. If omitted and --input-dir is unset, uses data/CoverLetterMar042004drft.pdf.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of PDFs to process in one run (loads model once).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --input-dir for PDFs.",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Glob pattern for PDF discovery inside --input-dir.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path for single-PDF mode. Defaults to input path with .md suffix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for batch mode. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files. Default behavior skips existing outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of PDFs in batch mode (0 = no cap).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=72,
        help="Render DPI for PDF pages. Lower is faster; 72 is throughput-optimized.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per page.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used per page.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Pages per generation call. 0 auto-selects (4 for CUDA, 2 for CPU, 1 for MPS).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "sdpa", "eager"),
        default="auto",
        help="Attention kernel selection.",
    )
    parser.add_argument(
        "--use-slow-processor",
        action="store_true",
        help="Use slow image processor. Default uses fast processor for throughput.",
    )

    args = parser.parse_args()
    if args.input_pdf is None and args.input_dir is None:
        args.input_pdf = default_pdf
    return args


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def resolve_runtime(args: argparse.Namespace) -> RuntimeConfig:
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        if device.type in ("cuda", "mps"):
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = _dtype_from_name(args.dtype)

    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    attn_implementation = None if args.attn_implementation == "auto" else args.attn_implementation

    if args.batch_size > 0:
        batch_size = args.batch_size
    elif device.type == "cuda":
        batch_size = 4
    elif device.type == "cpu":
        batch_size = 2
    else:
        batch_size = 1

    return RuntimeConfig(
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        batch_size=max(batch_size, 1),
    )


def discover_pdfs(args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    if args.input_dir is not None:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Expected directory: {input_dir}")

        if args.recursive:
            pdf_paths = sorted(p for p in input_dir.rglob(args.glob) if p.is_file())
        else:
            pdf_paths = sorted(p for p in input_dir.glob(args.glob) if p.is_file())

        if args.limit > 0:
            pdf_paths = pdf_paths[: args.limit]
        return pdf_paths, input_dir

    input_pdf = args.input_pdf.expanduser().resolve()
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    return [input_pdf], None


def output_path_for(
    pdf_path: Path,
    args: argparse.Namespace,
    input_root: Path | None,
) -> Path:
    if input_root is None:
        if args.output_md is not None:
            return args.output_md.expanduser().resolve()
        return pdf_path.with_suffix(".md")

    output_root = args.output_dir.expanduser().resolve() if args.output_dir else input_root
    relative = pdf_path.relative_to(input_root).with_suffix(".md")
    return output_root / relative


def render_pdf_to_images(pdf_path: Path, dpi: int, image_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = image_dir / f"page_{index + 1:04d}.png"
            pix.save(image_path)
            image_paths.append(image_path)

    return image_paths


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def is_oom_error(error: RuntimeError) -> bool:
    error_text = str(error).lower()
    return any(marker in error_text for marker in OOM_MARKERS)


def ocr_batch(
    image_paths: list[Path],
    prompt: str,
    processor: AutoProcessor,
    model: GlmOcrForConditionalGeneration,
    max_new_tokens: int,
) -> list[str]:
    batch_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image_path in image_paths
    ]

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    model_inputs = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    if generated.dim() == 1:
        generated = generated.unsqueeze(0)

    input_ids = model_inputs["input_ids"]
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )
    else:
        input_lengths = input_ids.ne(pad_token_id).sum(dim=1)

    decoded: list[str] = []
    for row in range(generated.shape[0]):
        start = int(input_lengths[row].item())
        new_tokens = generated[row, start:]
        decoded.append(processor.decode(new_tokens, skip_special_tokens=True).strip())
    return decoded


def ocr_pages_with_adaptive_batching(
    image_paths: list[Path],
    prompt: str,
    processor: AutoProcessor,
    model: GlmOcrForConditionalGeneration,
    max_new_tokens: int,
    target_batch_size: int,
) -> list[str]:
    if not image_paths:
        return []

    results: list[str] = []
    index = 0
    batch_size = max(target_batch_size, 1)

    while index < len(image_paths):
        chunk = image_paths[index : index + batch_size]
        try:
            chunk_results = ocr_batch(
                image_paths=chunk,
                prompt=prompt,
                processor=processor,
                model=model,
                max_new_tokens=max_new_tokens,
            )
            results.extend(chunk_results)
            index += len(chunk)
        except RuntimeError as runtime_error:
            if is_oom_error(runtime_error) and len(chunk) > 1:
                clear_device_cache(model.device)
                batch_size = max(1, len(chunk) // 2)
                continue
            raise

    return results


def process_pdf(
    pdf_path: Path,
    output_md: Path,
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    processor: AutoProcessor,
    model: GlmOcrForConditionalGeneration,
) -> tuple[int, float]:
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="glm_ocr_pages_") as temp_dir:
        image_paths = render_pdf_to_images(pdf_path, args.dpi, Path(temp_dir))
        page_markdown = ocr_pages_with_adaptive_batching(
            image_paths=image_paths,
            prompt=args.prompt,
            processor=processor,
            model=model,
            max_new_tokens=args.max_new_tokens,
            target_batch_size=runtime.batch_size,
        )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = "\n\n".join(text for text in page_markdown if text).strip()
    output_md.write_text(f"{markdown_text}\n", encoding="utf-8")

    duration = time.perf_counter() - start
    return len(page_markdown), duration


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime(args)
    pdf_paths, input_root = discover_pdfs(args)
    if not pdf_paths:
        raise FileNotFoundError("No PDF files were found with the provided inputs.")

    print(
        "Loading model "
        f"{args.model_id} on {runtime.device.type} "
        f"(dtype={runtime.dtype}, batch_size={runtime.batch_size})..."
    )

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    processor = AutoProcessor.from_pretrained(
        args.model_id, use_fast=not args.use_slow_processor
    )

    model_kwargs = {"dtype": runtime.dtype}
    if runtime.attn_implementation is not None:
        model_kwargs["attn_implementation"] = runtime.attn_implementation

    model = GlmOcrForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    model.to(runtime.device)
    model.eval()

    processed_docs = 0
    skipped_docs = 0
    total_pages = 0
    total_runtime = 0.0
    run_start = time.perf_counter()

    for index, pdf_path in enumerate(pdf_paths, start=1):
        output_md = output_path_for(pdf_path, args, input_root)
        if output_md.exists() and not args.overwrite:
            skipped_docs += 1
            print(f"[{index}/{len(pdf_paths)}] Skipped existing: {output_md}")
            continue

        pages, seconds = process_pdf(
            pdf_path=pdf_path,
            output_md=output_md,
            args=args,
            runtime=runtime,
            processor=processor,
            model=model,
        )
        processed_docs += 1
        total_pages += pages
        total_runtime += seconds

        print(
            f"[{index}/{len(pdf_paths)}] {pdf_path.name}: "
            f"{pages} pages in {seconds:.2f}s -> {output_md}"
        )

    wall_seconds = time.perf_counter() - run_start
    pages_per_second = total_pages / total_runtime if total_runtime > 0 else 0.0
    docs_per_min = (processed_docs / wall_seconds) * 60 if wall_seconds > 0 else 0.0

    print("")
    print("Run summary")
    print(f"Processed docs: {processed_docs}")
    print(f"Skipped docs: {skipped_docs}")
    print(f"Processed pages: {total_pages}")
    print(f"OCR compute time: {total_runtime:.2f}s")
    print(f"Wall-clock time: {wall_seconds:.2f}s")
    print(f"Throughput: {pages_per_second:.2f} pages/s, {docs_per_min:.2f} docs/min")


if __name__ == "__main__":
    main()
