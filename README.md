# annotate-tables

Pre-annotate tables in PDFs for correction in CVAT.

## Setup

Needs [uv](https://docs.astral.sh/uv/), Python 3.14 or later (uv installs it) and `pdftoppm` from poppler on the path.

```sh
sudo apt install poppler-utils  # or: brew install poppler
uv sync
```

The CVAT steps need an account on a running CVAT server:

```sh
export CVAT_USER=<user>
export CVAT_PASSWORD=<password>
export CVAT_HOST=http://<host>:8080  # optional, overrides the default server
```

Model weights are downloaded from Hugging Face on first use. A CUDA GPU is used when available, otherwise CPU.

## Usage

```sh
# 1. Fetch a PDF, detect its tables and create a CVAT task to correct them.
uv run annotate-tables <pdf-url-or-path> --push

# 2. Crop the corrected tables, pre-label rows, columns and spans, and create a structure task.
uv run annotate-structure <table-task-id>

# 3. Export the corrected structure task as a COCO dataset.
uv run export-structure-coco <structure-task-id>
```

Everything is written under `data/<document>/`.

### Web UI

```sh
uv run annotate-web --host <address> --port 8000
```

Runs the same three steps from one page, with a button for each document's next step and links to its CVAT tasks.
PDFs are added by URL or uploaded from the browser.
Progress is kept in `data/<document>/state.json`; documents without that file are not listed. There is no login, so
listen on a private address such as the machine's Tailscale IP rather than `0.0.0.0`. The page only answers under
the `--host` address and `localhost`; add other names you open it by with `--allow-host <name>`.

## Model selection

Last reviewed: 2026-10-06. Chosen on one hand-annotated document (27 pages, 31 ruled Japanese tables), so re-run
the comparison when more annotated documents exist. All models are Apache-2.0.

| Stage | Model | Why |
|---|---|---|
| Table detection | `PaddlePaddle/PP-DocLayoutV3_safetensors` | Found 30 of 31 tables; `docling-layout-heron` found 10. |
| Wired / wireless routing | `PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors` | Picks the cell detector per table. |
| Table structure | `PaddlePaddle/RT-DETR-L_wired_table_cell_det_safetensors`, `..._wireless_...` | 98% of rows, 73% of columns and 72% of spans usable as predicted (IoU 0.8). Table Transformer v1.1 reached 38% of rows and no spans. |

Also tried and dropped: docling heron-101 and egret (detection, no better than heron), PP-DocLayoutV2 and
PP-DocLayout_plus-L (detection, slightly behind V3), SLANeXt (structure, no gain over the cell detectors). Not tried
for licence reasons: Ultralytics YOLO layout models (AGPL-3.0) and Surya (non-commercial weights).
