# annotate-tables

Pre-annotate tables in PDFs for correction in CVAT.

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
