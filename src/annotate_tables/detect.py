from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

TABLE_DETECTOR = 'PaddlePaddle/PP-DocLayoutV3_safetensors'
TABLE_CLASSIFIER = 'PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors'
CELL_MODELS = {
    'wired_table': 'PaddlePaddle/RT-DETR-L_wired_table_cell_det_safetensors',
    'wireless_table': 'PaddlePaddle/RT-DETR-L_wireless_table_cell_det_safetensors',
}
STRUCTURE_LABELS = ('column', 'row', 'span')
# White border added around a crop for cell detection only; 10 beat 0 and 20 on hand-annotated tables.
INFERENCE_PAD = 10


def load_detector(repo: str) -> tuple[Any, Any]:
    import torch  # noqa: PLC0415 -- slow. Use `lazy` on 3.15
    from transformers import (  # noqa: PLC0415 -- imports torch internally.
        AutoImageProcessor,
        AutoModelForObjectDetection,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return AutoImageProcessor.from_pretrained(repo), AutoModelForObjectDetection.from_pretrained(repo).to(device).eval()


def run_detector(processor, model, image: Image.Image, threshold: float) -> dict:
    import torch  # noqa: PLC0415 -- slow. Use `lazy` on 3.15

    inputs = processor(images=image, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]], threshold=threshold)[0]


def detect_tables(pages: list[Path], threshold: float) -> list[dict]:
    processor, model = load_detector(TABLE_DETECTOR)

    records = []
    for page in pages:
        image = Image.open(page).convert('RGB')
        result = run_detector(processor, model, image, threshold)
        tables = [
            {
                'label': 'table',
                'box': [round(v, 2) for v in box.tolist()],
                'score': round(score.item(), 4),
            }
            for score, label, box in zip(result['scores'], result['labels'], result['boxes'], strict=False)
            if model.config.id2label[label.item()].lower() == 'table'
        ]
        records.append(
            {
                'image': page.name,
                'width': image.width,
                'height': image.height,
                'boxes': tables,
            }
        )
        print(f'{page.name}: {len(tables)} table(s)')
    return records


def grid_lines(edges: list[float], tol: float) -> list[float]:
    clusters: list[list[float]] = []
    for edge in sorted(edges):
        if clusters and edge - clusters[-1][-1] <= tol:
            clusters[-1].append(edge)
        else:
            clusters.append([edge])
    return [sum(c) / len(c) for c in clusters]


def cells_to_structure(cells: Sequence[Sequence[float]], tol: float) -> list[dict]:
    """Turn cell boxes into row, column and span boxes.

    Cell edges within tol pixels of each other become one grid line. Rows and columns are the gaps between
    consecutive lines, and a cell crossing more than one gap is a span.
    """
    if not cells:
        return []
    xs = grid_lines([v for c in cells for v in (c[0], c[2])], tol)
    ys = grid_lines([v for c in cells for v in (c[1], c[3])], tol)
    boxes = [('row', [xs[0], top, xs[-1], bottom]) for top, bottom in pairwise(ys)]
    boxes += [('column', [left, ys[0], right, ys[-1]]) for left, right in pairwise(xs)]
    for cell in cells:
        x1, x2 = (min(range(len(xs)), key=lambda i: abs(xs[i] - v)) for v in (cell[0], cell[2]))
        y1, y2 = (min(range(len(ys)), key=lambda i: abs(ys[i] - v)) for v in (cell[1], cell[3]))
        if x2 - x1 > 1 or y2 - y1 > 1:
            boxes.append(('span', [xs[x1], ys[y1], xs[x2], ys[y2]]))
    return [{'label': label, 'box': [round(v, 2) for v in box]} for label, box in boxes]


def expand_to_edges(boxes: list[dict], width: int, height: int) -> list[dict]:
    """Stretch rows to the full image width and columns to the full image height."""
    expanded = []
    for b in boxes:
        x1, y1, x2, y2 = b['box']
        if b['label'] == 'row':
            x1, x2 = 0, width
        elif b['label'] == 'column':
            y1, y2 = 0, height
        expanded.append(b | {'box': [x1, y1, x2, y2]})
    return expanded


def detect_structure(tables_dir: Path, names: list[str], threshold: float, tol: float) -> list[dict]:
    import torch  # noqa: PLC0415 -- slow. Use `lazy` on 3.15
    from transformers import (  # noqa: PLC0415 -- imports torch internally.
        AutoImageProcessor,
        AutoModelForImageClassification,
    )

    detectors = {kind: load_detector(repo) for kind, repo in CELL_MODELS.items()}
    device = detectors['wired_table'][1].device
    classifier_processor = AutoImageProcessor.from_pretrained(TABLE_CLASSIFIER)
    # The checkpoint is stored in half precision, which fails on float inputs.
    classifier = AutoModelForImageClassification.from_pretrained(TABLE_CLASSIFIER, dtype=torch.float32)
    classifier = classifier.to(device).eval()

    records = []
    for name in names:
        image = Image.open(tables_dir / name).convert('RGB')
        with torch.no_grad():
            # PPLCNetForImageClassification returns its logits as the first output, not as `.logits`.
            logits = classifier(**classifier_processor(images=image, return_tensors='pt').to(device))[0]
        kind = classifier.config.id2label[logits.argmax().item()]
        result = run_detector(*detectors[kind], ImageOps.expand(image, INFERENCE_PAD, fill='white'), threshold)
        cells = [[v - INFERENCE_PAD for v in box.tolist()] for box in result['boxes']]
        boxes = expand_to_edges(cells_to_structure(cells, tol), image.width, image.height)
        records.append({'image': name, 'width': image.width, 'height': image.height, 'boxes': boxes})
        counts = ', '.join(f'{sum(b["label"] == n for b in boxes)} {n}(s)' for n in STRUCTURE_LABELS)
        print(f'{name}: {kind}, {counts}')
    return records
