import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlparse

from cvat_sdk import make_client
from PIL import Image, ImageOps

MODEL = 'PaddlePaddle/PP-DocLayoutV3_safetensors'
USER_AGENT = 'Mozilla/5.0 (annotate-tables)'
CVAT_HOST = os.environ.get('CVAT_HOST', 'http://100.79.94.13:8080')
STRUCTURE_LABELS = ('column', 'row', 'span')
TABLE_CLASSIFIER = 'PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors'
CELL_MODELS = {
    'wired_table': 'PaddlePaddle/RT-DETR-L_wired_table_cell_det_safetensors',
    'wireless_table': 'PaddlePaddle/RT-DETR-L_wireless_table_cell_det_safetensors',
}
# White border added around a crop for cell detection only; 10 beat 0 and 20 on hand-annotated tables.
INFERENCE_PAD = 10


def download(url: str, out_root: Path) -> Path:
    if urlparse(url).scheme not in ('http', 'https'):
        sys.exit(f'not an http(s) URL: {url}')
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        final_url = response.url
        headers = response.headers
    if not data.startswith(b'%PDF-'):
        sys.exit(f'not a PDF (content-type {headers.get("Content-Type")}): {url}')

    sha256 = hashlib.sha256(data).hexdigest()
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(urlparse(final_url).path).stem) or 'document'
    doc_dir = out_root / f'{stem}-{sha256[:8]}'
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / 'source.pdf').write_bytes(data)
    metadata = {
        'url': url,
        'final_url': final_url,
        'downloaded_at': datetime.now(UTC).isoformat(),
        'content_type': headers.get('Content-Type'),
        'etag': headers.get('ETag'),
        'last_modified': headers.get('Last-Modified'),
        'size': len(data),
        'sha256': sha256,
    }
    (doc_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return doc_dir


def render_pages(doc_dir: Path, dpi: int) -> list[Path]:
    pages_dir = doc_dir / 'pages'
    pages_dir.mkdir(exist_ok=True)
    # The document name is the file prefix so images from several documents can share one CVAT task.
    subprocess.run(
        [
            'pdftoppm',
            '-r',
            str(dpi),
            '-png',
            str(doc_dir / 'source.pdf'),
            str(pages_dir / doc_dir.name),
        ],
        check=True,
    )
    return sorted(pages_dir.glob('*.png'))


def run_detector(processor, model, image: Image.Image, threshold: float) -> dict:
    import torch  # noqa: PLC0415 -- slow. Use `lazy` on 3.15

    inputs = processor(images=image, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]], threshold=threshold)[0]


def detect_tables(pages: list[Path], threshold: float) -> list[dict]:
    import torch  # noqa: PLC0415 -- slow. Use `lazy` on 3.15
    from transformers import (  # noqa: PLC0415 -- imports torch internally.
        AutoImageProcessor,
        AutoModelForObjectDetection,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForObjectDetection.from_pretrained(MODEL).to(device).eval()

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


def detect_structure(tables_dir: Path, names: list[str], threshold: float, tol: float) -> list[dict]:
    import torch  # noqa: PLC0415 -- slow. Use `lazy` on 3.15
    from transformers import (  # noqa: PLC0415 -- imports torch internally.
        AutoImageProcessor,
        AutoModelForImageClassification,
        AutoModelForObjectDetection,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier_processor = AutoImageProcessor.from_pretrained(TABLE_CLASSIFIER)
    # The checkpoint is stored in half precision, which fails on float inputs.
    classifier = AutoModelForImageClassification.from_pretrained(TABLE_CLASSIFIER, dtype=torch.float32)
    classifier = classifier.to(device).eval()
    detectors = {
        kind: (
            AutoImageProcessor.from_pretrained(repo),
            AutoModelForObjectDetection.from_pretrained(repo).to(device).eval(),
        )
        for kind, repo in CELL_MODELS.items()
    }

    records = []
    for name in names:
        image = Image.open(tables_dir / name).convert('RGB')
        with torch.no_grad():
            # PPLCNetForImageClassification returns its logits as the first output, not as `.logits`.
            logits = classifier(**classifier_processor(images=image, return_tensors='pt').to(device))[0]
        kind = classifier.config.id2label[logits.argmax().item()]
        result = run_detector(*detectors[kind], ImageOps.expand(image, INFERENCE_PAD, fill='white'), threshold)
        cells = [[v - INFERENCE_PAD for v in box.tolist()] for box in result['boxes']]
        boxes = cells_to_structure(cells, tol)
        records.append({'image': name, 'width': image.width, 'height': image.height, 'boxes': boxes})
        counts = ', '.join(f'{sum(b["label"] == n for b in boxes)} {n}(s)' for n in STRUCTURE_LABELS)
        print(f'{name}: {kind}, {counts}')
    return records


def to_cvat_xml(records: list[dict]) -> ET.Element:
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'
    for index, record in enumerate(records):
        width, height = record['width'], record['height']
        image = ET.SubElement(
            root,
            'image',
            id=str(index),
            name=record['image'],
            width=str(width),
            height=str(height),
        )
        for box in record['boxes']:
            x1, y1, x2, y2 = box['box']
            ET.SubElement(
                image,
                'box',
                label=box['label'],
                source='auto',
                occluded='0',
                xtl=f'{max(x1, 0):.2f}',
                ytl=f'{max(y1, 0):.2f}',
                xbr=f'{min(x2, width):.2f}',
                ybr=f'{min(y2, height):.2f}',
                z_order='0',
            )
    ET.indent(root)
    return root


def cvat_client():

    try:
        credentials = (os.environ['CVAT_USER'], os.environ['CVAT_PASSWORD'])
    except KeyError as e:
        sys.exit(f'CVAT access needs {e.args[0]} in the environment')
    return make_client(CVAT_HOST, credentials=credentials)


def push_to_cvat(doc_dir: Path, pages: list[Path]) -> None:
    with cvat_client() as client:
        task = client.tasks.create_from_data(
            spec={'name': doc_dir.name, 'labels': [{'name': 'table'}]},
            resources=pages,
            annotation_path=str(doc_dir / 'annotations.xml'),
            annotation_format='CVAT 1.1',
        )
    print(f'CVAT task {task.id}: {CVAT_HOST}/tasks/{task.id}')


def export_annotations(task, path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'export.zip'
        task.export_dataset('CVAT for images 1.1', archive, include_images=False)
        with zipfile.ZipFile(archive) as z:
            path.write_bytes(z.read('annotations.xml'))


def crop_tables(annotations: ET.Element, pages_dir: Path, tables_dir: Path, pad: int) -> list[dict]:

    tables_dir.mkdir(exist_ok=True)
    records = []
    for image in annotations.iter('image'):
        boxes = sorted(
            (
                [float(box.attrib[k]) for k in ('xtl', 'ytl', 'xbr', 'ybr')]
                for box in image.iter('box')
                if box.get('label') == 'table'
            ),
            key=lambda b: (b[1], b[0]),
        )
        if not boxes:
            continue
        page_name = Path(image.attrib['name']).name
        page = Image.open(pages_dir / page_name)
        for index, (x1, y1, x2, y2) in enumerate(boxes, 1):
            crop_box = (
                max(math.floor(x1) - pad, 0),
                max(math.floor(y1) - pad, 0),
                min(math.ceil(x2) + pad, page.width),
                min(math.ceil(y2) + pad, page.height),
            )
            name = f'{Path(page_name).stem}-t{index:02d}.png'
            page.crop(crop_box).save(tables_dir / name)
            # crop_box is the crop's position on the page, to map structure annotations back to page coordinates.
            records.append({'image': name, 'page': page_name, 'crop_box': list(crop_box)})
    return records


def structure_main() -> None:
    parser = argparse.ArgumentParser(
        description='Export a table task from CVAT, crop its tables and create a structure task (CVAT_HOST, CVAT_USER, CVAT_PASSWORD).'  # noqa: E501
    )
    parser.add_argument('task_id', type=int)
    parser.add_argument('--out', type=Path, default=Path('data'))
    parser.add_argument('--pad', type=int, default=0, help='pixels of page kept around each table')
    parser.add_argument('--threshold', type=float, default=0.5, help='cell detection score threshold')
    parser.add_argument('--tol', type=float, default=8, help='pixels within which cell edges share a grid line')
    args = parser.parse_args()

    with cvat_client() as client:
        task = client.tasks.retrieve(args.task_id)
        doc_dir = args.out / task.name
        if not (doc_dir / 'pages').is_dir():
            sys.exit(f'no page images for task {task.id} ({task.name}) at {doc_dir / "pages"}')
        export_annotations(task, doc_dir / 'tables.xml')
        records = crop_tables(
            ET.parse(doc_dir / 'tables.xml').getroot(),
            doc_dir / 'pages',
            doc_dir / 'tables',
            args.pad,
        )
        with (doc_dir / 'tables.jsonl').open('w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        print(f'{len(records)} table(s) -> {doc_dir / "tables"}')
        if not records:
            sys.exit('no table boxes in the task, structure task not created')
        structure = detect_structure(doc_dir / 'tables', [r['image'] for r in records], args.threshold, args.tol)
        ET.ElementTree(to_cvat_xml(structure)).write(doc_dir / 'structure.xml', encoding='utf-8', xml_declaration=True)
        structure_task = client.tasks.create_from_data(
            spec={
                'name': f'{task.name}-structure',
                'labels': [{'name': n} for n in STRUCTURE_LABELS],
            },
            resources=[doc_dir / 'tables' / r['image'] for r in records],
            annotation_path=str(doc_dir / 'structure.xml'),
            annotation_format='CVAT 1.1',
        )
    print(f'CVAT task {structure_task.id}: {CVAT_HOST}/tasks/{structure_task.id}')


def coco_main() -> None:
    parser = argparse.ArgumentParser(
        description='Export a structure task from CVAT as a COCO dataset (CVAT_HOST, CVAT_USER, CVAT_PASSWORD).'
    )
    parser.add_argument('task_id', type=int)
    parser.add_argument('--out', type=Path, default=Path('data'))
    args = parser.parse_args()

    with cvat_client() as client, tempfile.TemporaryDirectory() as tmp:
        task = client.tasks.retrieve(args.task_id)
        coco_dir = args.out / task.name.removesuffix('-structure') / 'coco'
        archive = Path(tmp) / 'export.zip'
        task.export_dataset('COCO 1.0', archive, include_images=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(coco_dir)
    print(f'COCO dataset for task {task.id} ({task.name}) -> {coco_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Download a PDF and pre-annotate its tables for CVAT.')
    parser.add_argument('url')
    parser.add_argument('--out', type=Path, default=Path('data'))
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument(
        '--push',
        action='store_true',
        help='create a CVAT task (CVAT_HOST, CVAT_USER, CVAT_PASSWORD)',
    )
    args = parser.parse_args()

    doc_dir = download(args.url, args.out)
    pages = render_pages(doc_dir, args.dpi)
    records = detect_tables(pages, args.threshold)
    with (doc_dir / 'detections.jsonl').open('w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    ET.ElementTree(to_cvat_xml(records)).write(doc_dir / 'annotations.xml', encoding='utf-8', xml_declaration=True)
    print(f'{sum(len(r["boxes"]) for r in records)} table(s) on {len(pages)} page(s) -> {doc_dir}')
    if args.push:
        push_to_cvat(doc_dir, pages)
