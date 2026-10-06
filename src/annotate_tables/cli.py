import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .cvat import create_task, crop_tables, cvat_client, export_annotations, write_cvat_xml
from .detect import STRUCTURE_LABELS, detect_structure, detect_tables
from .pdf import download, render_pages


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open('w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


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
    write_jsonl(records, doc_dir / 'detections.jsonl')
    write_cvat_xml(records, doc_dir / 'annotations.xml')
    print(f'{sum(len(r["boxes"]) for r in records)} table(s) on {len(pages)} page(s) -> {doc_dir}')
    if args.push:
        with cvat_client() as client:
            create_task(client, doc_dir.name, ['table'], pages, doc_dir / 'annotations.xml')


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
        write_jsonl(records, doc_dir / 'tables.jsonl')
        print(f'{len(records)} table(s) -> {doc_dir / "tables"}')
        if not records:
            sys.exit('no table boxes in the task, structure task not created')
        structure = detect_structure(doc_dir / 'tables', [r['image'] for r in records], args.threshold, args.tol)
        write_cvat_xml(structure, doc_dir / 'structure.xml')
        create_task(
            client,
            f'{task.name}-structure',
            STRUCTURE_LABELS,
            [doc_dir / 'tables' / r['image'] for r in records],
            doc_dir / 'structure.xml',
        )


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
