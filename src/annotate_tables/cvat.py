import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from pathlib import Path

from cvat_sdk import make_client
from PIL import Image

CVAT_HOST = os.environ.get('CVAT_HOST', 'http://100.79.94.13:8080')


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


def write_cvat_xml(records: list[dict], path: Path) -> None:
    ET.ElementTree(to_cvat_xml(records)).write(path, encoding='utf-8', xml_declaration=True)


def cvat_client():
    try:
        credentials = (os.environ['CVAT_USER'], os.environ['CVAT_PASSWORD'])
    except KeyError as e:
        sys.exit(f'CVAT access needs {e.args[0]} in the environment')
    return make_client(CVAT_HOST, credentials=credentials)


# Okabe-Ito hues: distinct from each other under colour-blindness and dark enough to read on white pages.
LABEL_COLORS = {'column': '#0072B2', 'row': '#D55E00', 'span': '#009E73'}


def create_task(client, name: str, labels: Sequence[str], images: list[Path], annotation_path: Path) -> int:
    task = client.tasks.create_from_data(
        spec={
            'name': name,
            'labels': [{'name': n, 'color': LABEL_COLORS[n]} if n in LABEL_COLORS else {'name': n} for n in labels],
        },
        resources=images,
        annotation_path=str(annotation_path),
        annotation_format='CVAT 1.1',
    )
    print(f'CVAT task {task.id}: {CVAT_HOST}/tasks/{task.id}')
    return task.id


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
