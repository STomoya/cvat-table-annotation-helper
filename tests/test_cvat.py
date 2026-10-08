import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from annotate_tables.cli import read_state, stage, update_state
from annotate_tables.cvat import crop_tables, to_cvat_xml
from annotate_tables.pdf import fetch
from annotate_tables.web import uploaded_file
from annotate_tables.detect import align_columns_to_rules, align_rows_to_text, cells_to_structure, expand_to_edges


def test_to_cvat_xml():
    records = [
        {
            'image': 'doc-1.png',
            'width': 100,
            'height': 200,
            'boxes': [{'label': 'table', 'box': [-3.0, 10.5, 120.0, 150.25], 'score': 0.9}],
        },
        {'image': 'doc-2.png', 'width': 100, 'height': 200, 'boxes': []},
    ]
    root = to_cvat_xml(records)
    images = root.findall('image')
    assert [i.get('name') for i in images] == ['doc-1.png', 'doc-2.png']
    assert (images[0].get('width'), images[0].get('height')) == ('100', '200')
    box = images[0].find('box')
    assert box is not None
    assert box.get('label') == 'table'
    assert [box.get(k) for k in ('xtl', 'ytl', 'xbr', 'ybr')] == [
        '0.00',
        '10.50',
        '100.00',
        '150.25',
    ]
    assert images[1].find('box') is None


def test_crop_tables():
    records = [
        {
            'image': 'doc-1.png',
            'width': 100,
            'height': 200,
            'boxes': [
                {'label': 'table', 'box': [20.0, 120.0, 60.0, 160.0]},
                {'label': 'table', 'box': [5.0, 10.0, 95.5, 50.0]},
            ],
        },
        {'image': 'doc-2.png', 'width': 100, 'height': 200, 'boxes': []},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        pages, tables = Path(tmp) / 'pages', Path(tmp) / 'tables'
        pages.mkdir()
        Image.new('RGB', (100, 200)).save(pages / 'doc-1.png')
        crops = crop_tables(to_cvat_xml(records), pages, tables, pad=10)
        assert crops == [
            {
                'image': 'doc-1-t01.png',
                'page': 'doc-1.png',
                'crop_box': [0, 0, 100, 60],
            },
            {
                'image': 'doc-1-t02.png',
                'page': 'doc-1.png',
                'crop_box': [10, 110, 70, 170],
            },
        ]
        assert Image.open(tables / 'doc-1-t02.png').size == (60, 60)


def test_cells_to_structure():
    # 3 rows x 2 columns with jittered edges; the right column's lower two rows are one merged cell.
    cells = [
        [0, 0, 50, 20],
        [51, 1, 100, 20],
        [0, 20, 49, 41],
        [50, 21, 100, 60],
        [1, 40, 50, 60],
    ]
    boxes = cells_to_structure(cells, tol=3)
    by_label = {label: [b['box'] for b in boxes if b['label'] == label] for label in ('row', 'column', 'span')}
    assert by_label['row'] == [[0.33, 0.5, 100.0, 20.25], [0.33, 20.25, 100.0, 40.5], [0.33, 40.5, 100.0, 60.0]]
    assert by_label['column'] == [[0.33, 0.5, 50.0, 60.0], [50.0, 0.5, 100.0, 60.0]]
    assert by_label['span'] == [[50.0, 20.25, 100.0, 60.0]]
    assert cells_to_structure([], tol=3) == []


if __name__ == '__main__':
    test_cells_to_structure()
    test_to_cvat_xml()
    test_crop_tables()
    print('ok')


def test_expand_to_edges():
    boxes = [
        {'label': 'row', 'box': [5, 10, 90, 20]},
        {'label': 'row', 'box': [5, 20, 90, 40]},
        {'label': 'column', 'box': [5, 10, 30, 40]},
        {'label': 'column', 'box': [30, 10, 90, 40]},
        {'label': 'span', 'box': [5, 20, 90, 40]},
    ]
    assert [b['box'] for b in expand_to_edges(boxes, 100, 50)] == [
        [0, 0, 100, 20],
        [0, 20, 100, 50],
        [0, 0, 30, 50],
        [30, 0, 100, 50],
        [0, 20, 100, 50],
    ]


def test_align_rows_to_text():
    image = Image.new('RGB', (100, 60), 'white')
    draw = ImageDraw.Draw(image)
    # Text running down the first column, and two lines of text in the other four: ink on pixel rows 5-14 and 35-44.
    draw.rectangle([5, 5, 15, 44], fill='black')
    for left in range(25, 100, 20):
        draw.rectangle([left, 5, left + 10, 14], fill='black')
        draw.rectangle([left, 35, left + 10, 44], fill='black')
    boxes = [
        {'label': 'row', 'box': [0, 0, 100, 18]},
        {'label': 'row', 'box': [0, 18, 100, 60]},
        {'label': 'span', 'box': [0, 18, 40, 60]},
        *({'label': 'column', 'box': [left, 0, left + 20, 60]} for left in range(0, 100, 20)),
    ]
    aligned = [b['box'] for b in align_rows_to_text(boxes, image, tol=8)]
    assert aligned[:3] == [[0, 0, 100, 25], [0, 25, 100, 60], [0, 25, 40, 60]]
    assert aligned[3:] == [b['box'] for b in boxes[3:]]

    draw.line([0, 22, 99, 22], fill='black')
    assert align_rows_to_text(boxes, image, tol=8)[0]['box'] == [0, 0, 100, 22.5]


def test_align_columns_to_rules():
    image = Image.new('RGB', (100, 60), 'white')
    ImageDraw.Draw(image).line([33, 0, 33, 59], fill='black')
    boxes = [
        {'label': 'column', 'box': [0, 0, 30, 60]},
        {'label': 'column', 'box': [30, 0, 70, 60]},
        {'label': 'column', 'box': [70, 0, 100, 60]},
        {'label': 'span', 'box': [30, 0, 100, 20]},
        {'label': 'row', 'box': [0, 0, 100, 60]},
    ]
    aligned = [b['box'] for b in align_columns_to_rules(boxes, image, tol=8)]
    # The boundary at 70 has no ruling line and stays.
    assert aligned == [[0, 0, 33.5, 60], [33.5, 0, 70, 60], [70, 0, 100, 60], [33.5, 0, 100, 20], [0, 0, 100, 60]]


def test_fetch_local_file():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / 'my report.pdf'
        source.write_bytes(b'%PDF-1.7 test')
        doc_dir = fetch(str(source), Path(tmp) / 'data')
        assert doc_dir.name.startswith('my_report-')
        assert (doc_dir / 'source.pdf').read_bytes() == b'%PDF-1.7 test'
        assert json.loads((doc_dir / 'metadata.json').read_text())['url'] == source.resolve().as_uri()


def test_uploaded_file():
    data = b'%PDF-1.7\r\n\xff\x00\r\rbinary\n--not-the-boundary\n'
    body = (
        b'--XYZ\r\nContent-Disposition: form-data; name="file"; filename="my report.pdf"\r\n'
        b'Content-Type: application/pdf\r\n\r\n' + data + b'\r\n--XYZ--\r\n'
    )
    assert uploaded_file('multipart/form-data; boundary=XYZ', body) == ('my report.pdf', data, '')
    field = b'--XYZ\r\nContent-Disposition: form-data; name="url"\r\n\r\nhttps://example.com/a.pdf\r\n'
    with_url = uploaded_file('multipart/form-data; boundary=XYZ', field + body)
    assert with_url == ('my report.pdf', data, 'https://example.com/a.pdf')
    assert uploaded_file('application/x-www-form-urlencoded', b'url=x') is None


def test_state_round_trip_and_stage():
    with tempfile.TemporaryDirectory() as tmp:
        doc_dir = Path(tmp)
        assert read_state(doc_dir) == {}
        assert stage(read_state(doc_dir)) == 'detected'
        update_state(doc_dir, table_task_id=3)
        assert stage(read_state(doc_dir)) == 'correct tables in CVAT'
        update_state(doc_dir, structure_task_id=4)
        assert read_state(doc_dir) == {'table_task_id': 3, 'structure_task_id': 4}
        assert stage(read_state(doc_dir)) == 'correct structure in CVAT'
        update_state(doc_dir, exported=True)
        assert stage(read_state(doc_dir)) == 'exported'
