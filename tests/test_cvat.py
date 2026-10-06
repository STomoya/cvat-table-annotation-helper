import tempfile
from pathlib import Path

from PIL import Image

from annotate_tables.cli import cells_to_structure, crop_tables, to_cvat_xml


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
