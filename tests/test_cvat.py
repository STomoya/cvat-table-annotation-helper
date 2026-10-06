import tempfile
from pathlib import Path

from PIL import Image

from annotate_tables.cli import crop_tables, to_cvat_xml


def test_to_cvat_xml():
    records = [
        {
            'image': 'doc-1.png',
            'width': 100,
            'height': 200,
            'tables': [{'box': [-3.0, 10.5, 120.0, 150.25], 'score': 0.9}],
        },
        {'image': 'doc-2.png', 'width': 100, 'height': 200, 'tables': []},
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
            'tables': [
                {'box': [20.0, 120.0, 60.0, 160.0]},
                {'box': [5.0, 10.0, 95.5, 50.0]},
            ],
        },
        {'image': 'doc-2.png', 'width': 100, 'height': 200, 'tables': []},
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


if __name__ == '__main__':
    test_to_cvat_xml()
    test_crop_tables()
    print('ok')
