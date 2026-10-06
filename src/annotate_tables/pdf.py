import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

USER_AGENT = 'Mozilla/5.0 (annotate-tables)'


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
