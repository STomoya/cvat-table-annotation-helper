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


def document_name(data: bytes, name: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(name).stem) or 'document'
    return f'{stem}-{hashlib.sha256(data).hexdigest()[:8]}'


def store(data: bytes, name: str, out_root: Path, **metadata: str | None) -> Path:
    """Write a PDF and the metadata on where it came from into its own directory under out_root."""
    if not data.startswith(b'%PDF-'):
        sys.exit(f'not a PDF: {name}')
    doc_dir = out_root / document_name(data, name)
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / 'source.pdf').write_bytes(data)
    record = {
        **metadata,
        'downloaded_at': datetime.now(UTC).isoformat(),
        'size': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }
    (doc_dir / 'metadata.json').write_text(json.dumps(record, indent=2) + '\n')
    return doc_dir


def fetch(source: str, out_root: Path) -> Path:
    """Store the PDF at a local path or an http(s) URL."""
    path = Path(source)
    if path.is_file():
        uri = path.resolve().as_uri()
        return store(path.read_bytes(), path.name, out_root, url=uri, final_url=uri)
    if urlparse(source).scheme not in ('http', 'https'):
        sys.exit(f'not a file or an http(s) URL: {source}')
    request = urllib.request.Request(source, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        final_url = response.url
        headers = response.headers
    return store(
        data,
        urlparse(final_url).path,
        out_root,
        url=source,
        final_url=final_url,
        content_type=headers.get('Content-Type'),
        etag=headers.get('ETag'),
        last_modified=headers.get('Last-Modified'),
    )


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
