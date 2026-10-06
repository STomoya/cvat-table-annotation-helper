import argparse
import html
import json
import threading
import traceback
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .cli import add_document, build_structure, export_coco, read_state, stage
from .cvat import CVAT_HOST, cvat_client

# ponytail: one job at a time and its status lives in memory, a queue and persisted errors if that gets in the way.
job: dict = {'label': '', 'error': '', 'thread': None}
job_lock = threading.Lock()


def running() -> bool:
    return job['thread'] is not None and job['thread'].is_alive()


def start_job(label: str, fn: Callable[[], object]) -> None:
    def work() -> None:
        try:
            fn()
        except (Exception, SystemExit) as e:
            traceback.print_exc()
            job['error'] = f'{label}: {e}'

    with job_lock:
        if running():
            return
        job.update(label=label, error='', thread=threading.Thread(target=work, daemon=True))
        job['thread'].start()


def documents(out: Path) -> dict[str, dict]:
    docs = {}
    for state_path in sorted(out.glob('*/state.json')):
        doc_dir = state_path.parent
        metadata_path = doc_dir / 'metadata.json'
        url = json.loads(metadata_path.read_text()).get('url', '') if metadata_path.exists() else ''
        docs[doc_dir.name] = {'dir': doc_dir, 'url': url, 'state': read_state(doc_dir)}
    return docs


def next_step(doc_dir: Path) -> None:
    state = read_state(doc_dir)
    with cvat_client() as client:
        if 'structure_task_id' in state:
            export_coco(client.tasks.retrieve(state['structure_task_id']), doc_dir)
        else:
            build_structure(client, client.tasks.retrieve(state['table_task_id']), doc_dir)


def task_link(state: dict, key: str, text: str) -> str:
    if key not in state:
        return ''
    return f'<a href="{html.escape(CVAT_HOST)}/tasks/{state[key]}" target="_blank">{text} #{state[key]}</a>'


def render(out: Path) -> str:
    busy = running()
    disabled = ' disabled' if busy else ''
    rows = []
    for name, doc in documents(out).items():
        state = doc['state']
        if 'structure_task_id' not in state:
            action = 'Build structure task'
        elif state.get('exported'):
            action = 'Re-export COCO'
        else:
            action = 'Export COCO'
        url = html.escape(doc['url'])
        links = ' '.join(
            filter(
                None, [task_link(state, 'table_task_id', 'tables'), task_link(state, 'structure_task_id', 'structure')]
            )
        )
        rows.append(
            f'<tr><td>{html.escape(name)}<br><small><a href="{url}">{url}</a></small></td>'
            f'<td>{stage(state)}</td><td>{links}</td>'
            f'<td><form method="post" action="/step"><input type="hidden" name="doc" value="{html.escape(name)}">'
            f'<button{disabled}>{action}</button></form></td></tr>'
        )
    banner = ''
    if busy:
        banner = f'<p class="busy">Running: {html.escape(job["label"])}</p>'
    elif job['error']:
        banner = f'<p class="error">Failed: {html.escape(job["error"])}</p>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
{'<meta http-equiv="refresh" content="3">' if busy else ''}
<title>annotate-tables</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #8884; padding: .5rem; text-align: left; vertical-align: top; }}
small {{ word-break: break-all; }}
input[type=url] {{ width: 70%; }}
.busy {{ color: #06c; }} .error {{ color: #c00; white-space: pre-wrap; }}
</style></head><body>
<h1>annotate-tables</h1>
{banner}
<form method="post" action="/add">
<input type="url" name="url" placeholder="https://example.com/document.pdf" required{disabled}>
<button{disabled}>Add PDF</button>
</form>
<table><tr><th>Document</th><th>Stage</th><th>CVAT</th><th>Next</th></tr>
{''.join(rows)}
</table></body></html>"""


def make_handler(out: Path, allowed_hosts: set[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def host_allowed(self) -> bool:
            # There is no login, so only answer to names we expect: this blocks DNS rebinding from a hostile site.
            if urlparse(f'//{self.headers.get("Host", "")}').hostname in allowed_hosts:
                return True
            self.send_error(HTTPStatus.FORBIDDEN, 'unexpected Host header, see --allow-host')
            return False

        def do_GET(self) -> None:
            if not self.host_allowed():
                return
            if self.path != '/':
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = render(out).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if not self.host_allowed():
                return
            # Browsers always send Origin on form posts, so a missing or foreign one is another site's form.
            if urlparse(self.headers.get('Origin', '')).netloc != self.headers.get('Host'):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            form = parse_qs(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode())
            docs = documents(out)
            url = form.get('url', [''])[0].strip()
            doc = docs.get(form.get('doc', [''])[0])
            if self.path == '/add' and url:
                if any(d['url'] == url for d in docs.values()):
                    job['error'] = f'already added: {url}'
                else:
                    start_job(f'adding {url}', lambda: add_document(url, out))
            elif self.path == '/step' and doc:
                start_job(f'next step for {doc["dir"].name}', lambda: next_step(doc['dir']))
            else:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header('Location', '/')
            self.end_headers()

    return Handler


def web_main() -> None:
    parser = argparse.ArgumentParser(
        description='Web UI for the annotation pipeline (CVAT_HOST, CVAT_USER, CVAT_PASSWORD).'
    )
    parser.add_argument('--host', default='127.0.0.1', help="address to listen on, e.g. this machine's Tailscale IP")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--out', type=Path, default=Path('data'))
    parser.add_argument(
        '--allow-host',
        action='append',
        default=[],
        help='extra name the UI may be opened under, e.g. a MagicDNS name (repeatable)',
    )
    args = parser.parse_args()

    allowed_hosts = {args.host, 'localhost', '127.0.0.1', *args.allow_host}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.out, allowed_hosts))
    print(f'http://{args.host}:{args.port}')
    server.serve_forever()
