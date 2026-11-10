"""Serve the first defect viewer together with one old-format run directory.

    python -m oph_defects.serve_legacy runs/z6_L10_seed7 --port 8777

For the Z6 runs on the refined carrier (``oph_defects.run``) and the first
federation format (``oph_defects.federation_run``). The viewer is served at
``/`` and the run at ``/run/``. Universe runs use ``python -m oph_defects.serve``.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse

VIEWER = Path(__file__).resolve().parent / 'viewer'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, run_dir: Path, **kwargs):
        self.run_dir = run_dir
        super().__init__(*args, directory=str(VIEWER), **kwargs)

    def translate_path(self, path: str) -> str:
        clean = urllib.parse.urlparse(path).path
        if clean.startswith('/run/'):
            target = (self.run_dir / urllib.parse.unquote(clean[5:])).resolve()
            if not str(target).startswith(str(self.run_dir.resolve())):
                return str(self.run_dir / 'forbidden')
            return str(target)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--port', type=int, default=8777)
    a = p.parse_args(argv)
    if not (a.run / 'run.json').is_file():
        raise SystemExit(f'no run.json in {a.run}')
    server = ThreadingHTTPServer(('127.0.0.1', a.port), partial(Handler, run_dir=a.run))
    print(f'viewer: http://127.0.0.1:{a.port}/   run: {a.run}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
