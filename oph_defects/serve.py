"""Serve the universe viewer for one run directory or a directory of runs.

    python -m oph_defects.serve runs/universe --port 8777

The viewer is at ``/``; every run directory below the given path that holds a
``run.json`` of schema ``oph.universe.run.v2`` is listed at ``/api/runs`` and
served at ``/runs/<name>/``. The particle table is always the current
``oph_defects/particle_map.json`` (rebuild it with ``python -m
oph_defects.particle_table``). Old runs use ``python -m oph_defects.serve_legacy``.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import urllib.parse

HERE = Path(__file__).resolve().parent
VIEWER = HERE / 'viewer2'


def find_runs(root: Path) -> dict[str, Path]:
    runs = {}
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for p in candidates:
        f = p / 'run.json'
        if f.is_file():
            try:
                if json.loads(f.read_text()).get('schema') == 'oph.universe.run.v2':
                    runs[p.name] = p
            except Exception:
                pass
    return runs


def summary(name: str, path: Path) -> dict:
    m = json.loads((path / 'run.json').read_text())
    g = m['geometry']
    return {'name': name, 'level': g['level'], 'carriers': g['carriers'], 'ports': g['ports'], 'seams': g['seams'],
            'sweeps': m['sweeps'], 'cuts': len(m['cuts']), 'identities': m['identities'], 'preparation': m['preparation']['name'],
            'backend': m['config']['backend'], 'observers': len(m['observers']), 'register': m['config'].get('register', True),
            'twists': bool(m['config'].get('twists')), 'sample': int(m.get('display_sample', 1))}


def closed_loop_tables(root: Path | None) -> list:
    out = []
    if root is None or not root.is_dir():
        return out
    for f in sorted(root.glob('*/summary.json')):
        try:
            d = json.loads(f.read_text())
            if 'rows' in d:
                out.append({'name': f.parent.name, 'law': d.get('law'), 'rows': d['rows']})
        except Exception:
            pass
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, runs: dict, closed: Path | None = None, **kwargs):
        self.runs = runs
        self.closed = closed
        super().__init__(*args, directory=str(VIEWER), **kwargs)

    def translate_path(self, path: str) -> str:
        clean = urllib.parse.unquote(urllib.parse.urlparse(path).path)
        if clean.startswith('/runs/'):
            name, _, rest = clean[6:].partition('/')
            base = self.runs.get(name)
            if base is None:
                return str(VIEWER / 'missing')
            target = (base / rest).resolve()
            return str(target) if target.is_relative_to(base.resolve()) else str(VIEWER / 'missing')
        if clean.startswith('/lab/L') and clean.endswith('.npz'):
            try:
                level = int(clean[6:-4])
            except ValueError:
                return str(VIEWER / 'missing')
            if not 1 <= level <= 5:
                return str(VIEWER / 'missing')
            f = Path.home() / '.cache' / 'oph_universe' / f'lab_L{level}.npz'
            if not f.is_file():
                from . import observer_lab
                observer_lab.export(level, f)
            return str(f)
        if clean.startswith('/rl/'):                       # explicitly frozen, read-only viewer reference bundle
            rl = (VIEWER / 'research').resolve()
            target = (rl / clean[4:]).resolve()
            ok = target.is_relative_to(rl) and target.suffix in ('.json', '.npz') and target.is_file()
            return str(target) if ok else str(VIEWER / 'missing')
        if clean == '/findings.json':
            return str(HERE / 'findings.json')
        if clean == '/particle_map.json':
            return str(HERE / 'particle_map.json')
        if clean == '/DATASET.md':
            return str(HERE / 'DATASET.md')
        return super().translate_path(path)

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        if route in ('/api/runs', '/api/closed_loop'):
            if route == '/api/runs':
                body = json.dumps([summary(n, p) for n, p in sorted(self.runs.items(), key=lambda kv: summary(*kv)['carriers'])]).encode()
            else:
                body = json.dumps(closed_loop_tables(self.closed)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', type=Path, help='a run directory or a directory of runs')
    p.add_argument('--port', type=int, default=8777)
    p.add_argument('--closed-loop', type=Path, default=None, help='directory of closed-loop matrices (default: closed_loop next to the runs)')
    a = p.parse_args(argv)
    closed = a.closed_loop if a.closed_loop is not None else a.path.resolve().parent / 'closed_loop'
    runs = find_runs(a.path)
    if not runs:
        raise SystemExit(f'no universe run below {a.path}')
    from . import dataset
    for name, path in runs.items():
        if not (path / 'view' / 'view.json').is_file():
            print(f'building viewer files for {name}', flush=True)
            dataset.build_view(path)
    server = ThreadingHTTPServer(('127.0.0.1', a.port), partial(Handler, runs=runs, closed=closed))
    print(f'viewer: http://127.0.0.1:{a.port}/   runs: {", ".join(runs)}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
