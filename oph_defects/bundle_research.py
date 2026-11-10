"""Freeze only the research JSON documents actually referenced by the viewer.

These are reference artifacts, not promoted derivations. The manifest retains
source-relative paths and hashes. Regeneration is explicit; serving the app
never imports or executes the research workspace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
VIEWER=HERE/'viewer2'
SOURCE=HERE.parents[1]/'repair-law'/'work'


def referenced_paths(viewer=VIEWER):
    paths=set()
    for file in viewer.iterdir():
        if file.suffix in ('.js','.html'):
            paths.update(re.findall(r'''["']/rl/([^"']+\.json)["']''',file.read_text()))
    for name in paths:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Research path escapes the selected root')
    return sorted(paths)


def bundle(source=SOURCE,viewer=VIEWER):
    source=Path(source).resolve();out=Path(viewer)/'research'
    records=[]
    for name in referenced_paths(viewer):
        path=(source/name).resolve()
        if not path.is_relative_to(source):raise ValueError('Research symlink escapes the selected root')
        raw=path.read_bytes();json.loads(raw)
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        records.append({'source':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    value={'schema':'oph.viewer.research_reference_bundle.v1',
           'scope':'Frozen reference artifacts. Their inclusion does not certify every claim or join them to the load run.',
           'source_root_role':'repair-law/work','files':records}
    (out/'manifest.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


def check(viewer=VIEWER):
    out=Path(viewer)/'research';value=json.loads((out/'manifest.json').read_text())
    if sorted(row['source'] for row in value['files'])!=referenced_paths(viewer):raise ValueError('Reference menu changed')
    for row in value['files']:
        raw=(out/row['source']).read_bytes();json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=row['sha256'] or len(raw)!=row['bytes']:raise ValueError('Reference content changed: '+row['source'])
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--check',action='store_true')
    a=p.parse_args();value=check() if a.check else bundle(a.source)
    print('Verified frozen viewer reference artifacts:',len(value['files']))


if __name__=='__main__':main()
