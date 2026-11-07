"""Check the static demo's release integrity before publishing it."""

from hashlib import sha256
from pathlib import Path
import json
import re


def main():
    root = Path(__file__).resolve().parents[1] / 'demo'
    manifest = json.loads((root / 'snapshot-manifest.json').read_text())
    files = manifest['files']
    required = {'index.html', 'images/geodi-lab-logo.png', 'data/demo.json',
                'data/graph.json', 'data/metro.json', 'data/metro-flow-days.json'}
    assert required <= files.keys(), 'The release is missing required entry files.'
    for name, record in files.items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve()), f'Unsafe path: {name}'
        payload = path.read_bytes()
        assert len(payload) == record['bytes'], f'Unexpected size: {name}'
        assert sha256(payload).hexdigest() == record['sha256'], f'Checksum mismatch: {name}'
        if path.suffix in {'.html', '.js', '.css'}:
            assert b'/u-discovery-geodi/' not in payload, f'Old deployment prefix: {name}'
        if path.suffix in {'.json', '.geojson'}:
            json.loads(payload)
            for ref in re.findall(r'"(/data/[^"\s]+)"', payload.decode()):
                assert (root / ref.lstrip('/')).is_file(), f'Missing data reference: {ref}'
    total = sum(record['bytes'] for record in files.values())
    print(f'Validated {len(files)} captured files ({total:,} bytes), including all data checksums.')


if __name__ == '__main__':
    main()
