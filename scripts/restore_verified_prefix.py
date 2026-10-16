#!/usr/bin/env python3
"""One-off offline recovery of hash-verified files from an interrupted upload.

No network, model execution, Git push, or unverified file reconstruction.
The original source_manifest.json is preserved unchanged.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import struct
import tarfile
import zipfile

PREFIX_DELTA_SHA256 = '08330ffe2c6d27cd3671da941cb66992a3027e8c0bd8d87413fd288c05f5dca2'
BASES = [('public-textbook.html', '0c501f0ab7a8ac485b0725996c73f63c7e08ae05323399f9a63d1d3e6ae5b718'),
         ('blank-notebook.html', '213bfe998e47245ca564c46a999da1232e9077d0f62fa0bc401e36ed3cebcb40')]

def recover(objects: Path, bases: Path) -> dict[str, bytes]:
    chunks = []
    for i in range(14):
        paths = list(objects.glob(f'{i:02d}-*.b64'))
        if len(paths) != 1:
            raise ValueError(f'Expected one object for part {i}')
        path = paths[0]; data = path.read_bytes()
        sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != 12000 or sha != path.stem.split('-', 1)[1]:
            raise ValueError(f'Object integrity failure: {path.name}')
        chunks.append(data)
    # Only the known-good prefix is used. The two damaged tail parts are excluded.
    delta = lzma.LZMADecompressor(memlimit=256*1024*1024).decompress(base64.b64decode(b''.join(chunks), validate=True))
    if hashlib.sha256(delta).hexdigest() != PREFIX_DELTA_SHA256:
        raise ValueError('Known prefix delta digest mismatch')
    dictionary = bytearray()
    for name, expected in BASES:
        data = (bases/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Dictionary mismatch: {name}')
        dictionary.extend(data)
    if delta[:4] != b'ALD1':
        raise ValueError('Invalid delta header')
    expected_size = struct.unpack_from('<Q', delta, 4)[0]
    if expected_size > 10*1024*1024:
        raise ValueError('Oversized archive')
    raw = bytearray(); pos = 12
    while pos < len(delta):
        op = delta[pos:pos+1]; pos += 1
        if op == b'C':
            if pos+8 > len(delta): break
            offset, length = struct.unpack_from('<II', delta, pos); pos += 8
            if offset+length > len(dictionary): raise ValueError('Copy outside dictionary')
            raw.extend(dictionary[offset:offset+length])
        elif op == b'L':
            if pos+4 > len(delta): break
            length = struct.unpack_from('<I', delta, pos)[0]; pos += 4
            if pos+length > len(delta): break
            raw.extend(delta[pos:pos+length]); pos += length
        else:
            raise ValueError('Invalid delta opcode')
        if len(raw) > expected_size: raise ValueError('Oversized delta output')
    parsed = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        while True:
            try:
                member = archive.next()
                if member is None: break
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or '..' in name.parts or '.git' in name.parts or member.size > 2*1024*1024:
                    raise ValueError('Unsafe archive member')
                if name.parts[0] == '.github' and str(name) != '.github/PULL_REQUEST_TEMPLATE.md':
                    raise ValueError('Unexpected workflow in source payload')
                if str(name) in parsed: raise ValueError('Duplicate member')
                parsed[str(name)] = archive.extractfile(member).read()
            except tarfile.ReadError:
                break
    manifest_data = parsed['bootstrap_manifest.json']
    manifest = json.loads(manifest_data)
    expected = {row['path']: row for row in manifest['files']}
    attrs = ['compress_type','create_system','create_version','extract_version','reserved','flag_bits','volume','internal_attr','external_attr']
    for name in list(parsed):
        if not name.endswith('.restore-zip.json'): continue
        recipe = json.loads(parsed[name]); target = name[:-len('.restore-zip.json')]
        if target not in expected: raise ValueError('Unknown zip target')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as output:
            output.comment = base64.b64decode(recipe['comment'])
            for row in recipe['entries']:
                info = zipfile.ZipInfo(row['name'], tuple(row['date_time']))
                for attr in attrs: setattr(info, attr, row[attr])
                info.comment = base64.b64decode(row['comment']); info.extra = base64.b64decode(row['extra'])
                data = row['text'].encode('utf-8') if 'text' in row else base64.b64decode(row['base64'])
                output.writestr(info, data, compress_type=info.compress_type, compresslevel=recipe['level'])
        parsed[target] = buffer.getvalue()
    verified = {'bootstrap_manifest.json': manifest_data}
    for name, row in expected.items():
        data = parsed.get(name)
        if data is not None and len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']:
            verified[name] = data
    if len(verified) != 44 or 'docs/reference/handbook_with_exercises_updated(4).html' not in verified:
        raise ValueError(f'Unexpected verified file set: {len(verified)}')
    return verified

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--objects', type=Path, required=True)
    ap.add_argument('--bases', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); files = recover(a.objects, a.bases)
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root): raise ValueError('Unsafe output path')
        if path.exists() and path.read_bytes() != data: raise ValueError(f'Existing output differs: {name}')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    print(f'Recovered {len(files)} exact, digest-verified files. Incomplete tail files were not written.')
if __name__ == '__main__': main()
