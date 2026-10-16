#!/usr/bin/env python3
"""Check current-document links, HTML anchors, Python syntax and source hashes.

Historical documents are preserved, so their old external/sandbox links are not
silently repaired. This is a bootstrap integrity check, not scientific validation.
"""
from __future__ import annotations
import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote
from check_sources import verify
ROOT=Path(__file__).resolve().parents[1]

class Anchors(HTMLParser):
    def __init__(self):super().__init__();self.ids=set()
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key=='id':self.ids.add(value)

def links(text: str):
    text=re.sub(r'```.*?```','',text,flags=re.S)
    i=0
    while True:
        i=text.find('](',i)
        if i<0:return
        start=i+2;j=start;depth=1
        while j<len(text) and depth:
            if text[j]=='(' and (j==0 or text[j-1]!='\\'):depth+=1
            elif text[j]==')' and (j==0 or text[j-1]!='\\'):depth-=1
            j+=1
        if depth==0:yield text[start:j-1].strip().strip('<>')
        i=j

def main():
    errors=verify(ROOT);checked=0;cache={}
    for p in ROOT.rglob('*.md'):
        if '.git' in p.parts or 'archive' in p.parts:continue
        for target in links(p.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
            name,_,anchor=target.partition('#')
            path=(p.parent/unquote(name)).resolve()
            if not path.is_relative_to(ROOT) or not path.exists():
                errors.append(f'Broken link: {p.relative_to(ROOT)} -> {target}');continue
            checked+=1
            if anchor and path.suffix=='.html':
                if path not in cache:
                    parser=Anchors();parser.feed(path.read_text(encoding='utf-8'));cache[path]=parser.ids
                if unquote(anchor) not in cache[path]:errors.append(f'Missing anchor: {target}')
    for p in ROOT.rglob('*.py'):
        if any(x in p.parts for x in ('.git','__pycache__','.venv')):continue
        try:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        except SyntaxError as e:errors.append(str(e))
    print(f'Checked {checked} current local document links, archived source hashes and Python syntax.')
    if errors:print('\n'.join(errors));return 1
    print('PASS — not a model or GPU validation.');return 0
if __name__=='__main__':raise SystemExit(main())
