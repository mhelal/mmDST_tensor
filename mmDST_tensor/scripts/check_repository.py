#!/usr/bin/env python3
from pathlib import Path
import ast,sys
root=Path(__file__).resolve().parents[1]
failed=[]
for p in root.rglob('*.py'):
 if '__pycache__' in p.parts:continue
 try:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 except Exception as e:failed.append((str(p.relative_to(root)),str(e)))
print('Python syntax issues:',len(failed))
for f,e in failed:print(f,e)
sys.exit(bool(failed))
