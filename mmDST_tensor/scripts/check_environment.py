#!/usr/bin/env python3
"""Subprocess isolation helps identify a native-library crash without hiding it."""
import subprocess,sys
checks={
 "numpy": "import numpy; print(numpy.__version__)",
 "torch": "import torch; print(torch.__version__); print('CUDA available:',torch.cuda.is_available())",
 "cupy": "import cupy; print(cupy.__version__); print('CUDA devices:',cupy.cuda.runtime.getDeviceCount())",
 "torch_then_cupy": "import torch,cupy; print('Both imports OK')",
 "cupy_then_torch": "import cupy,torch; print('Both imports OK')",
}
for label,code in checks.items():
 print('\n==',label,'==',flush=True)
 p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
 print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr)
 print('exit:',p.returncode, '(segfault)' if p.returncode in (-11,139) else '')
