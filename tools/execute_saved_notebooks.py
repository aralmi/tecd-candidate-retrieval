"""Execute only the saved-result presentation; this command cannot select training."""
from pathlib import Path
import json
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=[]
    for path in sorted((ROOT/'research').glob('*/analysis.ipynb')):
        nb=nbformat.read(path,as_version=4)
        if nb.metadata.get('recsys',{}).get('exclude_from_saved_batch'):
            print('SKIP_TRAINING',path.parent.name,flush=True)
            continue
        if 'MODE = "saved"' not in nb.cells[1].source:
            raise ValueError(f'{path}: saved mode is required')
        print('DEMO_START',path.parent.name,flush=True)
        kernel=KernelManager(kernel_name='python3')
        kernel.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(nb,km=kernel,timeout=180,resources={'metadata':{'path':str(path.parent)}},allow_errors=False)
        started=time.perf_counter()
        try:
            client.execute()
        finally:
            if kernel.has_kernel:
                kernel.shutdown_kernel(now=True)
        nb.metadata['recsys']={'mode':'saved','training_on_run_all':False,'executed_during_packaging':True,'result_origin':'historical recorded experiments'}
        nbformat.write(nb,path)
        errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
        if errors:raise RuntimeError(f'{path}: error output')
        record=dict(notebook=str(path.relative_to(ROOT)),mode='saved',seconds=round(time.perf_counter()-started,3),code_cells=sum(c.cell_type=='code' for c in nb.cells),output_blocks=sum(len(c.get('outputs',[])) for c in nb.cells),errors=0,training_started=False)
        report.append(record)
        (ROOT/'results/demo_execution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('DEMO_COMPLETE',record,flush=True)


if __name__=='__main__':main()
