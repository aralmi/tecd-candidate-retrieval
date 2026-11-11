"""Run a fresh five-family study, with original prepared inputs as an optional cache."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
from .session import ResearchSession


class BestModelStudy:
    phase_names = {
        'prepare': 'Подготовка данных',
        'classical': 'Классические методы',
        'two_tower': 'Двухбашенная модель',
        'sequence': 'Последовательная модель',
        'compare': 'Сравнение выдач',
    }
    phases = tuple(phase_names)

    def __init__(self, run_name, raw_dir, prepared_root=None):
        self.session=ResearchSession('train',run_name,raw_dir)
        self.root=self.session.root
        self.workspace=self.session.workspace
        self.results=self.workspace/'bench_results'
        self.prepared_root=Path(prepared_root).resolve() if prepared_root and Path(prepared_root).exists() else None
        self.config=json.loads((self.root/'configs/best_models.json').read_text())

    def run(self,phase):
        if phase not in self.phases:raise ValueError(phase)
        title = self.phase_names[phase]
        self.session._prepare_workspace()
        target=self.workspace/'benchmark_config.json'
        if target.exists() and json.loads(target.read_text())!=self.config:
            raise ValueError('Benchmark configuration changed; use another RUN_NAME')
        target.write_text(json.dumps(self.config,ensure_ascii=False,indent=2)+'\n')
        self.results.mkdir(exist_ok=True)
        marker=self.results/(phase+'_complete.json')
        if marker.exists() and json.loads(marker.read_text()).get('completed'):
            print(f'Этап «{title}» уже выполнен для этого RUN_NAME.', flush=True)
            return
        index=self.phases.index(phase)
        if index and not (self.results/(self.phases[index-1]+'_complete.json')).exists():
            raise RuntimeError('Run the preceding notebook cell first')
        env=os.environ.copy();env['PYTHONPATH']=str(self.workspace);env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
        if self.prepared_root:env['RECSYS_PREPARED_STUDY']=str(self.prepared_root)
        else:env.pop('RECSYS_PREPARED_STUDY',None)
        log=self.results/(phase+'.log')
        log_path = log.relative_to(self.root)
        status=dict(phase=phase,status='running',started_unix=time.time(),model_policy='fresh fits')
        (self.results/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        command=[sys.executable,'-u',str(self.workspace/'best_models.py'),phase]
        print(f'Этап «{title}» запущен. Журнал {log_path}', flush=True)
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=self.workspace, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait()
            except BaseException:
                process.terminate();process.wait();raise
        status.update(status='complete' if code==0 else 'failed',finished_unix=time.time(),exit_code=code)
        (self.results/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        if code:
            raise RuntimeError(f'Этап «{title}» завершился с кодом {code}. Журнал {log_path}')
        print(f'Этап «{title}» завершён.', flush=True)

    def table(self,name):
        return pd.read_csv(self.results/name)

    def model_metrics(self):
        folder=self.results/'rankings'
        records=[]
        for path in sorted(folder.glob('*.json')):
            if path.stem in self.config['fusions']:continue
            record=json.loads(path.read_text());record['method']=path.stem;records.append(record)
        return pd.DataFrame(records)

    def export(self,chapter):
        destination=Path(chapter)/'results';destination.mkdir(parents=True,exist_ok=True)
        names=['data_manifest.json','model_comparison.csv','fusion_comparison.csv','ranking_overlap.csv',
               'two_tower_selection.json','sequence_selection.json','two_tower_learning_curve.csv',
               'two_tower_training_loss.csv','sequence_learning_curve.csv','complete.json']
        for name in names:
            path=self.results/name
            if path.exists():shutil.copyfile(path,destination/name)
        (destination/'configuration.json').write_text(json.dumps(self.config,ensure_ascii=False,indent=2)+'\n')
        print(f'Таблицы и настройки сохранены в {os.path.relpath(destination, self.root)}.')
