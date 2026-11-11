"""The same result interface for recorded runs and explicit fresh computation."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pandas as pd

from .io import project_root

CHAPTERS = ['01_data_and_protocol','02_baselines_and_classical','03_two_tower',
            '04_sequence_models','05_fusion_and_robustness','06_final_results']


class ResearchSession:
    def __init__(self, mode='saved', run_name='reproduction', raw_dir=None):
        if mode not in ('saved','train'):
            raise ValueError("MODE must be 'saved' or 'train'")
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',run_name):
            raise ValueError('RUN_NAME: letters, numbers, underscore and hyphen only')
        self.root=project_root()
        self.mode=mode
        self.workspace=self.root/'local_runs'/run_name/'engine'
        self.raw_dir=Path(raw_dir).expanduser().resolve() if raw_dir else None

    def run(self, chapter):
        if chapter not in CHAPTERS:
            raise ValueError(chapter)
        if self.mode=='saved':
            print('Используем сохранённые результаты.')
            return
        self._prepare_workspace()
        index=CHAPTERS.index(chapter)
        if index:
            previous=self.workspace/'chapter_results'/CHAPTERS[index-1]/'execution.json'
            if not previous.exists():
                raise FileNotFoundError(f'First run chapter {CHAPTERS[index-1]} in train mode, with the same RUN_NAME.')
        marker=self.workspace/'chapter_results'/chapter/'execution.json'
        if marker.exists() and json.loads(marker.read_text()).get('completed'):
            print(f'Ранее вычисленный результат этого RUN_NAME: {chapter}. Для нового запуска задайте новый RUN_NAME.')
            return
        env=os.environ.copy()
        env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
        env['PYTHONPATH']=str(self.workspace)
        log=self.workspace/'chapter_results'/chapter/'execution.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(self.workspace/'reproduce.py'),chapter]
        print('Выполняется подготовка/обучение/оценка:',chapter,flush=True)
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=self.workspace,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            code=process.wait()
        if code:
            raise RuntimeError(f'Chapter failed with exit code {code}. Log: {log}')

    def _prepare_workspace(self):
        if self.raw_dir is None or not (self.raw_dir/'items.pq').exists() or not (self.raw_dir/'events').is_dir():
            raise FileNotFoundError('Set RAW_DIR to the downloaded Marketplace directory containing items.pq and events/. See data/README.md.')
        engine=self.root/'engine'
        identity=hashlib.sha256()
        sources=sorted(engine.rglob('*.py'))
        for path in sources:
            identity.update(str(path.relative_to(engine)).encode());identity.update(path.read_bytes())
        identity.update((self.root/'configs/active_history.json').read_bytes())
        raw_metadata=[]
        for day in [*range(1174,1300),*range(1301,1308)]:
            path=self.raw_dir/f'events/{day:05}.pq'
            if not path.exists():raise FileNotFoundError(path)
            raw_metadata.append([day,path.stat().st_size,path.stat().st_mtime_ns])
        item_path=self.raw_dir/'items.pq'
        raw_metadata.append(['items',item_path.stat().st_size,item_path.stat().st_mtime_ns])
        identity.update(json.dumps(raw_metadata).encode())
        expected=dict(code_and_raw_metadata_sha256=identity.hexdigest(),raw_dir=str(self.raw_dir))
        marker=self.workspace.parent/'workspace.json'
        if marker.exists():
            if json.loads(marker.read_text())!=expected:
                raise ValueError('Code/config/raw files changed. Use a new RUN_NAME; old results are preserved.')
            return
        if self.workspace.exists():raise FileExistsError('Incomplete workspace. Choose a new RUN_NAME.')
        shutil.copytree(engine,self.workspace,ignore=shutil.ignore_patterns('__pycache__'))
        for path in ['data','artifacts','runs/active_history/configs','runs/active_history/artifacts','runs/active_history/data/processed']:
            (self.workspace/path).mkdir(parents=True,exist_ok=True)
        (self.workspace/'data/raw').symlink_to(self.raw_dir,target_is_directory=True)
        (self.workspace/'runs/active_history/data/raw').symlink_to(self.raw_dir,target_is_directory=True)
        shutil.copyfile(self.root/'configs/active_history.json',self.workspace/'runs/active_history/configs/pilot.json')
        marker.write_text(json.dumps(expected,indent=2)+'\n')

    def metrics(self, chapter):
        if self.mode=='train':
            path=self.workspace/'chapter_results'/chapter/'overview.csv'
            if not path.exists():raise FileNotFoundError(path)
            return pd.read_csv(path)
        folder=self.root/'research'/chapter/'results'
        files={
            '01_data_and_protocol':'data_quality_20261104__processed_quality.csv',
            '02_baselines_and_classical':'closing_checks_20261107__development_common_pop28.csv',
            '03_two_tower':'full_catalog_20261104__comparison.csv',
            '04_sequence_models':'prefix_sequence_20261104__comparison.csv',
            '05_fusion_and_robustness':'temporal_als_20261104__quota10_sensitivity.csv',
            '06_final_results':'closing_checks_20261107__final_comparison.csv',
        }
        return pd.read_csv(folder/files[chapter])

    def read(self, chapter, name):
        """Live aliases exist only for recomputed primary outputs; historical evidence is explicit."""
        path=self.root/'research'/chapter/'results'/name
        if self.mode=='train':
            aliases={
                '01_data_and_protocol':{'data_quality_20261104__daily_events.csv':self.workspace/'chapter_results'/chapter/'daily_events.csv','data_quality_20261104__processed_quality.csv':self.workspace/'chapter_results'/chapter/'processed_quality.csv'},
                '03_two_tower':{'full_catalog_20261104__comparison.csv':self.workspace/'chapter_results'/chapter/'overview.csv'},
                '04_sequence_models':{'prefix_sequence_20261104__comparison.csv':self.workspace/'chapter_results'/chapter/'overview.csv',
                    **{f'prefix_sequence_20261104__{encoder}__learning_curve.csv':self.workspace/f'runs/prefix_sequence_20261104/{encoder}/learning_curve.csv' for encoder in ['pool','sequence']}},
                '06_final_results':{'closing_checks_20261107__final_comparison.csv':self.workspace/'runs/closing_checks_20261107/final_comparison.csv',
                    'closing_checks_20261107__final_paired_difference.json':self.workspace/'runs/closing_checks_20261107/final_paired_difference.json',
                    'closing_checks_20261107__final_protocol_frozen.json':self.workspace/'runs/closing_checks_20261107/final_protocol_frozen.json',
                    'paired_metrics.csv':self.workspace/'chapter_results'/chapter/'paired_metrics.csv'},
            }
            if name in aliases.get(chapter,{}):path=aliases[chapter][name]
            else:print('Историческая дополнительная проверка:',name)
        if not path.exists():raise FileNotFoundError(path)
        return json.loads(path.read_text()) if path.suffix=='.json' else pd.read_csv(path)
