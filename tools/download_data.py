"""Explicit download of the pinned Marketplace files. Never called in saved mode."""
from pathlib import Path
import argparse
import hashlib
import json
import requests

REVISION='79bcf78cd4699267f561ad3fd51fe7b52af53083'
BASE=f'https://huggingface.co/datasets/t-tech/T-ECD/resolve/{REVISION}/dataset/small/marketplace/'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    root=args.destination.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'download_manifest.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'revision':REVISION,'files':{}}
    if old['revision']!=REVISION:raise ValueError('Destination contains another dataset revision')
    names=['items.pq']+[f'events/{d:05}.pq' for d in [*range(1174,1300),*range(1301,1308)]]
    for name in names:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        known=old['files'].get(name)
        if target.exists() and known and target.stat().st_size==known['bytes']:
            digest=hashlib.sha256()
            with target.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
            if digest.hexdigest()==known['sha256']:
                print('Cached:',name,flush=True);continue
        if target.exists():raise FileExistsError(f'Existing unverified file: {target}. Use a separate download directory.')
        temporary=target.with_suffix('.part');digest=hashlib.sha256();size=0
        with requests.get(BASE+name,stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as out:
                for block in response.iter_content(4*1024**2):out.write(block);digest.update(block);size+=len(block)
        if not size:raise IOError(f'Empty download: {name}')
        temporary.replace(target);old['files'][name]={'bytes':size,'sha256':digest.hexdigest()}
        manifest_path.write_text(json.dumps(old,indent=2)+'\n');print('Downloaded:',name,size,flush=True)

if __name__=='__main__':main()
