"""One-attempt official HAM10000 download. No automatic retry or overwrite."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

NAMES = ['ISIC2018_Task3_Training_Input.zip',
         'ISIC2018_Task3_Training_LesionGroupings.csv',
         'ISIC2018_Task3_Training_GroundTruth.zip']
BASE = 'https://isic-archive.s3.amazonaws.com/challenges/2018/'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True)
    dest=Path(parser.parse_args().destination);dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in NAMES:
        path=dest/name; receipt=dest/(name+'.receipt.json')
        if path.exists() or path.with_suffix(path.suffix+'.part').exists() or receipt.exists():
            raise FileExistsError(f'Existing artifact: {path}; inspect before reuse, no overwrite')
        record=dict(url=BASE+name,file=name,status='STARTED',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        receipt.write_text(json.dumps(record,indent=2))
        try:
            with urllib.request.urlopen(BASE+name,timeout=60) as response:
                headers=dict(response.headers.items());length=int(response.headers['Content-Length'])
                if length>4*1024**3:raise ValueError('Unexpected file size over4GiB')
                if shutil.disk_usage(dest).free<length+1024**3:raise OSError('Insufficient local free space')
                record.update(headers=headers,expected_bytes=length,final_url=response.url)
                receipt.write_text(json.dumps(record,indent=2))
                print(f'START {name} {length} bytes',flush=True)
                sha=hashlib.sha256();md5=hashlib.md5();n=0;last=0
                part=path.with_suffix(path.suffix+'.part')
                with part.open('xb') as f:
                    while True:
                        b=response.read(4*1024**2)
                        if not b:break
                        n+=len(b)
                        if n>length:raise ValueError('Response exceeds declared length')
                        f.write(b);sha.update(b);md5.update(b)
                        if n-last>=128*1024**2:
                            print(f'PROGRESS {name} {n}/{length}',flush=True);last=n
                if n!=length:raise ValueError('Truncated download')
                etag=response.headers.get('ETag','').strip('"')
                # Multipart ETags are not a whole-file MD5; never label them so.
                match=None
                if re.fullmatch('[0-9a-fA-F]{32}',etag):
                    match=md5.hexdigest().lower()==etag.lower()
                    if not match:raise ValueError('Single-part ETag differs from MD5')
                part.rename(path)
                record.update(status='DOWNLOADED',bytes=n,sha256=sha.hexdigest(),md5=md5.hexdigest(),
                              single_part_etag_md5_match=match,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                receipt.write_text(json.dumps(record,indent=2));records.append(record)
                print(f'DONE {name} sha256={sha.hexdigest()}',flush=True)
        except Exception as exc:
            record.update(status='FAILED_NO_RETRY',error=repr(exc))
            receipt.write_text(json.dumps(record,indent=2));raise
    (dest/'download_manifest.json').write_text(json.dumps(dict(
        source='Official ISIC2018 Task3 Training HAM10000; not ISIC validation/test',
        source_page='https://challenge.isic-archive.com/data/',files=records),indent=2))


if __name__=='__main__':main()
