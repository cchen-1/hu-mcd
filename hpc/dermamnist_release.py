"""One-shot exact-release preparation, duplicate-safe reuse and RDM archival.

Slurm only; no image selection/model execution/retry/fallback dataset. A failed
RDM inventory is not evidence of absence: stop before downloading in that case.
All experimental paths remain unpublished until both audit and archival pass.
"""
import base64
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from utils.run_tracking import atomic_json, sha256, utc_now

RELEASE = 'zenodo-12739457-md5-84920fb70c83b234c295b6f0d4ae2bc0'
FILES = {
    'dermamnist_corrected_224.npz': (1131258316, '84920fb70c83b234c295b6f0d4ae2bc0'),
    'DermaMNIST-C.csv': (747941, '1c54bc9f483e97a9d7dbbde076a45900'),
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def check_release_file(path, name):
    size, md5 = FILES[name]
    return Path(path).stat().st_size == size and digest(path, 'md5') == md5


def inventory(roots, out):
    """Inspect only named dataset/preparation trees, never all scratch/RDM."""
    rows=[];errors=[];visited=0
    for root in roots:
        root=Path(root)
        try:
            root.stat()
        except FileNotFoundError:
            rows.append(dict(root=str(root),status='ABSENT'));continue
        except OSError as exc:
            errors.append(dict(root=str(root),error=str(exc)));continue
        def onerror(exc): errors.append(dict(root=str(root),error=str(exc)))
        for folder,dirs,files in os.walk(root,onerror=onerror,followlinks=False):
            depth=len(Path(folder).relative_to(root).parts)
            if depth>=8 and dirs:
                errors.append(dict(root=str(root),error='Inventory depth cap exceeded'));dirs[:]=[]
            visited+=len(files)
            if visited>20000:raise RuntimeError('Targeted inventory exceeds bound; no download authorized until clarified')
            for name in files:
                if name not in FILES and not ('dermamnist_corrected_224' in name and name.endswith(('.npz','.part'))):
                    continue
                p=Path(folder)/name;rec=dict(path=str(p))
                try:
                    rec['bytes']=p.stat().st_size
                    # Byte-identical complete files may have a renamed basename.
                    target=name if name in FILES else 'dermamnist_corrected_224.npz'
                    rec['target']=target;rec['exact_match']=check_release_file(p,target)
                    if rec['exact_match']:rec['sha256']=sha256(p)
                except OSError as exc:
                    rec['error']=str(exc);errors.append(rec)
                rows.append(rec)
    result=dict(roots=list(map(str,roots)),entries=rows,errors=errors,scope='Named dataset/preparation roots, bounded to depth8/20000files; unknown access is not absence')
    atomic_json(out/'existing_file_inventory.json',result)
    if errors:raise RuntimeError('Incomplete existing-file inventory; no download. See existing_file_inventory.json')
    return rows


def fetch_once(name, destination, out, max_seconds):
    """No retries, resume or alternative host; partial evidence remains on error."""
    size, expected = FILES[name]
    url='https://zenodo.org/api/records/12739457/files/'+name+'/content'
    part=Path(str(destination)+'.part')
    if destination.exists() or part.exists():raise FileExistsError('No overwrite or retry: '+str(destination))
    command=['curl','--location','--fail','--show-error','--silent','--retry','0',
        '--connect-timeout','20','--max-time',str(max_seconds),'--max-filesize',str(size),
        '--output',str(part),url]
    record=dict(file=name,url=url,command=command,started=utc_now(),attempt=1,retries=0)
    atomic_json(out/(name+'.download.json'),record)
    p=subprocess.run(command,text=True,capture_output=True,timeout=max_seconds+30)
    (out/(name+'.download.stderr')).write_text(p.stderr)
    record.update(returncode=p.returncode,finished=utc_now(),received_bytes=part.stat().st_size if part.exists() else 0)
    atomic_json(out/(name+'.download.json'),record)
    if p.returncode:raise RuntimeError('Download failed; partial retained, no retry: '+name)
    if not check_release_file(part,name):raise ValueError('Published MD5/size mismatch: '+name)
    part.rename(destination)
    return dict(source=url,method='SINGLE_DOWNLOAD',md5=expected,sha256=sha256(destination))


def verify_archive(root):
    """Never replace an existing version. Its sealed receipt must verify fully."""
    receipt=json.loads((root/'ARCHIVE_COMPLETE.json').read_text())
    if receipt['release']!=RELEASE or receipt['status']!='PASS':raise ValueError('Existing RDM archive not sealed for this release')
    for name,entry in receipt['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive index path')
        if sha256(root/rel)!=entry['sha256'] or (root/rel).stat().st_size!=entry['bytes']:
            raise ValueError('Existing archive content differs: '+name)
    for name in FILES:
        if not check_release_file(root/'raw'/name,name):raise ValueError('Existing archive release identity differs')
    return receipt


def publish_archive(dataset, final, job, out):
    if final.exists():
        receipt=verify_archive(final)
        return dict(status='REUSED_VERIFIED_ARCHIVE',path=str(final),receipt_sha256=sha256(final/'ARCHIVE_COMPLETE.json'))
    stage=final.parent/('.'+RELEASE+'.incoming-'+job)
    stage.mkdir(exist_ok=False)
    index={}
    for source in sorted(dataset.rglob('*')):
        if not source.is_file():continue
        rel=source.relative_to(dataset);dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        expected=sha256(source)
        if sha256(dest)!=expected:raise ValueError('RDM copy verification failed: '+str(rel))
        index[str(rel)]=dict(sha256=expected,bytes=dest.stat().st_size)
    receipt=dict(status='PASS',release=RELEASE,job=job,archived_at=utc_now(),files=index)
    atomic_json(stage/'ARCHIVE_COMPLETE.json',receipt)
    if final.exists():raise FileExistsError('Archive appeared during publication; do not overwrite')
    stage.rename(final)
    verify_archive(final)
    return dict(status='ARCHIVED_VERIFIED',path=str(final),receipt_sha256=sha256(final/'ARCHIVE_COMPLETE.json'))


def run(config,out):
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit() or not re.fullmatch(r'bun[0-9]{3}',socket.gethostname().split('.')[0]):
        raise RuntimeError('All remote inspection/processing requires Slurm compute node')
    if config['authorization']!='APPROVED_DATA_PREPARATION_ONLY':raise ValueError('Missing data-only authorization')
    start=time.monotonic()
    def mark(stage,**kw):
        atomic_json(out/'data_progress.json',dict(stage=stage,job=job,utc=utc_now(),**kw));print(stage,kw,flush=True)
    mark('RDM_access_and_existing_file_inspection')
    rdm=Path(config['rdm_collection']);diagnostics=[]
    # Explicit directory use triggers QRISdata automount. It does not bypass ACLs.
    for p in [Path('/QRISdata'),rdm,rdm/'datasets',Path(config['known_derm7pt_zip'])]:
        rec=dict(path=str(p),uid=os.geteuid(),gids=os.getgroups())
        try:
            st=p.stat();rec.update(mode=oct(st.st_mode),owner_uid=st.st_uid,owner_gid=st.st_gid)
            if p==rdm:rec['directory_entries']=sorted(os.listdir(str(p)+'/'))
            rec['status']='READABLE_STAT'
        except OSError as exc:rec.update(status='UNAVAILABLE',error=str(exc))
        diagnostics.append(rec)
    atomic_json(out/'rdm_access.json',diagnostics)
    # A permission error anywhere in the searched dataset tree cannot justify
    # a claim that the target file is absent, so do not download in that case.
    roots=[Path(p) for p in config['candidate_roots']]
    entries=inventory(roots,out)
    scratch_parent=Path(config['scratch_parent']);scratch_parent.mkdir(parents=True,exist_ok=True)
    final=rdm/'datasets/dermamnist-c'/RELEASE
    final.parent.mkdir(parents=True,exist_ok=True)
    # Shared release lock covers inventory recheck, download and publication.
    # A second preparation cannot download the same release concurrently.
    with (final.parent/('.'+RELEASE+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another preparation holds release lock; no duplicate download')
        # Verify archival write access before spending download time.
        if final.exists():verify_archive(final)
        entries=inventory(roots,out)
        dataset=scratch_parent/job;dataset.mkdir(exist_ok=False)
        for name in ('raw','provenance','manifests','prepared'):(dataset/name).mkdir()
        provenance=dataset/'provenance'
        atomic_json(provenance/'execution.json',dict(job=job,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            config=config,worker_sha256=sha256(Path(__file__)),scope='Data preparation only; no downstream experiment authorization'))
        atomic_json(provenance/'inventory.json',dict(entries=entries,roots=list(map(str,roots))))
        bundle=Path(config['support_bundle'])
        if sha256(bundle)!=config['support_bundle_sha256']:raise ValueError('Support bundle hash mismatch')
        for relative,item in json.loads(bundle.read_text()).items():
            rel=Path(relative)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe support path')
            content=base64.b64decode(item['base64'],validate=True)
            if hashlib.sha256(content).hexdigest()!=item['sha256']:raise ValueError('Provenance support hash mismatch: '+relative)
            dest=dataset/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        acquired={}
        # Metadata first; reuse any verified complete copy before network access.
        for name in ('DermaMNIST-C.csv','dermamnist_corrected_224.npz'):
            mark('acquire_'+name,scratch=str(dataset))
            matches=[e for e in entries if e.get('target')==name and e.get('exact_match')]
            dest=dataset/'raw'/name
            if dest.exists():
                if not check_release_file(dest,name):raise ValueError('Bundled metadata identity mismatch')
                acquired[name]=dict(method='REUSED_HASH_VERIFIED_LOCAL_OFFICIAL_METADATA',source=config['support_bundle'],sha256=sha256(dest))
            elif matches:
                source=Path(matches[0]['path']);shutil.copyfile(source,dest)
                if not check_release_file(dest,name):raise ValueError('Reused copy failed verification')
                acquired[name]=dict(method='REUSED_EXISTING_VERIFIED',source=str(source),sha256=sha256(dest))
            else:
                remaining=int(6600-(time.monotonic()-start))
                if remaining<60:raise RuntimeError('Download time budget exhausted; no extension')
                acquired[name]=fetch_once(name,dest,out,min(180,remaining) if name.endswith('.csv') else remaining)
            atomic_json(provenance/'acquisition.json',acquired)
        mark('array_and_identity_audit')
        audit=Path(__file__).with_name('prepare_dermamnist.py')
        if sha256(audit)!=config['auditor_sha256']:raise ValueError('Auditor hash mismatch')
        p=subprocess.run([sys.executable,str(audit),str(dataset)],text=True,capture_output=True,timeout=300)
        (out/'audit.stdout').write_text(p.stdout);(out/'audit.stderr').write_text(p.stderr)
        if p.returncode:raise RuntimeError('Exact array/split audit failed; no archive/config publication')
        report=json.loads((dataset/'manifests/dataset_audit.json').read_text())
        if report['status']!='PASS':raise ValueError('Auditor did not pass')
        expected=dict(train=8215,val=573,test=1227)
        if {s:r['images'] for s,r in report['splits'].items()}!=expected:raise ValueError('Unexpected published split counts')
        rows=list(csv.DictReader((dataset/'raw/DermaMNIST-C.csv').open()));byid={r['image_id']:r for r in rows}
        pairs=list(csv.DictReader((provenance/'author_confirmed_duplicates.csv').open()));cross=[]
        for p in pairs:
            a,b=Path(p['from_img']).stem,Path(p['to_img']).stem
            if a not in byid or b not in byid:raise ValueError('Author duplicate ID absent from metadata')
            if byid[a]['split']!=byid[b]['split']:cross.append(dict(left=a,right=b,split_left=byid[a]['split'],split_right=byid[b]['split']))
        atomic_json(dataset/'manifests/known_duplicates.json',dict(checked_pairs=len(pairs),cross_split_pairs=cross))
        if cross:raise ValueError('Known duplicate crossing found; no experiment-ready config')
        (provenance/'SHA256SUMS').write_text(''.join(sha256(dataset/'raw'/n)+'  raw/'+n+'\n' for n in FILES))
        for source in (Path(__file__),audit):shutil.copy2(source,provenance/source.name)
        # Include checks/metadata in small local collector output; raw images stay
        # in scratch/RDM, avoiding another automatic gigabyte-sized transfer.
        shutil.copytree(dataset/'manifests',out/'manifests')
        shutil.copytree(provenance,out/'provenance')
        mark('RDM_archival_copy_and_reverification')
        archive=publish_archive(dataset,final,job,out)
        mark('publish_dataset_paths')
        paths=dict(status='DATA_PREPARATION_PASS',release=RELEASE,scratch_root=str(dataset),
            scratch_npz=str(dataset/'raw/dermamnist_corrected_224.npz'),scratch_metadata=str(dataset/'raw/DermaMNIST-C.csv'),
            rdm_root=str(final),rdm_npz=str(final/'raw/dermamnist_corrected_224.npz'),
            expected_md5=FILES['dermamnist_corrected_224.npz'][1],archive_receipt_sha256=archive['receipt_sha256'],
            dataset_audit_sha256=sha256(dataset/'manifests/dataset_audit.json'),downstream_experiments_authorized=False)
        atomic_json(out/'dataset_paths.json',paths)
        # A versioned pointer can be consumed only after local acceptance; never
        # override an unrelated existing pointer, sample list or experiment config.
        pointer=scratch_parent/('dataset_paths-'+job+'.json');atomic_json(pointer,paths)
        result=dict(status='PREPARED_ARCHIVED_PENDING_LOCAL_ACCEPTANCE',dataset_paths=paths,path_config=str(pointer),archive=archive,acquired=acquired)
        mark(result['status'],path_config=str(pointer))
        return result
