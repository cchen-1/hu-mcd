"""Read-only Slurm diagnostics for a known Q9903 access failure; no download."""
import grp
import os
from pathlib import Path
import pwd
import socket
import subprocess
from utils.run_tracking import atomic_json, utc_now


def run(config,out):
    root=config['collection']
    if root!='/QRISdata/Q9903':raise ValueError('This diagnostic is scoped to the user collection')
    report=dict(time_utc=utc_now(),host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],
        uid=os.geteuid(),username=pwd.getpwuid(os.geteuid()).pw_name,
        gids=os.getgroups(),groups=[grp.getgrgid(g).gr_name for g in os.getgroups()],checks=[])
    commands=[['id','-u','uqcche38'],['id','-G','uqcche38'],
              ['ls','-ld',root+'/'],
              ['findmnt','--raw','--noheadings','--output','TARGET,FSTYPE','--target','/QRISdata']]
    for command in commands:
        try:
            p=subprocess.run(command,text=True,capture_output=True,timeout=30)
            item=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except Exception as exc:item=dict(command=command,error=str(exc))
        report['checks'].append(item);atomic_json(out/'rdm_diagnostic.json',report)
    # Try actual directory access with the documented trailing slash first.
    # Do not let a preliminary stat prevent the automount access attempt.
    for path in [root+'/',root+'/datasets/',root+'/datasets/derm7pt/release_v0/raw/']:
        item=dict(path=path)
        try:
            entries=sorted(os.listdir(path));item.update(status='READABLE',entries=entries)
            s=os.stat(path);item.update(owner_uid=s.st_uid,owner_gid=s.st_gid,mode=oct(s.st_mode),
                                     access_read=os.access(path,os.R_OK),access_write=os.access(path,os.W_OK),access_traverse=os.access(path,os.X_OK))
        except OSError as exc:item.update(status='DENIED_OR_MISSING',error=str(exc))
        report['checks'].append(item);atomic_json(out/'rdm_diagnostic.json',report)
    report.update(status='DIAGNOSTIC_COMPLETE',file_mutations=0,network_downloads=0,
        note='os.access is advisory only; no write attempted. No ACL or membership changes.')
    atomic_json(out/'rdm_diagnostic.json',report)
    return report
