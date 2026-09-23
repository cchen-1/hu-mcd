"""Fixed R101 fitting with accepted HAM checkpoint; no external hold-out claims."""
import json
import os
from pathlib import Path
import re
import socket
import time
from hpc.medical_protocol import digest, bound_inputs
from hpc.medical_discovery import SAM
from utils.run_tracking import atomic_json, sha256, utc_now

CAPS=dict(cpus=8,memory='32G',time='01:00:00',gpu='l40s:1')
ALGORITHM=dict(sam_type='vit_h',points_per_side=32,min_mask_region_area=256,
    cropping_mode=0,masking_mode=-1,erosion_threshold=0.25,layer='global_pool',
    feature_dim=2048,cluster_algo='sparse_subspace_clustering',n_clusters=None,
    norm_acts=True,min_coverage=0.0,max_samples=None,dimension_estimator='ratio',
    zero_features='released_exact_zero_exclusion_with_raw_evidence',display_seed=4301)
CHECKPOINT='bee957a2a13b894ea12a6f41cc9d5b38f206fc65b4a60fd1ebd81446099c0d37'

def validate(config):
    p=config['protocol'];a=config['authorization']
    expected=dict(training_images=101,held_out_images=0,target_label=4,batch_size=8,
        selected_epoch=27,outlier_quantile=1.0,min_cluster_size=50,sampling_seed=43,
        external_inference=False,classifier_training=False,automatic_retry=False,
        max_shortest_side=300,extra_square_resize_before_sam=False,
        work_seconds_ceiling=3420,output_bytes_ceiling=8*1024**3)
    if (config['mode']!='derm-fitting' or config['resources']!=CAPS or digest(p)!=config['protocol_sha256']
        or a['status']!='APPROVED' or a['approved_protocol_sha256']!=digest(p) or not a['user_decision']
        or p['discovery_algorithm']!=ALGORITHM or any(p[k]!=v for k,v in expected.items())
        or p['inputs']['classifier']['sha256']!=CHECKPOINT or p['inputs']['sam']['sha256']!=SAM):
        raise ValueError('R101 fitting protocol, resource or weight binding differs')
    return p

def execution_gate(config,mode=None):
    validate(config)
    if not(os.environ.get('SLURM_JOB_ID','').isdigit() and re.fullmatch('bun[0-9]{3}',socket.gethostname().split('.')[0])):
        raise RuntimeError('R101 fitting requires Slurm compute allocation')

def fit_rows(manifest):
    rows=manifest['rows']
    if (manifest['count']!=101 or manifest['held_out']!=0 or len(rows)!=101
        or [r['fit_order'] for r in rows]!=list(range(101))
        or len({r['image_id'] for r in rows})!=101):
        raise ValueError('Fixed R101 identity/order/count mismatch')
    for r in rows:
        if (r['role']!='DISCOVERY_FIT_NO_HELD_OUT' or r['extra_square_resize_before_sam']
            or r['uses_classifier_cache'] or min(r['input_size_wh'])!=300):
            raise ValueError('Pre-SAM geometry or input role mismatch')
    return dict(training=rows,held_out=[],role='DISCOVERY_FIT_NO_HELD_OUT',patient_independence='NOT_ESTABLISHED')

def run(config,out):
    execution_gate(config);p=config['protocol'];out=Path(out);start=time.monotonic()
    bound_inputs(p,out)
    receipt=json.loads(Path(p['inputs']['classifier_acceptance']['path']).read_text())
    if receipt['status']!='ACCEPTED_ENGINEERING_FIXED_CLASSIFIER' or receipt['checkpoint_sha256']!=CHECKPOINT:
        raise ValueError('Classifier not accepted for fixed-model explanation')
    source=fit_rows(json.loads(Path(p['inputs']['fit_manifest']['path']).read_text()))
    for r in source['training']:
        if sha256(r['source_path'])!=r['inventory']['sha256'] or sha256(r['input_path'])!=r['input_sha256']:
            raise ValueError('Bound original/pre-SAM file changed')
    def budget(stage,**kw):
        atomic_json(out/'medical_discovery_progress.json',dict(stage=stage,utc=utc_now(),elapsed_seconds=time.monotonic()-start,**kw))
        if time.monotonic()-start>p['work_seconds_ceiling']:raise TimeoutError('Fixed fitting time ceiling; no extension')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>p['output_bytes_ceiling']:raise RuntimeError('Fixed fitting output ceiling')
    from hpc.medical_discovery import discover
    return discover(config,out,budget,fit_context=source)
