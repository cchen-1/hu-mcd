"""Complete and verify the known probe's missing checks, without rerunning SAM."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_bound(item):
    path = Path(item['path'])
    if digest(path) != item['sha256']:
        raise ValueError('Prerequisite evidence changed: ' + str(path))
    return json.loads(path.read_text())


def verify_evidence(evidence, release, config):
    original = read_bound(evidence['original_probe'])
    diagnosis = read_bound(evidence['diagnosis'])
    if original['status'] != 'FAILED' or original['slurm_job_id'] != '28204575':
        raise ValueError('This resolution applies only to the preserved failed probe 28204575')
    if diagnosis['status'] != 'DIAGNOSIS_COMPLETED' or diagnosis['diagnosis_job_id'] != '28206208':
        raise ValueError('The bound L40S diagnosis is not complete')
    if diagnosis['research_commit'] != original['actual_commit'] or diagnosis['gpu_name'] != 'NVIDIA L40S':
        raise ValueError('Diagnostic research identity or GPU mismatch')
    if not diagnosis['model_state_unchanged'] or diagnosis['model_training']:
        raise ValueError('Diagnostic model state changed')
    checks = diagnosis['comparisons']
    for batch in (1, 2, 8):
        if not checks[f'cuda_default_same_batch{batch}_masked_vs_plain']['exact_equal']:
            raise ValueError('Matched-batch mask check failed')
    if (not checks['cuda_cudnn_tf32_off_original_mixed2_vs_plain1']['passes_original_tolerance']
            or not checks['cuda_default_vs_restored']['exact_equal']):
        raise ValueError('TF32 fault isolation did not pass')
    if checks['cuda_default_original_mixed2_vs_plain1'] != checks['cuda_default_plain2_vs_plain1']:
        raise ValueError('Original error is not accounted for by batch size')
    old = original['plan']['config']
    keys = ('dataset_manifest_sha256', 'sam_checkpoint_sha256', 'resnet_checkpoint_sha256',
            'class_name', 'model_name', 'layer_name', 'max_shortest_side', 'seed',
            'train_images', 'validation_images', 'segmentation', 'clustering')
    if any(config[k] != old[k] for k in keys) or config['batch_size'] != old['batch_size']:
        raise ValueError('Formal inputs/science settings differ from verified evidence')
    if config['precision'] != diagnosis['default_precision']:
        raise ValueError('Formal precision differs from diagnosed defaults')
    for name, expected in evidence['core_sha256'].items():
        if digest(Path(release) / name) != expected:
            raise ValueError('Scientific core differs from diagnosed source: ' + name)
    if digest(Path(release) / 'utils/scientific_records.py') != evidence['numerics_sha256']:
        raise ValueError('Numerical check implementation differs from completed preflight')
    for name in ('features', 'logits'):
        if digest(evidence[name]['path']) != evidence[name]['sha256']:
            raise ValueError('Original cached ' + name + ' changed')
    return original, diagnosis


def verify_completed(report_path, report_sha256, release, config):
    report = read_bound({'path': report_path, 'sha256': report_sha256})
    if report['status'] != 'PASS':
        raise ValueError('Cached prerequisite completion did not pass')
    if digest(Path(release) / 'hpc/prerequisites.py') != report['completion_helper_sha256']:
        raise ValueError('Prerequisite verifier changed since completion')
    if digest(Path(release) / 'hpc/reference_probe.py') != report['boundary_helper_sha256']:
        raise ValueError('Boundary helper changed since completion')
    verify_evidence(report['evidence'], release, config)
    if not report['checks']['ssc']['passed'] or not report['checks']['algebra']['passed']:
        raise ValueError('Missing completed algebra/SSC checks')
    # Preserve the original failure; this is a separate resolved-prerequisite record.
    return {'kind': 'failed_probe_plus_diagnosis_and_cached_completion',
            'original_probe_job_id': '28204575', 'original_probe_status': 'FAILED',
            'diagnosis_job_id': '28206208', 'completion_job_id': report['job_id'],
            'report_path': str(report_path), 'report_sha256': report_sha256}


def complete(plan):
    import re
    if not os.environ.get('SLURM_JOB_ID', '').isdigit() or not re.fullmatch(r'bun[0-9]{3}', socket.gethostname().split('.')[0]):
        raise RuntimeError('Slurm compute node required')
    release = Path(plan['release'])
    sys.path.insert(0, str(release))
    import numpy as np
    import torch
    from scipy import sparse
    from utils import utils_mcd
    # Numerical helper is transmitted and pinned, so it can be tested before deployment.
    import base64
    code = base64.b64decode(plan['numerics_base64'], validate=True)
    if hashlib.sha256(code).hexdigest() != plan['evidence']['numerics_sha256']:
        raise ValueError('Numerical helper payload mismatch')
    scope = {'__name__': 'preflight_numerical_checks'}
    exec(compile(code, 'preflight_numerical_checks.py', 'exec'), scope)
    # The release's core is unchanged; the numerical helper may add output records only.
    evidence = plan['evidence']
    check_evidence = dict(evidence)
    check_evidence['numerics_sha256'] = digest(release / 'utils/scientific_records.py')
    verify_evidence(check_evidence, release, plan['config'])
    output = Path(plan['output_root']) / os.environ['SLURM_JOB_ID'] / 'prerequisites'
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
              'evidence': evidence, 'completion_helper_sha256': plan['completion_helper_sha256'],
              'boundary_helper_sha256': plan['boundary_helper_sha256'],
              'research_commit': subprocess.check_output(['git', '-C', str(release), 'rev-parse', 'HEAD'], text=True).strip(),
              'checks': {}, 'scope': 'Cached FC + missing algebra/SSC; no SAM or concept fitting'}
    def save():
        target = output / 'prerequisites.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(target)
    save()
    try:
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        acts = np.load(evidence['features']['path'], allow_pickle=False)
        logits = np.load(evidence['logits']['path'], allow_pickle=False)
        weights_path = plan['config']['resnet_checkpoint']
        if digest(weights_path) != plan['config']['resnet_checkpoint_sha256']:
            raise ValueError('Classifier checkpoint changed')
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        numerical_checks = scope['numerical_checks']
        fc, _ = numerical_checks(acts, logits, state['fc.weight'].numpy(), state['fc.bias'].numpy(), None)
        report['checks']['cached_fc'] = fc
        save()
        fixture = np.array([[1., 2., 3.], [-4., 1., 0.]])
        weight = np.array([2., -3., 1.]); bias = 4.
        bases = [np.array([[1., 0., 0.]]), np.array([[.6, .8, 0.]]), np.array([[0., 0., 1.]])]
        checks, arrays = numerical_checks(fixture, fixture @ weight + bias, weight, bias, bases)
        np.savez_compressed(output / 'algebra.npz', **arrays)
        report['checks']['algebra'] = {**checks, 'passed': True}
        save()
        keep = ~np.all(acts == 0, axis=1)
        small = acts[keep][:128]
        small = small / np.linalg.norm(small, axis=1, keepdims=True)
        started = time.perf_counter()
        matrix = utils_mcd.compute_sparse_repr_matrix(small, n_jobs=1)
        if matrix.shape != (len(small), len(small)) or not np.isfinite(matrix.data).all() or not matrix.nnz:
            raise ValueError('SSC primitive returned invalid/empty coefficients')
        if np.any(matrix.diagonal() != 0):
            raise ValueError('SSC contains self representation')
        sparse.save_npz(output / 'ssc.npz', matrix)
        report['checks']['ssc'] = {'passed': True, 'rows': len(small), 'zero_rows_excluded': int((~keep).sum()),
                                  'nnz': int(matrix.nnz), 'seconds': time.perf_counter() - started}
        report['artifacts'] = {n: {'sha256': digest(output / n), 'bytes': (output / n).stat().st_size}
                               for n in ('algebra.npz', 'ssc.npz')}
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import base64
    complete(json.loads(base64.b64decode(sys.argv[1], validate=True)))
