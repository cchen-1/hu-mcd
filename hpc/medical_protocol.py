"""Bounded medical preparation; authorization and Slurm gates precede data access.

Pure helpers are testable with synthetic arrays. No data/model execution on import.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket

LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
COUNTS = {'train': 8215, 'val': 573, 'test': 1227}
MODES = ('medical-overlap', 'medical-classifier')
CAPS = {
    'medical-overlap': dict(cpus=4, memory='8G', time='00:30:00', gpu=None),
    'medical-classifier': dict(cpus=4, memory='16G', time='04:00:00', gpu='l40s:1'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def workload(images, epochs, batch_size, validation_images=573):
    """Every image exactly once per epoch, with the final partial batch kept."""
    if any(type(x) is not int or x <= 0 for x in (images, epochs, batch_size)):
        raise ValueError('Positive integer workload required')
    steps = math.ceil(images / batch_size)
    return dict(images=images, epochs=epochs, batch_size=batch_size,
                steps_per_epoch=steps, optimizer_updates=steps * epochs,
                training_image_visits=images * epochs,
                validation_image_visits=validation_images * epochs,
                final_batch_size=images % batch_size or batch_size)


def learning_rate(epoch, protocol):
    if not 1 <= epoch <= protocol['epochs']:
        raise ValueError('Epoch outside the fixed schedule')
    drops = sum(epoch > n for n in protocol['schedule']['drop_after_completed_epochs'])
    return protocol['optimizer']['lr'] * protocol['schedule']['gamma'] ** drops


def improves(score, previous):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Undefined/nonfinite validation selection metric')
    return previous is None or score > previous


def validate_config(config, mode, approved=False):
    if mode not in MODES or config['mode'] != mode:
        raise ValueError('Wrong medical execution mode')
    p = config['protocol']
    if digest(p) != config['protocol_sha256']:
        raise ValueError('Protocol digest mismatch')
    if config['resources'] != CAPS[mode]:
        raise ValueError('Resource change needs a new proposal/approval')
    if approved:
        a = config['authorization']
        if (a.get('status') != 'APPROVED' or not a.get('user_decision') or
                a.get('approved_protocol_sha256') != config['protocol_sha256']):
            raise ValueError('H/M execution requires explicit approval of this protocol digest')
    if mode == 'medical-classifier':
        if p['labels'] != LABELS or p['split_counts'] != COUNTS:
            raise ValueError('Source label/count identity differs from accepted release')
        w = workload(COUNTS['train'], p['epochs'], p['batch_size'])
        if w != p['workload']:
            raise ValueError('Workload inconsistent with schedule')
        if p['resume'] or p['automatic_extension'] or p['drop_last'] or p['amp']:
            raise ValueError('No resume, extension, dropped rows or AMP in this proposal')
        implemented = dict(model='timm0.6.13-standard-resnet50',
                           augmentation='independent_horizontal_vertical_flips_p0.5_train_only',
                           loss='unweighted_CE_no_smoothing',
                           selection='seven_class_validation_macroOVR_AUC_strict_improvement_earliest_tie',
                           downstream_medical_discovery=False, derm7pt_inference=False)
        if any(p[k] != v for k,v in implemented.items()):
            raise ValueError('Proposed implementation does not support the changed scientific rule')
        if p['work_seconds_ceiling'] > 14220 or p['output_bytes_ceiling'] > 5*1024**3:
            raise ValueError('Classifier processing/output budget exceeds proposal')
    elif (p['source_images'] != 10015 or p['derm_images'] != 1011 or
          p['hash_bits'] != 64 or p['automatic_exclusions'] or not p['retain_all_candidates'] or
          p['work_seconds_ceiling'] > 1680 or p['output_bytes_ceiling'] > 2*1024**3):
        raise ValueError('Overlap scope/budget differs from proposal')
    return p


def execution_gate(config, mode):
    validate_config(config, mode, approved=True)
    if not (os.environ.get('SLURM_JOB_ID', '').isdigit() and
            re.fullmatch(r'bun[0-9]{3}', socket.gethostname().split('.')[0])):
        raise RuntimeError('Medical execution requires an authenticated Slurm compute allocation')


def bound_inputs(protocol, output):
    """Bind accepted artifacts without repeating the dataset audit or mutating them."""
    from utils.run_tracking import sha256, atomic_json
    observed = {}
    for name, entry in protocol['inputs'].items():
        p = Path(entry['path'])
        actual = sha256(p)
        observed[name] = dict(path=str(p), sha256=actual,
                              expected_sha256=entry['sha256'])
        atomic_json(Path(output) / 'input_binding.json', observed)
        if actual != entry['sha256']:
            raise ValueError('Input/initializer identity mismatch: ' + name)
    return observed


def manifest_rows(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for split, count in COUNTS.items():
        r = [x for x in rows if x['split'] == split]
        if len(r) != count or [int(x['array_row']) for x in r] != list(range(count)):
            raise ValueError('Missing, duplicated or reordered source rows: ' + split)
        if any(LABELS[int(x['label'])] != x['class_name'] for x in r):
            raise ValueError('Manifest label mapping mismatch')
        result[split] = r
    if len(rows) != sum(COUNTS.values()) or len({x['image_id'] for x in rows}) != len(rows):
        raise ValueError('Duplicate image identity or unrecognized split')
    return result


def require_frozen_test(frozen, checkpoint_sha256, completed_epochs, planned_epochs):
    if (completed_epochs != planned_epochs or frozen.get('checkpoint_sha256') != checkpoint_sha256
            or frozen.get('test_outcomes_used_for_selection') is not False):
        raise ValueError('Test outcomes require a frozen selected checkpoint after the full schedule')
