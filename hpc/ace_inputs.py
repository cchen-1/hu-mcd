"""Slurm-only ACE R input preparation; no model, fitting, SSH or submission.

Shared worker: MODES['ace-inputs'] = 'hpc.ace_inputs'; run(config, output).
Required config: dataset_manifest, dataset_manifest_sha256, licensed_root,
execution_commit. Optional golden_summary/golden_summary_sha256 and
golden_run_manifest/golden_run_manifest_sha256 must be supplied as a pair.
Optional approval_path/approval_sha256 embeds the approved R receipt and budget.
The parent owns scheduling (4 CPU, 8 GiB, 15 min), immutable release and collection.

Consumer contract: hash-bind ace_inputs_manifest.json, require status PASS, verify
its owned files. roles contains target400, discovery50, validation50, random2000,
each {path, sha256, count}. Each role file is a JSON array in REQUIRED loader order;
rows have input_path/input_sha256, source/sha256, source_id and selection_index.
Golden rows retain every original field. No symlink staging is needed: the model
worker must explicitly load these rows, not rediscover/sort their paths.
control/gradient are deliberately DEFERRED_EFFECTIVE_FEATURE_POOL. Their actual
identities must be frozen after feature eligibility is known, before CAV fitting.

Selection: all immediate regular-file targets in all 1000 n######## train folders,
without an extension filter; sorted synset/basename IDs; random.Random(43).shuffle
of candidate indices; first 2000 successfully decoded mode-RGB images. No class
exclusion, conversion, content deduplication or scientific-RNG consumption.
Duplicate content across different training IDs is retained and reported, not
silently filtered. Validation path/ID collisions and known Golden validation
content are excluded. We do not claim a full validation-content hash audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import socket
import subprocess
import time
import traceback
import warnings

from PIL import Image, __version__ as PILLOW_VERSION
from utils.run_tracking import atomic_json, sha256, utc_now

ROOT = Path(__file__).resolve().parents[1]
TARGET_COUNT, DISCOVERY_COUNT, VALIDATION_COUNT, RANDOM_COUNT, SYNSET_COUNT = 400, 50, 50, 2000, 1000
CLASS_NAME, SYNSET = 'golden_retriever', 'n02099601'
ORDER_VERSION = 'ace-r-inputs-v1:sorted-synset-basename:python-Random43-shuffle-indices'


def _load_checked(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('SHA256 mismatch: ' + str(path))
    return json.loads(data)


def _assert_runtime(config):
    # This check precedes all dataset reads, enumeration, decoding and hashing.
    if (not os.environ.get('SLURM_JOB_ID', '').isdigit()
            or not re.fullmatch(r'bun\d{3}', socket.gethostname().split('.')[0])):
        raise RuntimeError('ACE input preparation requires a Slurm compute node')
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 4:
        raise RuntimeError('Expected the approved 4-CPU input allocation')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if commit != config['execution_commit']:
        raise ValueError('Execution commit differs from input-preparation config')
    return commit


def _within(path, root):
    return Path(path).resolve(strict=True).is_relative_to(root.resolve(strict=True))


def _identity(row):
    return row.get('source_id', Path(row['source']).parent.name + '/' + Path(row['source']).name)


def verify_golden(config, licensed):
    dataset = _load_checked(config['dataset_manifest'], config['dataset_manifest_sha256'])
    if (dataset['class_name'] != CLASS_NAME or dataset['synset'] != SYNSET
            or dataset['seed'] != 43 or not dataset.get('ready_for_reference')
            or dataset.get('validation_issues')):
        raise ValueError('Golden fixed input manifest is not ready or has wrong identity')
    if Path(dataset['licensed_source_root']).resolve() != licensed:
        raise ValueError('Golden and random pool licensed roots differ')
    roles = {}
    for split, count, role, relative in (
        ('training', TARGET_COUNT, 'target400', CLASS_NAME),
        ('validation', VALIDATION_COUNT, 'validation50', 'val_imgs/' + CLASS_NAME + '_val'),
    ):
        rows = dataset[split]
        if len(rows) != count:
            raise ValueError('Wrong frozen Golden count: ' + split)
        prepared = Path(dataset['source_dir']) / relative
        if sorted(p.name for p in prepared.iterdir()) != sorted(r['prepared_name'] for r in rows):
            raise ValueError('Frozen Golden directory differs from manifest: ' + split)
        verified = []
        for i, original in enumerate(rows):
            source, path = Path(original['source']), Path(original['input_path'])
            origin_root = licensed / ('train' if split == 'training' else 'val') / SYNSET
            if (not _within(source, origin_root)
                    or path != prepared / original['prepared_name']
                    or Path(original['prepared_name']).name != original['prepared_name']):
                raise ValueError('Changed Golden original/actual path mapping')
            if sha256(source) != original['sha256']:
                raise ValueError('Changed Golden original image: ' + str(source))
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != original['input_sha256']:
                raise ValueError('Changed Golden actual image: ' + str(path))
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.mode != 'RGB':
                    raise ValueError('Frozen actual input is not RGB; no replacement: ' + str(path))
            row = dict(original)
            row.update(source_id=_identity(original), selection_index=i, original_split=split)
            verified.append(row)
        roles[role] = verified
    # Preserve the existing manifest sequence, never sort/resample the prefix.
    roles['discovery50'] = roles['target400'][:DISCOVERY_COUNT]
    target_ids = {_identity(r) for r in roles['target400']}
    val_ids = {_identity(r) for r in roles['validation50']}
    target_hashes = {r[k] for r in roles['target400'] for k in ('sha256', 'input_sha256')}
    val_hashes = {r[k] for r in roles['validation50'] for k in ('sha256', 'input_sha256')}
    if target_ids & val_ids or target_hashes & val_hashes:
        raise ValueError('Frozen Golden training/validation identity overlap')
    for role in ('target400', 'validation50'):
        rows = roles[role]
        if len({_identity(r) for r in rows}) != len(rows) or len({r['input_sha256'] for r in rows}) != len(rows):
            raise ValueError('Duplicate frozen Golden identities: ' + role)
    optional = [config.get(k) is not None for k in ('golden_summary', 'golden_run_manifest')]
    if any(optional):
        if not all(optional):
            raise ValueError('Optional Golden summary and run manifest must be paired')
        summary = _load_checked(config['golden_summary'], config['golden_summary_sha256'])
        run = _load_checked(config['golden_run_manifest'], config['golden_run_manifest_sha256'])
        if (summary['status'] != 'PASS' or str(summary['run_id']) != '28208840'
                or run['git_dirty'] or summary['git_commit'] != run['git_commit']
                or str(run['run_id']) != '28208840'):
            raise ValueError('Golden run provenance mismatch')
        for split, role in (('training', 'target400'), ('validation', 'validation50')):
            if [r['input_sha256'] for r in roles[role]] != [r['sha256'] for r in run['input_files'][split]]:
                raise ValueError('Golden actual order differs from completed run')
    return dataset, roles


def enumerate_candidates(train, note):
    """Return all immediate file IDs; do not filter by target class or extension."""
    folders = []
    with os.scandir(train) as entries:
        for item in entries:
            if re.fullmatch(r'n\d{8}', item.name) and item.is_dir(follow_symlinks=False):
                folders.append(item.name)
            else:
                note('train_root_non_synset_entry', {'path': item.path}, 'excluded from class-folder enumeration')
    if len(folders) != SYNSET_COUNT:
        raise ValueError(f'Expected full ImageNet train with {SYNSET_COUNT} synset directories; got {len(folders)}')
    ids, counts = [], {}
    for synset in sorted(folders):
        names = []
        with os.scandir(train / synset) as entries:
            for item in entries:
                if item.is_file():
                    if '\n' in item.name or '\r' in item.name:
                        raise ValueError('Cannot encode newline-containing candidate ID')
                    names.append(item.name)
                else:
                    note('non_file_train_entry', {'path': item.path}, 'not an immediate image candidate')
        counts[synset] = len(names)
        if not names:
            raise ValueError('Empty ImageNet training class: ' + synset)
        ids.extend(synset + '/' + name for name in sorted(names))
    return ids, counts


def candidate_order(count):
    order = list(range(count))
    random.Random(43).shuffle(order)  # isolated; no global Python/NumPy/Torch stream
    return order


def write_candidate_ids(path, ids, indices):
    # Deterministic gzip header; inventories stay well below the collector cap.
    with Path(path).open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as f:
        for index in indices:
            f.write((ids[index]+'\n').encode('utf-8'))


def inspect_candidate(path, candidate_id, train, validation, candidate_rank):
    row = dict(source_id=candidate_id, source=str(path), candidate_rank=candidate_rank,
               synset=candidate_id.split('/')[0], warnings=[], reasons=[])
    name = path.name
    if name.startswith('ILSVRC2012_val_') or candidate_id in validation['ids'] or name in validation['basenames']:
        row['reasons'].append('validation_image_identity')
        return row
    try:
        if not _within(path, train):
            row['reasons'].append('resolved_path_outside_training_root')
            return row
        data = path.read_bytes()
        hashed = hashlib.sha256(data).hexdigest()
        row.update(sha256=hashed, bytes=len(data), resolved_source=str(path.resolve()))
        if hashed in validation['hashes']:
            row['reasons'].append('known_validation_content')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                with Image.open(io.BytesIO(data)) as im:
                    row.update(mode=im.mode, format=im.format, size=list(im.size))
                    if im.mode != 'RGB':
                        row['reasons'].append('unsupported_mode:' + im.mode)
                    im.load()  # header success alone is insufficient
            finally:
                row['warnings'] = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
    except Exception as exc:
        row['reasons'].append('read_or_decode_error:' + type(exc).__name__)
        row['error'] = str(exc)
    return row


def overlap_records(roles):
    """Keep duplicate-content evidence; do not change an approved random selection."""
    result = dict(intra_role={}, cross_role={})
    def groups(rows, key):
        index = defaultdict(list)
        for i, r in enumerate(rows):
            index[_identity(r) if key == 'id' else r[key]].append(i)
        return dict(index)
    for name, rows in roles.items():
        result['intra_role'][name] = {
            key: [{'identity': h, 'indices': ids} for h, ids in groups(rows, key).items() if len(ids) > 1]
            for key in ('id', 'sha256', 'input_sha256')}
    names = list(roles)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            matched = {}
            for key in ('id', 'sha256', 'input_sha256'):
                a, b = groups(roles[left], key), groups(roles[right], key)
                matched[key] = [dict(identity=h, left_indices=a[h], right_indices=b[h]) for h in sorted(a.keys() & b.keys())]
            result['cross_role'][left + '__' + right] = matched
    return result


def run(config, output):
    commit = _assert_runtime(config)
    started = time.monotonic()
    licensed = Path(config['licensed_root']).resolve(strict=True)
    out = Path(output).resolve()
    if out.is_relative_to(licensed) or licensed.is_relative_to(out):
        raise ValueError('Output overlaps licensed source tree')
    # Reject overlap before creating anything in the original private input tree.
    frozen = _load_checked(config['dataset_manifest'], config['dataset_manifest_sha256'])
    private = Path(frozen['source_dir']).resolve()
    if out.is_relative_to(private) or private.is_relative_to(out):
        raise ValueError('Output overlaps immutable Golden inputs')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('ace_*')):
        raise FileExistsError('ACE input output already used; no resume or replacement')
    owned = out / 'ace_inputs'
    owned.mkdir()  # own subdirectory isolates immutable manifest from parent logs
    manifest_path = out / 'ace_inputs_manifest.json'
    record = dict(schema='ace-r-inputs-v1', status='RUNNING', job_id=os.environ['SLURM_JOB_ID'],
                  execution_commit=commit, implementation_sha256=sha256(Path(__file__)),
                  started_at_utc=utc_now(), config=config, roles={},
                  deferred_roles=dict(control='DEFERRED_EFFECTIVE_FEATURE_POOL', gradient='DEFERRED_EFFECTIVE_FEATURE_POOL'),
                  versions=dict(python=platform.python_version(), pillow=PILLOW_VERSION),
                  selection=dict(version=ORDER_VERSION, seed=43, rng='independent random.Random(43).shuffle(indices)',
                                 candidate_universe='all immediate regular-file targets in all 1000 n######## train directories; no extension filter; symlink targets must resolve within train',
                                 label_policy='label-blind; Golden allowed', eligibility='successful RGB decode; no conversion',
                                 duplicate_content_policy='retain different training IDs and report overlap',
                                 validation_exclusion='all val path/filename identities; SHA256 exclusion against original and actual Golden validation50 only'))
    atomic_json(manifest_path, record)
    def progress(stage, **data):
        atomic_json(owned / 'progress.json', dict(status=record['status'], stage=stage,
                    elapsed_seconds=time.monotonic()-started, **data))
    def note(code, evidence, impact):
        with (owned / 'anomalies.jsonl').open('a') as f:
            f.write(json.dumps(dict(code=code, evidence=evidence, impact=impact, status='RECORDED'))+'\n')
    try:
        if config.get('approval_path'):
            approval = _load_checked(config['approval_path'], config['approval_sha256'])
            if approval.get('status') != 'APPROVED' or approval.get('arm') != 'R':
                raise ValueError('Expected approved R protocol receipt')
            if approval['budgets']['inputs'] != dict(cpus=4, mem='8G', time='00:15:00'):
                raise ValueError('Input budget differs from approved 4CPU/8GiB/15min')
            atomic_json(owned/'protocol_approval.json', approval)
            record['approval_sha256'] = config['approval_sha256']
        progress('verify_golden')
        dataset, roles = verify_golden(config, licensed)
        atomic_json(owned / 'golden_dataset_manifest.json', dataset)
        validation = dict(ids={_identity(r) for r in roles['validation50']},
                          basenames={Path(r['source']).name for r in roles['validation50']},
                          hashes={r[k] for r in roles['validation50'] for k in ('sha256', 'input_sha256')})
        progress('enumerate_training')
        ids, counts = enumerate_candidates(licensed / 'train', note)
        order = candidate_order(len(ids))
        for filename, indices in (('candidate_ids.sorted.txt.gz', range(len(ids))), ('candidate_ids.seed43.txt.gz', order)):
            write_candidate_ids(owned / filename, ids, indices)
        record['candidates'] = dict(total=len(ids), per_synset=counts,
            sorted_path='ace_inputs/candidate_ids.sorted.txt.gz', sorted_sha256=sha256(owned/'candidate_ids.sorted.txt.gz'),
            order_path='ace_inputs/candidate_ids.seed43.txt.gz', order_sha256=sha256(owned/'candidate_ids.seed43.txt.gz'),
            identity_format='gzip(mtime=0,empty filename); synset/basename, UTF-8 LF, final LF; candidate_rank is zero-based in seed43 order',
            unexamined_tail='IDs inventoried but not decoded or content-hashed')
        atomic_json(manifest_path, record)
        selected, skipped, examined = [], Counter(), 0
        with (owned/'selection_attempts.jsonl').open('x') as attempts, (owned/'skipped.jsonl').open('x') as skipfile:
            for rank, index in enumerate(order):
                entry = inspect_candidate(licensed/'train'/ids[index], ids[index], licensed/'train', validation, rank)
                examined += 1
                if entry['warnings']:
                    note('candidate_image_warning', entry, 'decode succeeded or failed as recorded; warning alone does not change selection')
                if entry['reasons']:
                    entry['decision'] = 'skipped'
                    skipped.update(entry['reasons'])
                    skipfile.write(json.dumps(entry)+'\n'); skipfile.flush()
                else:
                    entry.update(decision='selected', selection_index=len(selected), input_path=entry['source'],
                                 input_sha256=entry['sha256'], input_mode='RGB', original_split='training',
                                 prepared_name=f'{len(selected)+1:04d}_'+Path(entry['source']).name)
                    selected.append(entry)
                attempts.write(json.dumps(entry)+'\n'); attempts.flush()
                if examined % 100 == 0:
                    progress('select_random2000', examined=examined, selected=len(selected), skipped=examined-len(selected))
                if len(selected) == RANDOM_COUNT:
                    break
        roles['random2000'] = selected
        for name, rows in roles.items():
            path = owned/(name+'.json'); atomic_json(path, rows)
            record['roles'][name] = dict(path=str(path.relative_to(out)), sha256=sha256(path), count=len(rows))
        overlaps = overlap_records(roles)
        atomic_json(owned/'role_overlaps.json', overlaps)
        if overlaps['intra_role']['random2000']['sha256']:
            note('random_duplicate_content_retained', overlaps['intra_role']['random2000'], 'Different candidate IDs preserved; downstream must retain identity/order and disclose dependence')
        record['statistics'] = dict(candidates=len(ids), examined=examined, selected=len(selected),
                                    skipped=examined-len(selected), skip_reason_counts=dict(skipped),
                                    unexamined=len(ids)-examined)
        if len(selected) != RANDOM_COUNT:
            raise ValueError(f'Only {len(selected)} eligible random images; expected {RANDOM_COUNT}; no budget expansion')
        record['status'] = 'PASS'
        return dict(status='PASS', manifest_path=str(manifest_path), roles=record['roles'],
                    statistics=record['statistics'], deferred_roles=record['deferred_roles'])
    except BaseException as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        note('input_preparation_failed', record['traceback'], 'Incomplete inputs; no automatic retry or substitution')
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        progress('finished')
        record['files'] = {str(p.relative_to(out)):dict(bytes=p.stat().st_size, sha256=sha256(p))
                           for p in sorted(owned.rglob('*')) if p.is_file()}
        atomic_json(manifest_path, record)
