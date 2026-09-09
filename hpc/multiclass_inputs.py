"""Audit frozen class selections; publish only explicitly approved input adjustments.

Optional per-class publish config::

    validation_compatibility_allowlist = [
        {"source_basename": "ILSVRC2012_val_XXXXXXXX.JPEG", "source_sha256": "<64 hex>"}
    ]

A nonempty list must exactly cover the audited L-only validation issues, with the
approved class counts beach_wagon=1, zebra=5, ox=2. No training conversions. Actual
PNG names append '.rgb.png' to the original prepared name; original_prepared_name,
source/source SHA, original lists, and the explicit input_mapping.json retain
identity. All reads/conversions/publication run inside the caller's Slurm worker.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from hpc.audit_training_candidates import inspect_image, contact_sheets
from utils.run_tracking import atomic_json, sha256


def audit(config, output):
    output.mkdir(parents=True, exist_ok=True)
    root = Path(config['dataset_root']).resolve(strict=True)
    mapping = json.loads(Path('imagenet1k_class_info.json').read_text())
    records = []
    for name in config['classes']:
        folder = output / name
        folder.mkdir(exist_ok=False)
        record = {'class_name': name, 'synset': mapping[name]['wnid'], 'class_index': mapping[name]['class_index'],
                  'seed': config['seed'], 'status': 'RUNNING', 'issues': [], 'training': [], 'validation': []}
        try:
            paths = {}
            for split in ('train', 'val'):
                directory = root / split / record['synset']
                paths[split] = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
            if len(paths['train']) < 400 or len(paths['val']) != 50:
                raise ValueError('Expected at least 400 training candidates and exactly 50 validation IDs')
            order = random.Random(config['seed']).sample(paths['train'], len(paths['train']))
            candidates = [dict(inspect_image(p), candidate_rank=i) for i, p in enumerate(order, 1)]
            validation = [inspect_image(p) for p in paths['val']]
            selected, skipped = [], []
            seen = {e['sha256'] for e in validation}
            for entry in candidates:
                entry['decision'] = 'not_needed'
                if len(selected) == 400:
                    continue
                if entry['grayscale']:
                    entry.update(decision='skipped', skip_reasons=['grayscale:' + entry['grayscale_kind']])
                    skipped.append(dict(entry))
                elif entry['image_error'] or entry['sha256'] in seen:
                    record['issues'].append({'kind': 'training_input_invalid_or_duplicate', 'entry': dict(entry)})
                    entry['decision'] = 'blocked'
                    break
                else:
                    entry.update(decision='selected', prepared_name=f'{len(selected)+1:04d}_{Path(entry["source"]).name}')
                    selected.append(dict(entry)); seen.add(entry['sha256'])
            for i, entry in enumerate(validation, 1):
                entry['prepared_name'] = f'{i:04d}_{Path(entry["source"]).name}'
                if entry['image_error']:
                    record['issues'].append({'kind': 'validation_format', 'entry': dict(entry)})
            if len({e['sha256'] for e in validation}) != 50:
                record['issues'].append({'kind': 'validation_duplicate_content'})
            gray = [e for e in candidates if e['grayscale']]
            record.update(licensed_source_root=str(root), training=selected, validation=validation, candidates=candidates, skipped=skipped,
                          sampling='random.Random(43).sample(sorted training candidates, full population); first 400 non-grayscale RGB; all sorted 50 validation IDs',
                          candidate_filename_sha256=__import__('hashlib').sha256('\n'.join(p.name for p in paths['train']).encode()).hexdigest(),
                          candidate_order_sha256=__import__('hashlib').sha256('\n'.join(e['source'] for e in candidates).encode()).hexdigest(),
                          statistics={'training_candidates': len(candidates), 'grayscale_count': len(gray), 'grayscale_fraction': len(gray)/len(candidates),
                                      'skipped_grayscale_count': len(skipped), 'training_count': len(selected), 'validation_count': len(validation),
                                      'validation_format_issues': sum(bool(e['image_error']) for e in validation)},
                          scene_review='PENDING' if gray else 'NOT_NEEDED', audit_job_id=os.environ.get('SLURM_JOB_ID'))
            preview = [e for e in gray if not str(e['image_error']).startswith('decode_error')]
            record['preview_files'] = contact_sheets(preview, folder, 'training-grayscale') if preview else []
            record['status'] = 'INPUT_REVIEW_REQUIRED' if record['issues'] or len(selected) != 400 or gray else 'READY_TO_PUBLISH'
        except Exception as exc:
            record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        atomic_json(folder/'selection_audit.json', record)
        for split in ('training', 'validation'):
            (folder/(split+'_images.txt')).write_text('\n'.join(e['source'] for e in record[split])+'\n')
        records.append({'class_name': name, 'status': record['status'], 'statistics': record.get('statistics'), 'issues': record['issues'],
                        'audit_path': str(folder/'selection_audit.json'), 'audit_sha256': sha256(folder/'selection_audit.json'),
                        'preview_files': record.get('preview_files', []), 'error': record.get('error')})
        atomic_json(output/'input_audit_progress.json', {'status':'RUNNING','classes':records})
    result={'status':'AUDITED', 'classes':records, 'selection_policy':'same seed43 training eligibility as Golden; no publication, conversion or experiment'}
    atomic_json(output/'input_audit_progress.json',result)
    return result


APPROVED_VALIDATION_L_COUNTS = {'beach_wagon': 1, 'zebra': 5, 'ox': 2}


def validation_compatibility(record, config):
    """Resolve a strict per-class basename/SHA allowlist against pinned audit rows."""
    allowlist = config.get('validation_compatibility_allowlist', [])
    if not isinstance(allowlist, list):
        raise ValueError('Validation compatibility allowlist must be a list')
    allowed = {}
    for item in allowlist:
        if not isinstance(item, dict) or set(item) != {'source_basename', 'source_sha256'}:
            raise ValueError('Allowlist entries require exactly source_basename and source_sha256')
        name, digest = item['source_basename'], item['source_sha256']
        if (not isinstance(name, str) or not name or Path(name).name != name
                or name in ('.', '..') or '\\' in name or name in allowed
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Invalid or duplicate validation allowlist identity')
        allowed[name] = digest
    if allowed and len(allowed) != APPROVED_VALIDATION_L_COUNTS.get(record['class_name']):
        raise ValueError('Allowlist differs from explicitly approved class/count')
    validation = {Path(e['source']).name: e for e in record['validation']}
    if len(validation) != len(record['validation']):
        raise ValueError('Duplicate validation source IDs')
    selected = {}
    for name, digest in allowed.items():
        entry = validation.get(name)
        if (entry is None or entry['sha256'] != digest or entry['mode'] != 'L'
                or entry['image_error'] != 'unsupported_mode:L'):
            raise ValueError('Allowlist source hash/mode/ID differs from audited L input')
        selected[entry['source']] = entry
    seen = set()
    for issue in record['issues']:
        entry = issue.get('entry', {})
        source = entry.get('source')
        if (issue.get('kind') != 'validation_format' or source not in selected
                or entry != selected[source] or source in seen):
            raise ValueError('Unlisted or unexpected audited input issue')
        seen.add(source)
    anomalous = {e['source'] for e in record['validation'] if e['image_error']}
    if seen != set(selected) or anomalous != set(selected):
        raise ValueError('Allowlist must exactly cover all audited validation issues')
    return selected


def lossless_l_to_rgb(entry, destination):
    """Convert a hash-verified byte snapshot and verify decoded output channel bytes."""
    from PIL import Image
    original = Path(entry['source']).read_bytes()
    if hashlib.sha256(original).hexdigest() != entry['sha256']:
        raise ValueError('Changed approved L source hash before conversion')
    with Image.open(io.BytesIO(original)) as image:
        image.load()
        if image.mode != 'L' or list(image.size) != entry['size']:
            raise ValueError('Approved conversion requires unchanged L mode and dimensions')
        pixels, size = image.tobytes(), image.size
        image.convert('RGB').save(destination, format='PNG')
    with Image.open(destination) as converted:
        converted.load()
        if (converted.format != 'PNG' or converted.mode != 'RGB' or converted.size != size
                or any(channel.tobytes() != pixels for channel in converted.split())):
            raise ValueError('Lossless RGB compatibility pixel verification failed')
    return {'kind': 'L_to_RGB_PNG',
            'reason': 'User-approved validation input compatibility: classifier requires RGB; replicate original L pixels without resampling',
            'source_mode': 'L', 'input_mode': 'RGB', 'source_sha256': entry['sha256'],
            'source_size': list(size), 'input_size': list(size),
            'pixel_equality_verified': True, 'pixel_rule': 'R == G == B == original decoded L at every pixel'}


def publish(config, output):
    """Atomically publish fixed lists, permitting only the exact approved L adjustments."""
    source = Path(config['selection_audit'])
    if sha256(source) != config['selection_audit_sha256']:
        raise ValueError('Selection audit identity mismatch')
    record = json.loads(source.read_text())
    if record['class_name'] != config['class_name'] or len(record['training']) != 400 or len(record['validation']) != 50:
        raise ValueError('Unresolved input issue; refuse publication')
    conversions = validation_compatibility(record, config)
    if record['scene_review'] == 'PENDING' and not config.get('scene_review_evidence'):
        raise ValueError('Grayscale selection requires documented scene review')
    dest = Path(config['prepared_root']); root = Path(record['licensed_source_root'])
    if not dest.is_absolute() or dest.exists() or dest.resolve().is_relative_to(root.resolve()):
        raise ValueError('Require a new private input directory')
    allhash = set()
    for split in ('training', 'validation'):
        names = set()
        for entry in record[split]:
            prepared = entry['prepared_name']
            if not prepared or Path(prepared).name != prepared or prepared in ('.', '..'):
                raise ValueError('Invalid audited prepared name')
            actual = prepared + '.rgb.png' if split == 'validation' and entry['source'] in conversions else prepared
            if actual in names:
                raise ValueError('Colliding actual input names')
            names.add(actual)
            found = inspect_image(Path(entry['source']))
            compatible = split == 'validation' and entry['source'] in conversions
            expected_error = 'unsupported_mode:L' if compatible else None
            if (found['sha256'] != entry['sha256'] or found['image_error'] != expected_error
                    or found['mode'] != ('L' if compatible else 'RGB')
                    or found['size'] != entry['size'] or found['sha256'] in allhash):
                raise ValueError('Changed, unsupported or duplicate frozen input')
            allhash.add(found['sha256'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=dest.parent) as td:
        stage = Path(td) / 'data'; stage.mkdir()
        manifest = {k: record[k] for k in ('class_name', 'synset', 'seed', 'licensed_source_root', 'sampling',
                    'candidate_filename_sha256', 'candidate_order_sha256', 'statistics', 'training', 'validation', 'skipped')}
        manifest.update(schema_version=5 if conversions else 4, source_dir=str(dest), dataset='ImageNet1k',
                        ready_for_reference=True, validation_issues=[], audit_path=str(source), audit_sha256=sha256(source),
                        scene_review_evidence=config.get('scene_review_evidence'), preparation_job_id=os.environ['SLURM_JOB_ID'])
        if conversions:
            # Keep resolved issues verbatim; an empty active issue list does not erase history.
            manifest.update(validation_compatibility_allowlist=config['validation_compatibility_allowlist'],
                            resolved_validation_issues=record['issues'], validation_compatibility_count=len(conversions))
            (stage/'validation_original_prepared_names.txt').write_text(
                '\n'.join(e['prepared_name'] for e in record['validation'])+'\n')
        mapping = []
        for split, rel in [('training', record['class_name']), ('validation', 'val_imgs/'+record['class_name']+'_val')]:
            (stage/rel).mkdir(parents=True)
            for entry in manifest[split]:
                original_name = entry['prepared_name']
                adjustment = None
                if split == 'validation' and entry['source'] in conversions:
                    actual_name = original_name + '.rgb.png'
                    adjustment = lossless_l_to_rgb(entry, stage/rel/actual_name)
                    entry.update(original_prepared_name=original_name, prepared_name=actual_name,
                                 input_size=adjustment['input_size'])
                    input_hash = sha256(stage/rel/actual_name)
                else:
                    actual_name = original_name
                    (stage/rel/actual_name).symlink_to(entry['source'])
                    input_hash = entry['sha256']
                entry.update(input_path=str(dest/rel/actual_name), input_sha256=input_hash,
                             input_mode='RGB', compatibility_adjustment=adjustment)
                mapping.append(dict(split=split, source=entry['source'], source_basename=Path(entry['source']).name,
                    source_sha256=entry['sha256'], original_prepared_name=original_name,
                    prepared_name=actual_name, input_path=entry['input_path'], input_sha256=input_hash,
                    source_size=entry['size'], input_size=entry['size'], adjustment=adjustment))
            (stage/(split+'_images.txt')).write_text('\n'.join(e['source'] for e in manifest[split])+'\n')
            (stage/(split+'_actual_inputs.txt')).write_text('\n'.join(e['input_path'] for e in manifest[split])+'\n')
        if conversions:
            atomic_json(stage/'input_mapping.json', mapping)
            manifest['input_mapping_sha256'] = sha256(stage/'input_mapping.json')
        atomic_json(stage/'dataset_manifest.json', manifest)
        import fcntl
        with (dest.parent/(dest.name+'.publish.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dest.exists():
                raise FileExistsError(dest)
            os.rename(stage, dest)
    return {'status': 'PASS', 'dataset_manifest': str(dest/'dataset_manifest.json'),
            'dataset_manifest_sha256': sha256(dest/'dataset_manifest.json')}
