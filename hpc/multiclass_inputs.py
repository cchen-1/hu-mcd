"""Audit new class candidates in Slurm; never change a frozen class or auto-convert inputs."""
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


def publish(config, output):
    """Publish only a hash-bound audited list explicitly cleared by the coordinator."""
    source = Path(config['selection_audit'])
    if sha256(source) != config['selection_audit_sha256']:
        raise ValueError('Selection audit identity mismatch')
    record=json.loads(source.read_text())
    if record['class_name']!=config['class_name'] or record['issues'] or len(record['training'])!=400 or len(record['validation'])!=50:
        raise ValueError('Unresolved input issue; refuse publication')
    if record['scene_review']=='PENDING' and not config.get('scene_review_evidence'):
        raise ValueError('Grayscale selection requires documented scene review')
    dest=Path(config['prepared_root']); root=Path(record['licensed_source_root'])
    if not dest.is_absolute() or dest.exists() or dest.resolve().is_relative_to(root.resolve()):
        raise ValueError('Require a new private input directory')
    allhash=set()
    for split in ('training','validation'):
        for e in record[split]:
            found=inspect_image(Path(e['source']))
            if found['sha256']!=e['sha256'] or found['image_error'] or found['sha256'] in allhash:
                raise ValueError('Changed, unsupported or duplicate frozen input')
            allhash.add(found['sha256'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.publish-',dir=dest.parent) as td:
        stage=Path(td)/'data';stage.mkdir()
        manifest={k:record[k] for k in ('class_name','synset','seed','licensed_source_root','sampling','candidate_filename_sha256','candidate_order_sha256','statistics','training','validation','skipped')}
        manifest.update(schema_version=4,source_dir=str(dest),dataset='ImageNet1k',ready_for_reference=True,validation_issues=[],
                        audit_path=str(source),audit_sha256=sha256(source),scene_review_evidence=config.get('scene_review_evidence'),preparation_job_id=os.environ['SLURM_JOB_ID'])
        for split,rel in [('training',record['class_name']),('validation','val_imgs/'+record['class_name']+'_val')]:
            (stage/rel).mkdir(parents=True)
            for e in manifest[split]:
                (stage/rel/e['prepared_name']).symlink_to(e['source'])
                e.update(input_path=str(dest/rel/e['prepared_name']),input_sha256=e['sha256'],input_mode='RGB',compatibility_adjustment=None)
            (stage/(split+'_images.txt')).write_text('\n'.join(e['source'] for e in manifest[split])+'\n')
            (stage/(split+'_actual_inputs.txt')).write_text('\n'.join(e['input_path'] for e in manifest[split])+'\n')
        atomic_json(stage/'dataset_manifest.json',manifest)
        import fcntl
        with (dest.parent/(dest.name+'.publish.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if dest.exists():raise FileExistsError(dest)
            os.rename(stage,dest)
    return {'status':'PASS','dataset_manifest':str(dest/'dataset_manifest.json'),'dataset_manifest_sha256':sha256(dest/'dataset_manifest.json')}
