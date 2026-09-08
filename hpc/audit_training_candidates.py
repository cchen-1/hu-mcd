"""Audit all training candidates, refill a frozen seeded selection, preserve validation."""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def inspect_image(path):
    from PIL import Image, ImageChops
    record = {'source': str(path.resolve(strict=True)), 'sha256': digest(path),
              'bytes': path.stat().st_size, 'mode': None, 'size': None,
              'grayscale': False, 'grayscale_kind': None, 'image_error': None}
    try:
        with Image.open(path) as image:
            image.load()
            record.update(mode=image.mode, size=list(image.size))
            if image.mode in ('1', 'L', 'LA', 'I', 'F') or image.mode.startswith('I;16'):
                record.update(grayscale=True, grayscale_kind='grayscale_mode')
            elif image.mode == 'RGB':
                r, g, b = image.split()
                if ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(r, b).getbbox() is None:
                    record.update(grayscale=True, grayscale_kind='exact_achromatic_RGB')
            if image.mode != 'RGB':
                record['image_error'] = 'unsupported_mode:' + image.mode
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        record['image_error'] = 'decode_error:' + str(exc)
    return record


def contact_sheets(entries, output, prefix):
    from PIL import Image, ImageDraw
    names = []
    for start in range(0, len(entries), 16):
        page = entries[start:start + 16]
        canvas = Image.new('RGB', (1000, ((len(page) + 3) // 4) * 210), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, entry in enumerate(page):
            x, y = (i % 4) * 250, (i // 4) * 210
            with Image.open(entry['source']) as im:
                im.thumbnail((244, 175))
                canvas.paste(im, (x + (250 - im.width) // 2, y))
            draw.text((x + 3, y + 177), Path(entry['source']).name, fill='black')
            draw.text((x + 3, y + 192), str(entry['mode']), fill='black')
        name = f'{prefix}-{start // 16 + 1:02d}.jpg'
        canvas.save(output / name, quality=88)
        names.append(name)
    return names


def audit(dataset_root, frozen_path, frozen_sha256, output, seed=43,
          train_count=400, val_count=50, previews=True):
    root, output = Path(dataset_root).resolve(strict=True), Path(output)
    if seed != 43:
        raise ValueError('Only the user-approved seed 43 is allowed')
    if digest(frozen_path) != frozen_sha256:
        raise ValueError('Frozen selection identity changed')
    frozen = json.loads(Path(frozen_path).read_text())
    if frozen['seed'] != seed or frozen['licensed_source_root'] != str(root) or frozen['synset'] != 'n02099601':
        raise ValueError('Frozen selection dataset/seed mismatch')
    paths = sorted(p for p in (root / 'train/n02099601').iterdir()
                   if p.is_file() and p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    names_sha = hashlib.sha256('\n'.join(p.name for p in paths).encode()).hexdigest()
    if names_sha != frozen['candidate_filename_hashes']['train']:
        raise ValueError('Training candidate population changed')
    # With the actual 1300/400 counts both calls use the same pool algorithm.
    # Check that property against the immutable historical manifest, never assume it.
    order = random.Random(seed).sample(paths, len(paths))
    if [str(p.resolve()) for p in order[:train_count]] != [e['source'] for e in frozen['training']]:
        raise ValueError('Full candidate order does not preserve the frozen prefix')
    if len(frozen['validation']) != val_count:
        raise ValueError('Wrong frozen validation count')
    records = [inspect_image(p) for p in order]
    by_source = {e['source']: e for e in records}
    for old in frozen['training']:
        if by_source[old['source']]['sha256'] != old['sha256']:
            raise ValueError('Previously selected training content changed: ' + old['source'])
    validation = []
    for old in frozen['validation']:
        source = Path(old['source'])
        if source.resolve().parent != root / 'val/n02099601':
            raise ValueError('Frozen validation path is outside the verified split')
        entry = inspect_image(source)
        if entry['sha256'] != old['sha256']:
            raise ValueError('Frozen validation content changed: ' + old['source'])
        entry['prepared_name'] = old['prepared_name']
        validation.append(entry)
    if len({e['source'] for e in validation}) != val_count or len({e['sha256'] for e in validation}) != val_count:
        raise ValueError('Duplicate frozen validation content/path')
    seen_paths = {e['source'] for e in validation}
    seen_hashes = {e['sha256'] for e in validation}
    selected, skipped = [], []
    for rank, entry in enumerate(records, 1):
        entry['candidate_rank'] = rank
        entry['decision'] = 'not_needed'
        if len(selected) == train_count:
            continue
        reasons = []
        if entry['grayscale']:
            reasons.append('grayscale:' + entry['grayscale_kind'])
        if entry['image_error']:
            reasons.append(entry['image_error'])
        if entry['source'] in seen_paths or entry['sha256'] in seen_hashes:
            reasons.append('duplicate_selected_or_validation_content_or_path')
        if reasons:
            entry.update(decision='skipped', skip_reasons=reasons)
            skipped.append(dict(entry))
            continue
        entry.update(decision='selected', prepared_name=f"{len(selected)+1:04d}_{Path(entry['source']).name}")
        selected.append(dict(entry))
        seen_paths.add(entry['source'])
        seen_hashes.add(entry['sha256'])
    gray = [e for e in records if e['grayscale']]
    issues = [e for e in validation if e['image_error']]
    stats = {'training_candidates': len(records), 'grayscale_count': len(gray),
             'grayscale_fraction': len(gray) / len(records) if records else 0,
             'grayscale_kinds': dict(collections.Counter(e['grayscale_kind'] for e in gray)),
             'modes': dict(collections.Counter(e['mode'] for e in records)),
             'other_invalid_count': sum(bool(e['image_error']) and not e['grayscale'] for e in records),
             'selected_count': len(selected), 'examined_for_selection': len(selected) + len(skipped),
             'skipped_count': len(skipped),
             'skipped_grayscale_count': sum(e['grayscale'] for e in skipped),
             'validation_count': len(validation), 'validation_format_issue_count': len(issues)}
    manifest = {'schema_version': 2, 'dataset': 'ImageNet1k', 'synset': 'n02099601',
                'class_name': 'golden_retriever', 'seed': seed, 'python': sys.version,
                'licensed_source_root': str(root), 'audit_job_id': os.environ.get('SLURM_JOB_ID'),
                'frozen_selection': str(frozen_path), 'frozen_selection_sha256': frozen_sha256,
                'sampling': 'random.Random(43).sample(sorted candidates, full population); first 400 eligible; frozen prefix checked',
                'grayscale_definition': 'PIL grayscale mode or RGB with exactly equal channels for every pixel; no low-saturation heuristic',
                'candidate_filename_sha256': names_sha,
                'candidate_order_sha256': hashlib.sha256('\n'.join(e['source'] for e in records).encode()).hexdigest(),
                'statistics': stats, 'training': selected, 'validation': validation,
                'skipped': skipped, 'candidates': records, 'validation_issues': issues,
                'ready_for_reference': len(selected) == train_count and not issues,
                'overlap_check': {'duplicate_paths': 0, 'duplicate_contents': 0},
                'scene_review': 'PENDING; contact sheets are previews only, not converted model inputs'}
    manifest['preview_files'] = []
    if previews:
        manifest['preview_files'] += contact_sheets([e for e in gray if not str(e['image_error']).startswith('decode_error')], output, 'training-grayscale')
        manifest['preview_files'] += contact_sheets(selected[:16], output, 'training-selected-context')
        manifest['preview_files'] += contact_sheets([e for e in issues if not str(e['image_error']).startswith('decode_error')], output, 'validation-format')
    write_json(output / 'training_audit.json', manifest)
    for name, entries in [('training_images', selected), ('validation_images', validation), ('candidate_order', records)]:
        (output / (name + '.txt')).write_text('\n'.join(e['source'] for e in entries) + '\n')
    write_json(output / 'skipped_training.json', skipped)
    write_json(output / 'audit_summary.json', {**stats, 'validation_issues': issues, 'preview_files': manifest['preview_files']})
    if len(selected) != train_count:
        raise ValueError('Insufficient eligible training images; see saved audit')
    return stats


def publish(audit_path, audit_sha256, output):
    if digest(audit_path) != audit_sha256:
        raise ValueError('Reviewed audit identity changed')
    manifest = json.loads(Path(audit_path).read_text())
    output = Path(output)
    root = Path(manifest['licensed_source_root']).resolve(strict=True)
    if not output.is_absolute() or output.exists() or root == output.resolve() or root in output.resolve().parents:
        raise ValueError('Require a new private output directory outside the licensed dataset')
    if manifest['seed'] != 43 or manifest['synset'] != 'n02099601' or len(manifest['training']) != 400 or len(manifest['validation']) != 50:
        raise ValueError('Unexpected reviewed selection')
    seen_paths, seen_hashes = set(), set()
    for split in ('training', 'validation'):
        for e in manifest[split]:
            source = Path(e['source']).resolve(strict=True)
            expected_parent = root / ('train' if split == 'training' else 'val') / 'n02099601'
            if source.parent != expected_parent or Path(e['prepared_name']).name != e['prepared_name']:
                raise ValueError('Invalid input path/name')
            current = inspect_image(source)
            if any(current[k] != e[k] for k in ('sha256', 'mode', 'grayscale', 'image_error')):
                raise ValueError('Audited input changed: ' + str(source))
            if split == 'training' and (current['grayscale'] or current['image_error']):
                raise ValueError('Ineligible training image')
            if str(source) in seen_paths or current['sha256'] in seen_hashes:
                raise ValueError('Duplicate selected/validation path or content')
            seen_paths.add(str(source))
            seen_hashes.add(current['sha256'])
    manifest.update(source_dir=str(output), selection_audit=str(audit_path), selection_audit_sha256=audit_sha256,
                    preparation_job_id=os.environ.get('SLURM_JOB_ID'),
                    scene_review='Reviewed locally before publishing this exact audit; see accompanying research notes')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.filtered-golden-', dir=output.parent) as td:
        staging = Path(td) / 'split'
        for split, folder in [('training', staging / 'golden_retriever'),
                              ('validation', staging / 'val_imgs/golden_retriever_val')]:
            folder.mkdir(parents=True)
            for e in manifest[split]:
                (folder / e['prepared_name']).symlink_to(e['source'])
            (staging / (split + '_images.txt')).write_text('\n'.join(e['source'] for e in manifest[split]) + '\n')
        write_json(staging / 'dataset_manifest.json', manifest)
        with (output.parent / (output.name + '.publish.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if output.exists():
                raise FileExistsError(output)
            os.rename(staging, output)
    return {'source_dir': str(output), 'manifest_sha256': digest(output / 'dataset_manifest.json'),
            'training_count': 400, 'validation_count': 50, 'ready_for_reference': manifest['ready_for_reference'],
            'validation_issues': manifest['validation_issues']}
