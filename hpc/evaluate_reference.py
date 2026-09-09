"""Cache-only setup and faithful single-class HU-MCD/Random evaluation.

Called by the shared Slurm worker as run(config, output); no scheduler/network calls.
Required config keys: source_run_dir, source_cache_dir, source_audit_path,
source_audit_sha256, source_files_sha256 (all SOURCE_FILES), dataset_manifest,
dataset_manifest_sha256, resnet_checkpoint, resnet_checkpoint_sha256, device,
torch_num_threads, precision, batch_size=8, random_seeds={sdc:43, ssc:43}.
The shared worker owns release/environment/input/weight attestation. This module
also verifies the immutable source chain and every cache/input it consumes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import warnings
import numpy as np

SOURCE_FILES = (
    'summary.json', 'run_manifest.json', 'resolved_config.json',
    'scientific/discovery.json', 'scientific/discovery.npz',
    'scientific/validation.npz', 'scientific/validation_segments.json',
)
SOURCE_JOB = '28208840'
SOURCE_COMMIT = 'eeb27e9fb31122edbfda1109c69ab7e64b094941'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, expected: str, size=None) -> Path:
    path = Path(path)
    if len(expected) != 64 or (size is not None and path.stat().st_size != size):
        raise ValueError(f'Invalid hash or size: {path}')
    if sha256(path) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return path


def under(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes source root: {relative}')
    return path


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def random_code_identity(source):
    """Use actual immutable evaluation code; old run manifests omit masking files.

    The shared Slurm worker verifies installed timm masking files against this
    release. Ten-class source audits separately bind them to the discovery code.
    """
    repo = Path(__file__).resolve().parents[1]
    result = dict(source['manifest']['source_sha256'])
    for relative in ('input_masking/resnet.py', 'input_masking/sal_layers.py'):
        actual = sha256(repo / relative)
        audited = source['audit'].get('core_sha256', {}).get(relative)
        if source['audit_schema'] == 'ten-class-visual-v1' and audited != actual:
            raise ValueError('Audited input-mask code differs from evaluation release')
        result[relative] = actual
    return result


def random_signature(config, model_default_cfg, rows, source_sha256):
    """Identity required for reusing both original Random trajectories; no paths-as-identity."""
    identity = dict(schema='humcd-random-v1',
        inputs=[dict(source_id=Path(r['source']).name, sha256=r['input_sha256']) for r in rows],
        classifier_sha256=config['resnet_checkpoint_sha256'],
        model_name=config['model_name'], model_default_cfg=model_default_cfg,
        max_shortest_side=config['max_shortest_side'], precision=config['precision'],
        prediction=dict(batch_size=config['batch_size'], grouping='global image then state; final remainder',
                        cropping_mode=0, use_masks=True, masking_mode=1, erosion_threshold=1.0),
        perturbation=dict(grid_size=[10,10], cell_rule='floor division; uncovered borders preserved',
                          directions=['sdc','ssc'], seeds=config['random_seeds'],
                          python_random_version=random.Random.VERSION),
        implementation={k:source_sha256[k] for k in ('benchmark_methods.py', 'classes.py',
             'utils/utils_general.py', 'input_masking/resnet.py', 'input_masking/sal_layers.py')})
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), identity=identity)


def verify_sources(config: dict) -> dict:
    source_job = str(config.get('source_job', SOURCE_JOB))
    source_commit = config.get('source_commit', SOURCE_COMMIT)
    root = Path(config['source_run_dir'])
    hashes = config['source_files_sha256']
    if set(SOURCE_FILES) - hashes.keys():
        raise ValueError('Missing pinned source file hashes')
    for relative in SOURCE_FILES:
        checked_file(under(root, relative), hashes[relative])
    read = lambda name: json.loads((root / name).read_text())
    summary, manifest, resolved = map(read, SOURCE_FILES[:3])
    audit = json.loads(checked_file(Path(config['source_audit_path']),
                                   config['source_audit_sha256']).read_text())
    if audit['status'] != 'PASS' or str(audit['source_job']) != source_job:
        raise ValueError('Source cache audit identity/status mismatch')
    schema = config.get('source_audit_schema', 'golden-completed-run-v1')
    if schema == 'golden-completed-run-v1':
        if source_job != SOURCE_JOB or str(audit['audit_job']) != '28211056':
            raise ValueError('Legacy audit is only valid for the original Golden run')
    elif schema == 'ten-class-visual-v1':
        expected = dict(expected_commit=source_commit, code_commit=source_commit,
                        class_name=config['class_name'],
                        config_sha256=hashes['resolved_config.json'],
                        dataset_sha256=config['dataset_manifest_sha256'],
                        source_cache=config['source_cache_dir'])
        if any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError('Ten-class source audit identity mismatch')
        model = audit['model_identity']
        if (model['resnet_sha256'] != config['resnet_checkpoint_sha256']
                or model['precision'] != config['precision']):
            raise ValueError('Audited model identity mismatch')
        for split, n in [('training', 400), ('validation', 50)]:
            record = audit['splits'][split]
            if (record['images'] != n or record['masks_features_mapping'] != 'PASS'
                    or record['saved_reconstruction_checks']['status'] != 'PASS'):
                raise ValueError('Source split has not passed mapping/reconstruction acceptance')
    else:
        raise ValueError(f'Unknown source audit schema: {schema}')
    if (summary['status'] != 'PASS' or str(summary['run_id']) != source_job
            or str(manifest['run_id']) != source_job
            or manifest['git_commit'] != source_commit
            or summary['git_commit'] != source_commit or manifest['git_dirty']):
        raise ValueError('Source discovery identity/status mismatch')
    for record in (summary, manifest):
        if record['resolved_config_sha256'] != hashes['resolved_config.json']:
            raise ValueError('Source resolved configuration identity mismatch')
    for key in ('dataset_manifest_sha256', 'resnet_checkpoint_sha256', 'precision', 'batch_size'):
        if config[key] != resolved[key]:
            raise ValueError(f'Evaluation changes fixed source setting: {key}')
    if config['batch_size'] != 8 or config['random_seeds'] != {'sdc': 43, 'ssc': 43}:
        raise ValueError('Expected batch 8 and separate random ordering seeds 43')
    if resolved['validation_images'] != 50 or resolved['train_images'] != 400:
        raise ValueError('Unexpected fixed sample sizes')
    if manifest['precision'] != config['precision']:
        raise ValueError('Source runtime precision differs from evaluation')
    # Bind released scientific functions, including the installed input-mask model.
    repo = Path(__file__).resolve().parents[1]
    for relative in ('benchmark_methods.py', 'classes.py', 'concept_explainer.py',
                     'utils/utils_general.py', 'utils/utils_mcd.py'):
        checked_file(repo / relative, manifest['source_sha256'][relative])
    dataset = json.loads(checked_file(Path(config['dataset_manifest']),
                                     config['dataset_manifest_sha256']).read_text())
    if (len(dataset['training']) != 400 or len(dataset['validation']) != 50
            or dataset['class_name'] != resolved['class_name']
            or config['class_name'] != resolved['class_name']):
        raise ValueError('Dataset manifest class/count mismatch')
    for split in ('training', 'validation'):
        if [r['input_sha256'] for r in dataset[split]] != [
                r['sha256'] for r in manifest['input_files'][split]]:
            raise ValueError(f'{split} source manifest ordering mismatch')
    checked_file(Path(config['resnet_checkpoint']), config['resnet_checkpoint_sha256'])
    inventory = {}
    for item in audit['cache_inventory']:
        relative = item['path'] if schema == 'ten-class-visual-v1' else item['relative_path']
        if relative in inventory:
            raise ValueError(f'Duplicate cache identity: {relative}')
        under(Path(config['source_cache_dir']), relative)
        inventory[relative] = item
    return dict(source_job=source_job, source_commit=source_commit, audit_schema=schema,
                summary=summary, manifest=manifest, resolved=resolved,
                audit=audit, dataset=dataset, inventory=inventory,
                discovery=read('scientific/discovery.json'))


def audited_cache(config, source, relative):
    record = source['inventory'].get(relative)
    if record is None:
        raise ValueError(f'Cache absent from pinned audit: {relative}')
    return checked_file(under(Path(config['source_cache_dir']), relative),
                        record['sha256'], record['bytes'])


def validate_alignment(raw, logits, counts, retained, records, masks, image_paths):
    """Catch row permutations even when all arrays have plausible dimensions."""
    keep = np.any(raw != 0, axis=1)
    if (sum(counts) != len(raw) or logits.shape != (len(raw), 1000)
            or not np.isfinite(raw).all() or not np.isfinite(logits).all()):
        raise ValueError('Raw feature/logit counts or values invalid')
    if not np.array_equal(raw[keep], retained['features']) or not np.array_equal(
            logits[keep], retained['logits']):
        raise ValueError('Raw feature/logit ordering differs from source discovery')
    cursor = 0
    row = 0
    for image_index, (image_masks, filename) in enumerate(zip(masks, image_paths)):
        retained_index = 0
        if len(image_masks) != counts[image_index]:
            raise ValueError('Raw mask count mismatch')
        for mask in image_masks:
            if keep[cursor]:
                record = records[row]
                if (record['image_index'] != image_index
                        or record['segment_index'] != retained_index
                        or record['image_path'] != filename
                        or record['mask_shape'] != list(mask.shape)
                        or record['mask_dtype'] != str(mask.dtype)
                        or record['mask_sha256'] != hashlib.sha256(mask.tobytes()).hexdigest()):
                    raise ValueError('Segment identity/mask ordering mismatch')
                row += 1
                retained_index += 1
            cursor += 1
    if row != len(records):
        raise ValueError('Unconsumed segment source records')
    return keep


class CachedScores:
    """Protocol adapter: reuse exactly the same upstream scores in both modes."""
    def __init__(self, explainer, features, scores, relevance):
        self.concept_bases = explainer.concept_bases
        self.max_shortest_side = explainer.max_shortest_side
        self.features, self.scores, self.relevance = features, scores, relevance

    def concept_activations(self, acts, batch_sizes, norm_batch):
        if not norm_batch or batch_sizes != [len(self.features)] or not np.array_equal(acts, self.features):
            raise ValueError('Cached score input identity or normalization mismatch')
        return self.scores

    def concept_relevances(self, acts):
        if not np.array_equal(acts, self.features):
            raise ValueError('Cached relevance input identity mismatch')
        return self.relevance


def upstream_scores(explainer, features, counts):
    # Upstream normalization is per IMAGE; preserve float32 division before solve.
    # In particular, do not derive norm_batch=True by rescaling already-solved data.
    offsets = np.r_[0, np.cumsum(counts)]
    if any(not np.any(features[a:b]) for a, b in zip(offsets[:-1], offsets[1:])):
        raise ValueError('An all-zero image would yield upstream 0/0 normalization; review required')
    scores = explainer.concept_activations(features, batch_sizes=list(counts), norm_batch=True, n_jobs=1)
    relevance = explainer.concept_relevances(features, n_jobs=1)
    if not np.isfinite(scores).all() or not np.isfinite(relevance).all():
        raise ValueError('Non-finite upstream concept scores/relevance')
    return scores, relevance


def save_masks(path, masks):
    masks = np.asarray(masks)
    if masks.ndim != 3 or not np.logical_or(masks == 0, masks == 1).all():
        raise ValueError('Expected binary masks (states,height,width)')
    np.savez_compressed(path, shape=np.array(masks.shape),
                        bits=np.packbits(masks.astype(bool), axis=None), bitorder=np.array('big'))


def predict_stream(model, trajectories, batch_size, progress, on_batch=None):
    """Exact upstream global batches; discard unused activations, never hook layer4.

    trajectories yields (image_index, masked ImageClass). Each mode/setting starts
    a new stream. Batches can cross image boundaries; only the final batch is short.
    """
    import torch
    import classes
    from utils import utils_general
    from types import SimpleNamespace
    pending, identities, results, groups = [], [], [], []

    def flush():
        dataset = classes.ConceptDatasetClass(
            [SimpleNamespace(segments=pending)], model.default_cfg,
            cropping_mode=0, use_masks=True, masking_mode=1, erosion_threshold=1.0)
        inputs = utils_general.custom_collate([dataset[i] for i in range(len(dataset))])
        with torch.no_grad():
            values = model(inputs).detach().cpu().numpy()
        if values.shape != (len(pending), 1000) or not np.isfinite(values).all():
            raise ValueError('Invalid prediction logits')
        results.extend(values)
        groups.append([list(identity) for identity in identities])
        if on_batch is not None:
            on_batch(values, groups[-1])
        progress(len(results), len(groups))
        pending.clear()
        identities.clear()

    for image_index, image in trajectories:
        for step, segment in enumerate(image.segments):
            pending.append(segment)
            identities.append((image_index, step))
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()
    return np.asarray(results), groups


def runtime_precision(torch):
    return dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                float32_matmul_precision=torch.get_float32_matmul_precision())


def run(config: dict, output: Path) -> dict:
    """Run in a Slurm allocation. Failure receipts are kept; never resume/overwrite."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Evaluation must run through the shared Slurm worker')
    output = Path(output)
    for source_key in ('source_run_dir', 'source_cache_dir'):
        if output.resolve().is_relative_to(Path(config[source_key]).resolve()):
            raise ValueError('Evaluation output must not modify an immutable source directory')
    output.mkdir(parents=True, exist_ok=True)
    # Shared worker may already have placed its own manifest/config here.
    owned = ('evaluation_progress.json', 'evaluation_summary.json', 'evaluation_anomalies.json',
             'evaluation_config.json', 'evaluation_sources.json', 'evaluation_scores.npz',
             'evaluation_curves.json', 'evaluation_artifacts.json', 'evaluation_data')
    if any((output / name).exists() for name in owned):
        raise FileExistsError('Evaluation output already used; choose a fresh independent directory')
    stage = 'verify_sources'
    start = time.monotonic()
    anomalies = []
    def mark(state='RUNNING', **extra):
        write_json(output / 'evaluation_progress.json', dict(
            status=state, stage=stage, slurm_job_id=os.environ['SLURM_JOB_ID'],
            elapsed_seconds=time.monotonic() - start, **extra))
    def note(code, evidence, impact='documented upstream behavior', status='preserved'):
        anomalies.append(dict(code=code, stage=stage, evidence=evidence, impact=impact,
                              status=status, job_id=os.environ['SLURM_JOB_ID']))
        write_json(output / 'evaluation_anomalies.json', anomalies)
    old_warning = warnings.showwarning
    def capture_warning(message, category, filename, lineno, file=None, line=None):
        note('python_warning', dict(message=str(message), category=category.__name__,
                                   filename=filename, line=lineno), 'review warning in context', 'recorded')
        old_warning(message, category, filename, lineno, file, line)
    try:
        write_json(output / 'evaluation_config.json', config)
        write_json(output / 'evaluation_anomalies.json', anomalies)
        mark()
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = capture_warning
            source = verify_sources(config)
            import torch
            import timm
            import classes
            import benchmark_methods as benchmark
            from concept_explainer import ConceptExplainer
            from utils import utils_general
            if runtime_precision(torch) != config['precision']:
                raise ValueError('Runtime precision differs from fixed source flags')
            torch.set_num_threads(config['torch_num_threads'])
            utils_general.DEVICE = config['device']
            if config['device'].startswith('cuda'):
                torch.cuda.set_device(config['device'])
            root = Path(config['source_run_dir'])
            with np.load(root / 'scientific/discovery.npz', allow_pickle=False) as arrays:
                bases = [arrays[k] for k in sorted(arrays.files) if k.startswith('concept_basis_')]
                complement = arrays['complement_basis']
                fc_weight, fc_bias = arrays['fc_weight'], arrays['fc_bias']
            model = timm.create_model('resnet50', pretrained=False)
            model.load_state_dict(torch.load(config['resnet_checkpoint'], map_location='cpu'), strict=True)
            model.eval().to(config['device'])
            if not np.array_equal(model.fc.weight.detach().cpu().numpy(), fc_weight) or not np.array_equal(
                    model.fc.bias.detach().cpu().numpy(), fc_bias):
                raise ValueError('Classifier weights differ from source scientific record')
            model_cfg = json.loads(json.dumps(model.default_cfg))
            if model_cfg != source['discovery']['model_default_cfg']:
                raise ValueError('Model preprocessing configuration differs from source')
            explainer = ConceptExplainer.__new__(ConceptExplainer)
            explainer.model = model
            explainer.target_class = source['resolved']['class_name']
            explainer.concept_bases, explainer.compl_basis = bases, complement
            explainer.max_shortest_side = source['resolved']['max_shortest_side']
            target = utils_general.get_imagenet_class_index(explainer.target_class)
            if target != source['discovery']['class_index']:
                raise ValueError('Target label mapping mismatch')
            stage = 'load_verified_cache'
            mark()
            def one_cache(suffix):
                matches = [r for r in source['inventory'] if r.startswith('activations_validation/') and r.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous raw validation cache: {suffix}')
                return np.load(audited_cache(config, source, matches[0]), allow_pickle=False)
            features, raw_logits = one_cache('_acts.npy'), one_cache('_logits.npy')
            rows = source['dataset']['validation']
            audit_rows = [r for r in source['audit']['images'] if r['split'] == 'validation']
            if len(audit_rows) != 50:
                raise ValueError('Expected exactly 50 audited validation image records')
            images, masks, counts = [], [], []
            for index, (row, audit_row) in enumerate(zip(rows, audit_rows)):
                filename = row['input_path']
                if source['audit_schema'] == 'ten-class-visual-v1':
                    same_input = (audit_row['input_name'] == Path(filename).name
                                  and audit_row['actual_sha256'] == row['input_sha256']
                                  and audit_row['source_id'] == Path(row['source']).name)
                else:
                    same_input = audit_row['actual_input'] == filename
                if audit_row['image_index'] != index or not same_input:
                    raise ValueError('Audit/manifest image order mismatch')
                checked_file(Path(filename), row['input_sha256'])
                image = classes.ImageClass(filename, explainer.max_shortest_side)
                mask_path = audited_cache(config, source, 'segments_validation/' + Path(filename).name + '_sam.npy')
                image_masks = np.load(mask_path, allow_pickle=False)
                if (image_masks.shape[1:] != image.img_numpy.shape[:2]
                        or len(image_masks) != audit_row['raw_segments']
                        or not np.logical_or(image_masks == 0, image_masks == 1).all()):
                    raise ValueError('Raw masks do not match audited image dimensions/count/binary values')
                masks.append(image_masks)
                counts.append(len(image_masks))
                images.append(image)
            if len(images) != len(rows) or features.shape != (sum(counts), fc_weight.shape[1]):
                raise ValueError('Audited image/raw feature counts or feature width mismatch')
            with np.load(root / 'scientific/validation.npz', allow_pickle=False) as retained:
                keep = validate_alignment(features, raw_logits, counts, retained,
                    json.loads((root / 'scientific/validation_segments.json').read_text()),
                    masks, [row['input_path'] for row in rows])
            expected_zero = (sum(r['zero_features'] for r in audit_rows)
                             if source['audit_schema'] == 'ten-class-visual-v1' else 35)
            if int((~keep).sum()) != expected_zero:
                raise ValueError('Zero-feature count differs from pinned source audit')
            offset = 0
            for image, image_masks in zip(images, masks):
                for mask in image_masks:
                    segment = classes.SegmentClass(mask, image)
                    segment.model_act = features[offset]
                    image.segments.append(segment)
                    offset += 1
            stage = 'upstream_scores_once'
            mark()
            scores, relevance = upstream_scores(explainer, features, counts)
            offsets = np.r_[0, np.cumsum(counts)]
            assignments = scores.argmax(axis=1)
            np.savez_compressed(output / 'evaluation_scores.npz', features=features,
                raw_logits=raw_logits, scores=scores, local_relevance=relevance,
                assignments=assignments, zero_features=~keep, offsets=offsets)
            note('zero_feature_argmax_tie', dict(count=int((~keep).sum()), concept_index=0),
                 'Upstream benchmark keeps zero features; mechanical C0 assignment, zero relevance')
            note('upstream_endpoint_and_aggregation', dict(mask_expansion=8, norm_batch=True,
                 relevance='mean per concept per image', endpoint='HU-MCD breaks before appending full endpoint',
                 inclusion='strictly greater than 75% of all 50 images', prediction_masking_mode=1))
            data = output / 'evaluation_data'
            data.mkdir()
            for i, image_masks in enumerate(masks):
                save_masks(data / f'source_masks_{i:03d}.npz', image_masks)
            write_json(output / 'evaluation_sources.json', dict(
                source_job=source['source_job'], source_commit=source['source_commit'],
                class_name=explainer.target_class,
                source_files_sha256=config['source_files_sha256'],
                source_audit_sha256=config['source_audit_sha256'],
                inputs=rows, mask_counts=counts, target_index=target, precision=runtime_precision(torch),
                classifier_sha256=config['resnet_checkpoint_sha256'],
                grouping='Per setting/mode: image order, then state order, global batches of 8; final remainder only',
                partial_prediction_format='*_partial_logits.f32: native float32 [states,1000], corresponding *_partial_batches.jsonl',
                state_mask_format='NPZ shape=[states,height,width], bits=packbits C-order, bitorder=big'))
            write_json(output / 'random_signature.json', random_signature(
                config, model_cfg, rows, random_code_identity(source)))
            curves = {}
            for setting in ('humcd', 'rdm'):
                for mode in ('sdc', 'ssc'):
                    stage = f'{setting}_{mode}'
                    mark()
                    prefix = f'{setting}_{mode}'
                    state_counts, fractions, details = [], [], []
                    seed = config['random_seeds'][mode]
                    previous_rng = random.getstate()
                    random.seed(seed)
                    order_rng = random.Random(seed)
                    def trajectories():
                        for i, image in enumerate(images):
                            a, b = offsets[i:i+2]
                            if setting == 'humcd':
                                adapter = CachedScores(explainer, features[a:b], scores[a:b], relevance[a:b])
                                masked = benchmark.iter_mask_imgs_humcd(adapter, [image], mode)[0]
                                assigned = assignments[a:b]
                                importance = [float(relevance[a:b][assigned == c, c].mean())
                                              if np.any(assigned == c) else 0.0 for c in range(len(bases))]
                                order = np.argsort(importance)[::-1].tolist()
                                extra = dict(importance=importance, concept_order=order,
                                    absent_concepts=[c for c in range(len(bases)) if not np.any(assigned == c)],
                                    complement_segments=int(np.sum(assigned == len(bases))))
                            else:
                                masked = benchmark.iter_mask_imgs_rdm([image], mode, grid_size=(10, 10))[0]
                                order = list(range(100))
                                order_rng.shuffle(order)
                                h, w = image.img_numpy.shape[:2]
                                extra = dict(grid_size=[10, 10], grid_cell_shape=[h // 10, w // 10],
                                             grid_order=order, uncovered_border_pixels=int(h*w-(h//10*10)*(w//10*10)))
                            state_masks = np.stack([s.mask for s in masked.segments])
                            save_masks(data / f'{prefix}_masks_{i:03d}.npz', state_masks)
                            state_counts.append(len(masked.segments))
                            # Preserve upstream float32/bool/float64 mean dtype before multiplying by 100.
                            fractions.append([float(100 * np.mean(s.mask)) for s in masked.segments])
                            details.append(dict(image_index=i, input_path=rows[i]['input_path'],
                                source_id=Path(rows[i]['source']).name, states=len(masked.segments), **extra))
                            yield i, masked
                    try:
                        # Durable append-only per-batch recovery evidence, including on failure.
                        with (data / f'{prefix}_partial_logits.f32').open('xb') as partial, (
                                data / f'{prefix}_partial_batches.jsonl').open('x') as batch_log:
                            def checkpoint(values, identities):
                                if values.dtype != np.float32:
                                    raise ValueError('Expected fixed float32 prediction precision')
                                values.tofile(partial)
                                partial.flush()
                                batch_log.write(json.dumps(identities) + '\n')
                                batch_log.flush()
                            logits, grouping = predict_stream(model, trajectories(), 8,
                                lambda n, batches: mark(predicted_states=n, prediction_batches=batches),
                                on_batch=checkpoint)
                    finally:
                        random.setstate(previous_rng)
                    state_offsets = np.r_[0, np.cumsum(state_counts)]
                    correct = logits.argmax(axis=1) == target
                    by_image = [correct[a:b].tolist() for a, b in zip(state_offsets[:-1], state_offsets[1:])]
                    avg, std = benchmark.calc_avg_and_std(by_image, 50)
                    pixel_avg, pixel_std = benchmark.calc_avg_and_std(fractions, 50)
                    contributors = [sum(c > step for c in state_counts) for step in range(max(state_counts))]
                    included = [i for i, n in enumerate(contributors) if n > 0.75 * 50]
                    curves[prefix] = dict(setting=setting, mode=mode, metric_scope='single_class', class_name=explainer.target_class, target_index=target,
                        step_indices=included, contributors=[contributors[i] for i in included],
                        all_step_contributors=contributors, accuracy_mean=avg.tolist(), accuracy_std=std.tolist(),
                        visible_pixel_percent_mean=pixel_avg.tolist(), visible_pixel_percent_std=pixel_std.tolist(),
                        upstream_plot_x_deleted_fraction=(1 - .01 * pixel_avg).tolist(),
                        state_counts=state_counts, total_states=len(logits), random_seed=seed if setting == 'rdm' else None)
                    np.savez_compressed(data / f'{prefix}_predictions.npz', logits=logits, correct=correct,
                        predicted_class=logits.argmax(axis=1), offsets=state_offsets,
                        visible_pixel_percent=np.concatenate(fractions))
                    write_json(data / f'{prefix}_states.json', dict(images=details, global_batches=grouping,
                        random_seed=seed if setting == 'rdm' else None, python_random_version=random.Random.VERSION))
                    write_json(output / 'evaluation_curves.json', curves)
                    if setting == 'rdm':
                        note('random_grid_uncovered_border', [dict(image_index=d['image_index'],
                            pixels=d['uncovered_border_pixels']) for d in details if d['uncovered_border_pixels']],
                            'Upstream floor-divided grid leaves bottom/right border unchanged')
                    else:
                        note('variable_trajectory_lengths', dict(all_step_contributors=contributors,
                            omitted_steps=[i for i,n in enumerate(contributors) if n <= 37.5]),
                            'Later steps omit images with shorter trajectories; no endpoint padding/interpolation')
            stage = 'charts_and_summary'
            mark()
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            for mode in ('sdc', 'ssc'):
                fig, ax = plt.subplots(figsize=(7, 4))
                for setting in ('humcd', 'rdm'):
                    curve = curves[f'{setting}_{mode}']
                    ax.plot(curve['upstream_plot_x_deleted_fraction'], curve['accuracy_mean'],
                            marker='.', label='HU-MCD' if setting == 'humcd' else 'Random (seed 43)')
                ax.set(xlabel='Deleted pixel fraction (upstream x coordinate)', ylabel='Top-1 accuracy',
                       title=f'{explainer.target_class.replace("_", " ")}: C-{ "Deletion" if mode == "sdc" else "Insertion"}')
                if mode == 'ssc':
                    ax.invert_xaxis()
                ax.legend()
                ax.grid(alpha=.25)
                fig.tight_layout()
                fig.savefig(data / f'{mode}.png', dpi=160)
                plt.close(fig)
            summary = dict(status='PASS', metric_scope='single_class', class_name=explainer.target_class, target_index=target,
                source_job=source['source_job'], source_commit=source['source_commit'],
                validation_images=len(rows), raw_segments=len(features), zero_segments=int((~keep).sum()),
                concepts=len(bases), curves=curves, elapsed_seconds=time.monotonic()-start,
                anomaly_count=len(anomalies), precision=runtime_precision(torch), batch_size=8,
                gpu_name=torch.cuda.get_device_name() if config['device'].startswith('cuda') else None,
                peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated() if config['device'].startswith('cuda') else None)
            write_json(output / 'evaluation_summary.json', summary)
            write_json(output / 'evaluation_artifacts.json', [dict(path=str(p.relative_to(output)),
                bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(output.rglob('*'))
                if p.is_file() and (p.relative_to(output).parts[0] == 'evaluation_data'
                    or p.name.startswith('evaluation_'))
                and p.name not in ('evaluation_artifacts.json', 'evaluation_progress.json')])
            mark('PASS')
            return summary
    except Exception as error:
        note('evaluation_failure', dict(error=repr(error), traceback=traceback.format_exc()),
             'Affected evaluation stopped; partial outputs preserved', 'unresolved')
        mark('FAIL', error=repr(error))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
