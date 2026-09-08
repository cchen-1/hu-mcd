"""Compute-only preparation of fixed inputs and official SAM ViT-H weights."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

L_ID = 'ILSVRC2012_val_00019590.JPEG'
L_SHA = 'ff191fd4f872734df5bab18cc8feef77d12d14ad163c85781cc685ad752e3614'
SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def compatibility(manifest_path, expected_sha, output):
    from PIL import Image
    if digest(manifest_path) != expected_sha:
        raise ValueError('Frozen prepared manifest changed')
    old = json.loads(Path(manifest_path).read_text())
    if old['seed'] != 43 or old['synset'] != 'n02099601' or len(old['training']) != 400 or len(old['validation']) != 50:
        raise ValueError('Wrong frozen split')
    output = Path(output)
    source_root = Path(old['licensed_source_root']).resolve(strict=True)
    if not output.is_absolute() or output.exists() or output.resolve() == source_root or source_root in output.resolve().parents:
        raise ValueError('Require a new private directory outside licensed source')
    original_root = Path(old['source_dir'])
    for split in ('training','validation'):
        original_text = (original_root/(split+'_images.txt')).read_text()
        if original_text != '\n'.join(e['source'] for e in old[split])+'\n':
            raise ValueError('Original text list differs from frozen manifest')
    all_paths,all_hashes = set(),set()
    for split in ('training','validation'):
        for entry in old[split]:
            src = Path(entry['source']).resolve(strict=True)
            if src.parent != source_root / ('train' if split=='training' else 'val') / 'n02099601':
                raise ValueError('Source outside original split')
            if digest(src) != entry['sha256'] or str(src) in all_paths or entry['sha256'] in all_hashes:
                raise ValueError('Changed or duplicate source input')
            all_paths.add(str(src));all_hashes.add(entry['sha256'])
            with Image.open(src) as im:
                im.load()
                if im.mode != 'RGB' and not (split=='validation' and src.name==L_ID and entry['sha256']==L_SHA and im.mode=='L'):
                    raise ValueError('Unexpected non-RGB source: '+str(src))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compat-',dir=output.parent) as td:
        stage=Path(td)/'split'; stage.mkdir()
        updated=json.loads(json.dumps(old))
        updated.update(schema_version=3, source_dir=str(output), original_prepared_root=str(original_root),
                       original_manifest_sha256=expected_sha, preparation_job_id=os.environ.get('SLURM_JOB_ID'),
                       original_validation_issues=old.get('validation_issues',[]), validation_issues=[],ready_for_reference=True)
        mapping=[]; conversions=0
        for split,folder in [('training','golden_retriever'),('validation','val_imgs/golden_retriever_val')]:
            (stage/folder).mkdir(parents=True)
            for entry in updated[split]:
                src=Path(entry['source'])
                if Path(entry['prepared_name']).name != entry['prepared_name']:
                    raise ValueError('Unsafe prepared name')
                conversion=None
                if split=='validation' and src.name==L_ID:
                    conversions+=1
                    entry['prepared_name']=Path(entry['prepared_name']).with_suffix('.png').name
                    dest=stage/folder/entry['prepared_name']
                    with Image.open(src) as original:
                        pixels=original.tobytes()
                        original.convert('RGB').save(dest,format='PNG')
                        with Image.open(dest) as actual:
                            actual.load()
                            if actual.mode!='RGB' or actual.size!=original.size or any(c.tobytes()!=pixels for c in actual.split()):
                                raise ValueError('RGB PNG did not preserve every grayscale pixel')
                        conversion={'kind':'input_compatibility_adjustment','operation':'PIL L -> RGB PNG, identical replicated channels',
                                    'original_mode':'L','input_mode':'RGB','size':list(original.size),
                                    'grayscale_pixels_sha256':hashlib.sha256(pixels).hexdigest(),
                                    'pixel_equality_checked':True}
                else:
                    (stage/folder/entry['prepared_name']).symlink_to(src)
                entry.update(input_path=str(output/folder/entry['prepared_name']),
                             input_sha256=digest(stage/folder/entry['prepared_name']),input_mode='RGB',
                             compatibility_adjustment=conversion)
                mapping.append({'split':split,'image_id':src.name,'original_source':str(src),'original_sha256':entry['sha256'],
                                'input_path':entry['input_path'],'input_sha256':entry['input_sha256'],
                                'adjustment':conversion})
            # Preserve original IDs/source lists byte-for-byte; separate actual input list.
            (stage/(split+'_images.txt')).write_bytes((original_root/(split+'_images.txt')).read_bytes())
            (stage/(split+'_actual_inputs.txt')).write_text('\n'.join(e['input_path'] for e in updated[split])+'\n')
        if conversions!=1:
            raise ValueError('Expected exactly one approved compatibility adjustment')
        if len({e['input_sha256'] for e in mapping})!=450:
            raise ValueError('Actual input content duplicates')
        save(stage/'input_mapping.json',mapping)
        updated['input_mapping_sha256']=digest(stage/'input_mapping.json')
        save(stage/'dataset_manifest.json',updated)
        # Use link directory rename under exclusive cooperative publication lock.
        import fcntl
        with (output.parent/(output.name+'.publish.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if output.exists(): raise FileExistsError(output)
            os.rename(stage,output)
    return {'source_dir':str(output),'dataset_manifest':str(output/'dataset_manifest.json'),
            'dataset_manifest_sha256':digest(output/'dataset_manifest.json'),
            'input_mapping_sha256':digest(output/'input_mapping.json'),
            'training_count':400,'validation_count':50,'conversions':1,'ready_for_reference':True}


def prepare_model(root, evidence_dir):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/'sam_vit_h_4b8939.pth'
    metadata=target.with_suffix('.provenance.json')
    if target.exists():
        if not metadata.is_file():
            raise ValueError('Existing checkpoint has no provenance; inspect before replacing')
        record=json.loads(metadata.read_text())
        if record['url']!=SAM_URL or record['sha256']!=digest(target):
            raise ValueError('Existing checkpoint provenance mismatch')
    else:
        part=root/('sam_vit_h_4b8939.pth.part-'+os.environ['SLURM_JOB_ID'])
        with urllib.request.urlopen(SAM_URL,timeout=60) as response, part.open('xb') as stream:
            if not response.geturl().startswith('https://dl.fbaipublicfiles.com/'):
                raise ValueError('Unexpected checkpoint redirect')
            length=int(response.headers['Content-Length'])
            if length<2_000_000_000 or length>3_000_000_000:
                raise ValueError('Unexpected ViT-H checkpoint size')
            for chunk in iter(lambda:response.read(8*1024*1024),b''):
                stream.write(chunk)
        if part.stat().st_size!=length:
            raise ValueError('Incomplete checkpoint')
        record={'url':SAM_URL,'bytes':length,'sha256':digest(part),'preparation_job_id':os.environ['SLURM_JOB_ID'],
                'verification':'HTTPS official published URL; full bytes and local SHA256 recorded; architecture load checked in GPU probe'}
        # Do not overwrite a checkpoint published concurrently.
        os.link(part,target);part.unlink()
        save(metadata,record)
    save(Path(evidence_dir)/'sam_checkpoint.json',record)
    return record
