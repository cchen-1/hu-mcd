"""Class identity and dependency gates for the unchanged Golden baseline protocols.

Runs only inside workstream_runtime on a Slurm compute node. Never submits jobs.
"""
import importlib
import json
import os
from pathlib import Path
import socket

from hpc.evaluate_reference import checked_file, sha256, write_json, verify_sources

ROOT=Path(__file__).resolve().parents[1]


def verify_reference(config, science_files):
    from hpc.workstream_runtime import verify_inputs
    from hpc.mcd_reference import PROTOCOL
    if not config.get('multiclass_reference') or config['class_name']=='golden_retriever':
        raise ValueError('Use the original accepted Golden path')
    if config.get('protocol',PROTOCOL)!=PROTOCOL:
        raise ValueError('MCD scientific protocol changed')
    if config['max_shortest_side']!=300 or config['model_name']!='resnet50':
        raise ValueError('Changed Golden classifier preprocessing')
    source=verify_sources(config)
    verify_inputs(config)
    launch=json.loads(checked_file(Path(config['golden_launch_manifest']),config['golden_launch_sha256']).read_text())
    if launch['status']!='EXECUTION_COMPLETED':raise ValueError('Golden environment attestation incomplete')
    science={}
    for name in science_files:
        if name.startswith('input_masking/'):
            proof=launch['masking'][Path(name).name]
            if not proof['match'] or proof['expected_sha256']!=proof['installed_sha256']:
                raise ValueError('Golden masking attestation failed')
            expected=proof['installed_sha256']
            if source['audit']['core_sha256'][name]!=expected:
                raise ValueError('Class and Golden mask implementations differ')
        else:expected=source['manifest']['source_sha256'][name]
        science[name]=sha256(checked_file(ROOT/name,expected))
    if config['layer_name'] not in ('layer4','global_pool'):
        raise ValueError('Unsupported baseline feature layer')
    source.update(science=science,launch=launch)
    return source


def resolve(config):
    """Resolve future manifest hashes only from exact successful producer jobs.

    afterok provides scheduling, while this gate establishes data/commit/class
    identity. A failed gate stops only its own dependency chain; no retries.
    """
    resolved=dict(config);bindings=[]
    root=Path(config['runtime_root'])/'outputs/workstreams'
    for dep in config.get('stage_dependencies',[]):
        directory=root/str(dep['job_id'])
        launch=json.loads((directory/'launch_manifest.json').read_text())
        index=json.loads((directory/'artifacts.json').read_text())
        old=json.loads((directory/'actual_config.json').read_text())
        if (launch['status']!='PASS' or index['status']!='PASS'
                or str(index['job_id'])!=str(dep['job_id'])
                or index['commit']!=config['execution_commit']
                or launch['commit']!=config['execution_commit']
                or launch['task_key']!=dep['task_key']):
            raise ValueError('Upstream stage did not complete with expected job/code/task identity')
        for key in ('class_name','dataset_manifest_sha256','resnet_checkpoint_sha256','precision'):
            if old[key]!=config[key]:raise ValueError('Upstream stage changed '+key)
        rel=dep['manifest'];item=index['files'][rel]
        path=checked_file(directory/rel,item['sha256'],item['bytes'])
        manifest=json.loads(path.read_text())
        if manifest['status']!='PASS':raise ValueError('Failed upstream scientific manifest')
        if dep['prefix']=='ace_input':
            if manifest['schema']!='ace-r-inputs-v1':raise ValueError('Wrong ACE input schema')
            resolved['ace_input_manifest']=str(path);resolved['ace_input_manifest_sha256']=item['sha256']
        else:
            if manifest['stage']!=dep['stage']:raise ValueError('Wrong upstream scientific stage')
            resolved[dep['prefix']+'_dir']=str(directory)
            resolved[dep['prefix']+'_manifest_sha256']=item['sha256']
        bindings.append(dict(**dep,manifest_sha256=item['sha256'],producer_artifacts_sha256=sha256(directory/'artifacts.json')))
    return resolved,bindings


def run(config,output):
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().split('.')[0].startswith('bun'):
        raise RuntimeError('Requires Slurm compute allocation')
    resolved,bindings=resolve(config)
    write_json(Path(output)/'resolved_dependencies.json',dict(status='PASS',bindings=bindings))
    write_json(Path(output)/'resolved_stage_config.json',resolved)
    method=resolved['baseline_method'];stage=resolved['stage']
    module='hpc.ace_inputs' if method=='ACE' and stage=='inputs' else {'ACE':'hpc.ace_reference','MCD':'hpc.mcd_reference'}[method]
    return importlib.import_module(module).run(resolved,output)
