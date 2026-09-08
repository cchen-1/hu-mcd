"""Small, stdlib-only run records written on the machine doing the experiment."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     delete=False, encoding='utf-8') as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class StageTimes(dict):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def __setitem__(self, name, seconds):
        super().__setitem__(name, seconds)
        self.tracker.progress('RUNNING', last_completed_stage=name)


class RunTracker:
    def __init__(self, config, config_path, repo_root, run_id=None):
        if run_id is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
            raise ValueError('run-id must contain 1-80 letters, digits, underscores or hyphens')
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id and run_id != job_id:
            raise ValueError('A Slurm smoke run requires --run-id equal to SLURM_JOB_ID')
        self.config = copy.deepcopy(config)
        self.repo_root = Path(repo_root).resolve()
        # Relative paths in the runner are relative to the repository.
        for key in ('source_dir', 'output_dir', 'cache_root'):
            value = Path(self.config[key]).expanduser()
            self.config[key] = str((value if value.is_absolute() else self.repo_root / value).resolve())
        checkpoint = Path(self.config['segmentation']['checkpoint']).expanduser()
        self.config['segmentation']['checkpoint'] = str(
            (checkpoint if checkpoint.is_absolute() else self.repo_root / checkpoint).resolve())
        if run_id:
            for key in ('output_dir', 'cache_root'):
                self.config[key] = str(Path(self.config[key]).parent / 'runs' / run_id)
        self.output = Path(self.config['output_dir'])
        self.output.mkdir(parents=True, exist_ok=run_id is None)
        # Always start with an empty run cache; reusing a job ID must fail.
        Path(self.config['cache_root']).mkdir(parents=True, exist_ok=run_id is None)
        self.identity = {'run_id': run_id, 'slurm_job_id': job_id}
        self.started = utc_now()
        self.stage_times = StageTimes(self)
        self.last_stage = None
        self.artifacts = {}
        self.manifest = {'schema_version': 1, **self.identity, 'started_at_utc': self.started,
                         'config_source': str(config_path), 'repo_root': str(self.repo_root),
                         'git_commit': None, 'git_dirty': None, 'source_sha256': {},
                         'source_hash_scope': 'repository Python files and hpc scripts, excluding runtime data',
                         'input_files': {}}
        try:
            def git(*args):
                return subprocess.run(['git', '-C', str(self.repo_root), *args], capture_output=True,
                                      text=True, timeout=10, check=True).stdout.strip()
            self.manifest['git_commit'] = git('rev-parse', 'HEAD')
            self.manifest['git_dirty'] = bool(git('status', '--porcelain', '--untracked-files=normal'))
        except (OSError, subprocess.SubprocessError):
            self.manifest['git_note'] = 'Git provenance unavailable; consult source file hashes.'
        # Named source directories only; never traverse datasets, models, cache or outputs.
        sources = list(self.repo_root.glob('*.py'))
        for directory in ('utils', 'hpc'):
            if (self.repo_root / directory).is_dir():
                sources.extend(p for p in (self.repo_root / directory).rglob('*')
                               if p.suffix in ('.py', '.sh', '.sbatch', '.json') and p.is_file())
        self.manifest['source_sha256'] = {str(p.relative_to(self.repo_root)): sha256(p) for p in sorted(sources)}
        atomic_json(self.output / 'resolved_config.json', self.config)
        self.manifest['resolved_config_sha256'] = sha256(self.output / 'resolved_config.json')
        atomic_json(self.output / 'run_manifest.json', self.manifest)
        self.progress('RUNNING')

    def record_inputs(self, split, images):
        self.manifest['input_files'][split] = [
            {'path': str(Path(image.filename).resolve()), 'sha256': sha256(image.filename)} for image in images]
        atomic_json(self.output / 'run_manifest.json', self.manifest)

    def progress(self, status, last_completed_stage=None, error=None):
        if last_completed_stage:
            self.last_stage = last_completed_stage
            # Publish only after save_concepts has returned, so polls can reuse
            # the finalized training sheet during subsequent validation stages.
            sheets = {'training_assignment_and_prototypes': 'training_prototypes/overview.png',
                      'validation_assignment_and_prototypes': 'validation_prototypes/overview.png'}
            if last_completed_stage in sheets:
                name = sheets[last_completed_stage]
                path = self.output / name
                self.artifacts[name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
        atomic_json(self.output / 'artifacts.json',
                    {'schema_version': 1, **self.identity, 'files': self.artifacts})
        atomic_json(self.output / 'progress.json', {
            'schema_version': 1, **self.identity, 'status': status, 'started_at_utc': self.started,
            'updated_at_utc': utc_now(), 'last_completed_stage': self.last_stage,
            'stage_elapsed_seconds': dict(self.stage_times), 'error': error})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.progress('FAILED', error=f'{exc_type.__name__}: {exc}')
            return False
        files = ('summary.json', 'metrics_report.md', 'training_prototypes/overview.png',
                 'validation_prototypes/overview.png')
        try:
            index = {name: {'bytes': (self.output / name).stat().st_size,
                            'sha256': sha256(self.output / name)} for name in files}
            scientific = self.output / 'scientific'
            if scientific.is_dir():
                for path in sorted(scientific.iterdir()):
                    if path.is_file():
                        index[str(path.relative_to(self.output))] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
            self.artifacts = index
            self.progress('PASS')
        except BaseException as error:
            self.progress('FAILED', error=f'Artifact finalization: {error}')
            raise
        return False
