"""Archive-bound R101 fitting; uses released discovery operations and accepted classifier."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import traceback
import warnings

from hpc.derm_fitting import execution_gate
from utils.run_tracking import atomic_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    config = json.loads(Path(a.config).read_text())
    execution_gate(config)
    if config['mode'] != 'derm-fitting' or not re.fullmatch('[a-f0-9]{40}', a.commit):
        raise ValueError('Wrong HAM protocol or execution commit')
    if sha256(a.archive) != a.archive_sha256:
        raise ValueError('Execution archive identity mismatch')
    # The launcher hashes/extracts this archive and checks this worker before invocation.
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    config['execution_commit'] = a.commit
    atomic_json(out/'resolved_config.json', config)
    atomic_json(out/'execution.json', dict(commit=a.commit, archive=a.archive,
        archive_sha256=a.archive_sha256, config_file_sha256=sha256(a.config),
        job_id=os.environ['SLURM_JOB_ID'], utc=utc_now()))
    atomic_json(out/'status.json', dict(status='RUNNING', utc=utc_now()))
    # Persist warnings as they occur, including those preceding a failure.
    original_showwarning = warnings.showwarning
    def record_warning(message, category, filename, lineno, file=None, line=None):
        with (out/'warnings.jsonl').open('a') as f:
            f.write(json.dumps(dict(message=str(message), category=category.__name__,
                filename=filename, line=lineno, utc=utc_now()))+'\n')
        original_showwarning(message, category, filename, lineno, file=file, line=line)
    warnings.showwarning = record_warning
    warnings.simplefilter('always')
    try:
        from hpc.derm_fitting import run
        result = run(config, out)
        atomic_json(out/'result.json', result)
        atomic_json(out/'status.json', dict(status=result['status'], utc=utc_now()))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        atomic_json(out/'status.json', dict(status='FAILED_NO_AUTOMATIC_RETRY', utc=utc_now()))
        raise
    finally:
        warnings.showwarning = original_showwarning
        files = {str(f.relative_to(out)):dict(bytes=f.stat().st_size, sha256=sha256(f))
                 for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'artifacts.json'}
        atomic_json(out/'artifacts.json', dict(job_id=os.environ['SLURM_JOB_ID'],
                    execution_commit=a.commit, files=files))


if __name__ == '__main__':
    main()
