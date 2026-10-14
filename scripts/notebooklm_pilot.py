#!/usr/bin/env python3
"""Check the local NotebookLM MCP connection and optionally query one notebook.

Run with the isolated NotebookLM environment's Python. Raw source/answer data
is saved locally under ignored artifacts/notebooklm, never into tracked docs.
A successful tool call does not automatically pass a scientific grounding check.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED = {'notebook_list', 'notebook_get', 'source_get_content', 'notebook_query',
            'notebook_query_start', 'notebook_query_status'}
QUESTIONS = [
    'Using only the sources in this notebook, explain the role of the concept representation '
    'and compare independent, sequential, and joint training if those variants are present. '
    'Cite the relevant source passages and say explicitly if a variant is not covered.',
    'What exact GPU-hour budget and Slurm partition do the sources prescribe for my next '
    'HU-MCD reproduction experiment? If the sources do not contain these details, say they '
    'are not specified. Do not estimate or infer a budget or partition.'
]


def result_data(result):
    if result.structuredContent is not None:
        return result.structuredContent
    for item in result.content:
        if item.type == 'text':
            return json.loads(item.text)
    raise RuntimeError('MCP response contains no structured result')


async def execute(args):
    os.umask(0o077)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    folder = args.output_root.expanduser().resolve() / name
    folder.mkdir(parents=True)
    params = StdioServerParameters(command=str(Path.home() / '.local/bin/humcd-notebooklm-mcp'))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            actual = {tool.name for tool in (await session.list_tools()).tools}
            if actual != EXPECTED:
                raise RuntimeError(f'Tool allowlist mismatch: {actual ^ EXPECTED}')
            checks = {'initialize': 'passed', 'tool_allowlist': 'passed', 'tools': sorted(actual),
                      'grounding_review': 'not_evaluated'}
            (folder/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
            print('MCP initialize and exact six-tool allowlist: PASS', flush=True)
            async def call(name, arguments, filename):
                result = await session.call_tool(name, arguments)
                data = result_data(result)
                (folder/filename).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
                if result.isError or data.get('status') != 'success':
                    raise RuntimeError(f'{name} did not succeed; inspect {folder/filename}')
                return data
            listing = await call('notebook_list', {'max_results': 100}, 'notebooks.json')
            print(f'Authenticated notebook listing: PASS ({len(listing.get("notebooks", []))} notebooks)', flush=True)
            if args.notebook_id:
                notebook = await call('notebook_get', {'notebook_id': args.notebook_id}, 'notebook.json')
                sources = notebook.get('sources', [])
                if not sources:
                    raise RuntimeError('Selected notebook has no sources')
                source_id = sources[0].get('id') or sources[0].get('source_id')
                if not source_id:
                    raise RuntimeError('Source response has no identifiable source ID')
                await call('source_get_content', {'source_id': source_id}, 'source.json')
                print('Notebook and first source retrieval: PASS', flush=True)
                for number, question in enumerate(QUESTIONS, 1):
                    await call('notebook_query', {'notebook_id': args.notebook_id, 'query': question,
                              'source_ids': [source_id], 'new_conversation': True, 'timeout': 120}, f'query-{number}.json')
                    print(f'Query {number}: returned; source-grounding review still required', flush=True)
                checks['query_calls'] = 'passed'
                checks['notebook_id'] = args.notebook_id
            checks['authenticated_listing'] = 'passed'
            (folder/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook-id', help='Existing notebook UUID to evaluate; omit for connection-only checks')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/notebooklm')
    args = parser.parse_args()
    if args.notebook_id:
        uuid.UUID(args.notebook_id)
    asyncio.run(execute(args))


if __name__ == '__main__':
    main()
