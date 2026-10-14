# NotebookLM MCP pilot

## Verified on this workstation

The pilot runs entirely in Ubuntu WSL, alongside the repository and SSH. It uses
`notebooklm-mcp-cli==0.10.1` in an isolated environment, a separate Chrome for
Testing browser, and a dedicated `humcd-pilot` login profile. This is an
unofficial client using NotebookLM's internal interface; it is not a Google
public API integration. Keep the tested version pinned for this pilot.

The user completed Google sign-in manually. The browser closing afterward is
normal: the login helper saves the session and closes its own browser.

Verified on 2026-09-08:

- MCP initialize and exact six-tool server allowlist passed.
- Authenticated notebook listing, notebook metadata and source retrieval passed.
- Two grounded queries completed against the existing **Concept Bottleneck
  Models** notebook, using its Koh et al. (2020) PDF.
- All 31 returned citation passages were found in the independently retrieved
  source text after Unicode/whitespace normalization.
- The source-supported training definitions were recovered, and an intentionally
  unsupported GPU-budget/Slurm-partition question received an explicit
  not-specified answer.
- The factual answer also overgeneralized some dataset-specific performance
  comparisons. Citation presence alone did not guarantee a precise summary.

Verdict: useful as a source-retrieval and research-query assistant with review.
This pilot does not validate HU-MCD paper reproduction; that paper was not among
the selected notebook's sources. No repository, datasets, weights or experiment
outputs were uploaded. Queries can create NotebookLM conversation history, even
though notebook/source-management tools are excluded.

## Installed locations

| Item | Location |
|---|---|
| Python environment and browser | `/home/chen/.local/share/hu-mcd-notebooklm/` |
| Dedicated authentication state | `/home/chen/.local/share/hu-mcd-notebooklm/state/` |
| CLI wrapper | `/home/chen/.local/bin/humcd-nlm` |
| MCP wrapper | `/home/chen/.local/bin/humcd-notebooklm-mcp` |
| Browser wrapper | `/home/chen/.local/bin/humcd-nlm-chrome` |
| Codex server configuration | `/mnt/c/Users/uqcche38/.codex/config.toml`, server `notebooklm_pilot` |
| Configuration backup | `/home/chen/.local/share/hu-mcd-notebooklm/codex-config.before-notebooklm.toml` |

Reviewable copies of the three wrappers, dependency lock and browser version
record are in `tools/notebooklm/`. They reflect this workstation's absolute
paths. They contain no login credentials. Authentication state is outside the
repo with private directory permissions. The browser's missing Ubuntu libraries
were extracted into its private directory; system packages and Windows firewall
rules were not modified. Chrome retains normal browser sandboxing. The standalone
browser is not auto-updated; refresh it deliberately before continuing a long-term
installation, keeping a new version record and rerunning the launch check.

## Use

Codex reads the server from its normal configuration. Its CLI recognizes the
entry, and a real stdio MCP client verified startup and tool calls. If an already
open Codex task does not expose the new tools, restart Codex and reopen the WSL
project to load the configuration. The six exposed tools are:

`notebook_list`, `notebook_get`, `source_get_content`, `notebook_query`,
`notebook_query_start`, `notebook_query_status`.

All server tool groups are disabled first, then only those six are enabled.
Codex's own `enabled_tools` list applies the same selection. Notebook creation,
source upload/delete, sharing, artifact generation and authentication tools are
excluded from the MCP pilot. The CLI remains a full-capability manual tool.

Connection check without notebook queries:

```bash
cd /home/chen/projects/hu-mcd
/home/chen/.local/share/hu-mcd-notebooklm/bin/python scripts/notebooklm_pilot.py
```

Repeat the two-question pilot on an existing notebook:

```bash
/home/chen/.local/share/hu-mcd-notebooklm/bin/python scripts/notebooklm_pilot.py \
  --notebook-id YOUR_NOTEBOOK_UUID
```

This records raw tool evidence in a fresh private directory under
`artifacts/notebooklm/`. It deliberately leaves `grounding_review` as
`not_evaluated`: a successful API response is not an automatic scientific pass.
The current run has a separate human-readable `review.md` with the findings.
Notebook sources and answers are ignored by Git; do not commit raw notebook data
as part of the software changes.

If authentication expires, sign in yourself:

```bash
/home/chen/.local/bin/humcd-nlm login
```

Use this wrapper so the correct WSL browser and dedicated profile are selected.
Do not add `--wsl`; this pilot uses the local WSL browser and needs no Windows
port proxy. Do not paste cookies or passwords into Codex. Individual queries
have a 120-second timeout and automatic rate-limit retries are disabled.

To remove only this Codex connection:

```bash
codex mcp remove notebooklm_pilot
```

That leaves the isolated environment, authentication state and other Codex servers
in place. Avoid restoring the entire old config after other settings have changed.

## Next research step

Create a dedicated HU-MCD notebook in the NotebookLM UI containing the paper,
supplement and public author documentation, with source versions recorded. Ask
Codex to retrieve the source passages for a specific reproduction decision,
compare them with the code, and write a reviewed note that distinguishes paper
requirements, implementation choices and unresolved details. Keep those concise
reviewed notes in Git. A NotebookLM summary should not automatically become an
agent skill or override project instructions.

Prioritize this deliberate workflow over adding Mutagen now. The collector
already brings selected Bunya results into WSL. The remaining execution task is
to wire the smoke launcher to the exact prepared Git release and validate the
new per-run output path in the next planned Slurm run. Continuous remote sync
would not solve either provenance requirement.

Sources: [client repository](https://github.com/jacob-bd/gemini-notebook-mcp-cli),
[Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli),
[Chrome for Testing](https://github.com/GoogleChromeLabs/chrome-for-testing).
