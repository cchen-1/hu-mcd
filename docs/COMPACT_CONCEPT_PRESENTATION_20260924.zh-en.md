# Compact concept presentation / 紧凑概念展示

2026-09-24 Brisbane. Presentation-only revision; no remote operations, model calls, sample changes or scientific configuration changes.

- Ten per-class method summaries: one row per ACE/MCD/HU-MCD, four saved-rank groups per method, prototype/random pairs. Actual retained counts, stable IDs, selected-area labels and differing mask rules are included. Absent ranks remain absent. Existing detailed views are folded, not deleted.
- Fig3-style HU overview: all ten classes, each with its top four saved ranks and four existing validation prototypes per concept. This shows40/147 learned concepts, not all147. Four examples are reused rather than generating the paper's six. Complete class sheets include every learned concept, with random-member companion sheets and paged presentation exports. Complements remain in the original separate details.
- 57 high-resolution PNG exports, with exact source/mask checks and a selection CSV. Four-example groups with fewer saved examples remain incomplete explicitly; no replacement is made. The report's cyan/grey full-canvas display is retained consistently; this is a layout adaptation of Fig3, not a claim of identical example selection or rendering.
- Method ranks do not establish semantic identity. Different sample counts (ACE50 versus HU/MCD400) and example/mask construction remain disclosed. Human review status is unchanged.

紧凑图优先整体比较；逐成员鉴别仍使用原有大图、原图与掩码详情。不能仅凭作者展示三类断言挑选不当；全十类展示减少本报告的类别选择偏差，而top4仍是明确披露的展示子集。

## Inpainting and area / 方法说明

The source implements fixed-colour filling, not generative restoration: `I*M + 0.4588*(1-M)`. ACE uses cropped filling; current HU uses Input Masking. The missing experiment is the supplement Fig7 HU alternative-masking ablation. Released `run_benchmark` separates method-specific feature extraction from final flipped-region predictions, which explicitly use Input Masking with mode1. An ablation must freeze which stage changes; code primitives alone do not establish that exact experiment.

Selected area is the binary selected pixel count divided by full mask canvas area. HU display uses original SAM masks, before released erosion/resizing; MCD uses224×224 display masks with prototype>.4 versus final argmax rules. ACE cards use full display canvases. Card area is not importance, confidence or cumulative evaluation coverage. Benchmark cumulative regions handle overlap and changing contributor counts separately.

## Delivery / 交付

Canonical private report unchanged: `artifacts/bunya/ten-class-review/report/index.html`.
New compact atlas: `concept-summary.html`; explanation: `masking-and-area.html`; method figures appear first in every `comparison-CLASS.html`; all-concept figures appear first in each `classes/CLASS.html`.

Generator: `scripts/research_hub/compact_figures.py`, connected to the existing interpretation generator for regeneration.
Evidence, QA, sources and publication receipt: `artifacts/report-compact-20260924/`.
Records: REPRO26-006/007, plus English Obsidian Implementation Notes. All original scientific results, stable review IDs, storage schema, hpc/check_job.sh, outputs and unrelated work remain preserved.

## Navigation follow-up / 导航衔接

The existing `reproduction.html#class-directory` is now the shared ten-class directory. Homepage → directory → HU concepts / method comparison / class evaluation. Existing class pages share the same tabs and a class switch preserving view and language; the atlas exposes links before each tall figure. No separate report entry was created. Generator `scripts/research_hub/navigation.py` is called by compact and full report generation. REPRO26-008 records the resolved usability issue and targeted offline browser checks. Receipt: `artifacts/report-navigation-20260924/publication.json`.
