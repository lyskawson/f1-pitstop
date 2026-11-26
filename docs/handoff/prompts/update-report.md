TASK — update and simplify the shared F1 Pitstop report

Work in the existing report repository:
https://github.com/lyskawson/f1-pitstop-report
The application source is:
https://github.com/lyskawson/f1-pitstop

Resolve local paths. Alek's report checkout is under f1-notes/raport-latex,
but use the receiving human's actual checkout or clone the existing remote.
Preserve teammates' edits and integrate the report remote before writing.
The current report repository contains the template, title page and logo.

Update the actual LaTeX sources, build/check the PDF and publish the checked
sources. The audience is the three students and their course supervisor.
Write in Polish so that one reading explains what was built, how it works,
what has been verified and what is still incomplete.

Establish the evidence:
- Read report-state.json, the existing chapters and report maintenance files.
  Resolve accepted application main once and read versioned files at that SHA
  through read-only Git operations or a temporary source export. Preserve the
  live application checkout. Compare with the last documented source revision.
- Read the relevant application code, tests, decisions, contracts, runbook and
  accepted task evidence. Use available author reports and verified CI records.
  Reuse saved measurements. Application testing/training and predictive
  evaluation belong to their implementation tasks.
- The 2026-11-26 checkpoint (228f320) adds T4c/T4d contracts, T4 features and
  windows, T6 baselines, T8–T10 MLP/GRU/TCN and the T11 validation study
  (reports/validation-v1.md, val only) to T2c, data-v1, T3 and T5. The report
  still documents T2. T12 and later work count as completed only when present
  in the accepted main you inspect. Test results exist only after T12.
  Current source checkpoint and baseline evidence are in docs/handoff/.
- Keep producing revisions and data limitations accurate. Existing sources
  may have unknown acquisition dates with documented explanations. The shared
  raw archive is distinct from the future feature dataset.

Edit the whole narrative, including older paragraphs:
- Preserve the supplied report-class template, title-page metadata, logo,
  typography and restrained layout.
- Keep the existing five chapters, from Cel i zakres projektu through
  Dane i przygotowanie zbioru, Metody i realizacja, Eksperymenty i wyniki
  and Dyskusja i wnioski.
- Integrate completed work into the corresponding sections and replace
  fulfilled plans. Update conclusions and degree of goal completion.
- Prefer short, connected sentences and familiar technical terms. Explain
  each important concept when first used, then use it consistently.
  Remove bureaucratic wording, repetitive caveats and process narration.
- Explain each stage through its input, operation, output and purpose.
  Include implementation details only when they help readers understand a
  method, reproduce the main reasoning or assess a material limitation.
- Give each subject one main home. Compress repeated lists of files,
  validators, safeguards, exclusions and pending work.
- Move full hashes, commit IDs, task numbers, exact dependency pins, command
  inventories, exhaustive schema checks and arithmetic for synthetic tests
  into maintenance/evidence files. Use the PDF space for the project.
- Explain the practical significance of the tyre-label ambiguity and causal
  time cutoff concisely. A reader needs the adopted rule and its consequences.
  Recounting each debugging step or penalty example is unnecessary.
- Describe software verification and predictive experiments with accurate
  scope. T5's synthetic perfect scores verify metric calculations. Current
  model performance claims require actual approved runs and saved results.
- Keep the main text proportionate to completed work. Integrate and shorten
  older text as content grows. Choose length from substance and readability,
  preserving comfortable fonts and margins. There is no required page target.
- Read the revised chapters as continuous prose. Delete sentences whose only
  role is to restate a nearby paragraph or announce an obvious section.
  Check that a student unfamiliar with implementation can follow the account.

For an update through T5, cover the concrete additions:
- The portable raw archive and honest legacy provenance.
- The implemented next-lap tyre-change rule, reasons for unresolved labels,
  and the fixed season/round split before exclusions.
- The implemented AP/F1 threshold procedure, early/on-time event evaluation,
  race bootstrap and verified artifact workflow at an explanatory level.
- The available offline verification evidence. Keep models, processed
  features, training and dashboard described according to the actual source.
For later updates, apply the same approach to the newly merged work and remove
obsolete limitations. The same prompt serves the teammate after their tasks.

Retain the course requirements in a coherent account: project description,
implementation, technologies and their roles, tests and conclusions, possible
applications, and degree of achievement. Preserve substantive author content
and material limitations while rewriting for clarity. Never invent results,
references, approval or a successful check.

Finish:
- Build using the report's existing toolchain. Inspect all pages after a
  structural reflow, including tables, diagrams, Polish glyphs, bibliography,
  references and page breaks. Fix compilation and layout defects.
- Update report-state.json and evidence-map/provenance files to the exact
  application source and evidence used. Keep technical provenance outside
  the PDF's narrative.
- Produce the current PDF and a portable source ZIP compatible with Overleaf.
- Commit and push the verified sources to the existing report remote under
  the user's publication authorization. Preserve concurrent work and use
  ordinary Git operations. Report actual publication and compilation status.
- Record a concise update report in the human's notes directory. Return
  the report commit, covered application commit, PDF and source ZIP paths,
  build/visual results, actual page count and substantive remaining gaps.
