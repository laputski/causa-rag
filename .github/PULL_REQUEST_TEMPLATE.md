## What this changes

<!-- The behaviour that differs after this, in a sentence or two. -->

## Why

<!--
If this fixes a defect, say what the symptom looked like. A description of how
the failure presented is what lets the next person recognise it.
-->

## Checklist

- [ ] `make test` passes
- [ ] New behaviour is covered by a test at the layer that can actually decide it
- [ ] Documentation under `docs/` updated, if this changes something a reader was told
- [ ] `make doctor` still reports a healthy install, if this touches setup or services
- [ ] `make test-e2e` passes, if this touches the run, ingestion or diagnostics paths
