# PR0 evidence snapshot

This directory versions the inventory, development-only priors, frozen selections,
baseline summaries, provenance, failure tokens, repeat comparisons and verification
counts. The complete analysis is in [PR0_REPORT.md](../../docs/autofit/PR0_REPORT.md).

Large generated files are kept out of Git:

- `nuscenes_manifest.json.gz`: rebuild from the local read-only dataset. Its exact
  expected SHA256 and size are recorded in `evidence-index.json`.
- `*/cases.jsonl`, `*/summary.json`, `*/cold.json`: six original baseline/replay
  ledgers, full per-stratum statistics and cold-response diagnostics are preserved,
  byte for byte, in `raw-evidence.tar.gz`. Compact `overview.json` files remain
  readable in Git. Extract before running comparison or replay tools.
- `test-logs/`: local runner output. Versioned `verification.json` records commands,
  counts and unexecuted suites; it does not claim browser acceptance.
- `local-runs/`: destination for future scratch measurements. A new reviewed
  baseline must be promoted explicitly rather than replacing this snapshot.

Only the final CPU replay pair, `run-3` and `run-4`, is retained. Earlier `run-1` and
`run-2` were intermediate source-verification runs with the same reported outcomes;
they are not needed to reproduce or assess the final pair. The canonical original
GPU comparison is `original-repeatability-check.json`; the fixed comparison is
`fixed-repeatability-check.json`. No measured JSON or ledger bytes were rewritten
during cleanup.

## Restore the recorded ledgers

From the repository root, verify the committed archive and its contents:

```bash
python3 - <<'PY'
import hashlib, json, pathlib, tarfile
root = pathlib.Path('artifacts/autofit-eval')
index = json.loads((root / 'evidence-index.json').read_text())
archive = root / index['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == index['archive_sha256']
with tarfile.open(archive, 'r:gz') as bundle:
    assert sorted(bundle.getnames()) == sorted(item['path'] for item in index['entries'])
    for item in index['entries']:
        payload = bundle.extractfile(item['path']).read()
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
print('All original evidence files verified')
PY
tar -xzf artifacts/autofit-eval/raw-evidence.tar.gz -C artifacts/autofit-eval
```

Extraction restores the original filenames expected by existing tools and recorded
hashes. The restored files are ignored, so they do not dirty the worktree.

## Rebuild the full manifest

The source dataset is not included. With the original dataset location available:

```bash
python3 -m components.vehicle_fit.evaluation.nuscenes_dataset \
  --dataroot "$HOME/Datasets/nuScenes" --version v1.0-mini \
  --seed 20261016 --created-at 2026-10-16 --output /tmp/pr0-dataset-rebuild
```

Use a fresh output directory. Check the generated manifest against the committed
index before restoring it:

```bash
python3 - <<'PY'
import hashlib, json, pathlib, shutil
root = pathlib.Path('artifacts/autofit-eval')
expected = json.loads((root / 'evidence-index.json').read_text())['regenerated_manifest']
source = pathlib.Path('/tmp/pr0-dataset-rebuild') / expected['path']
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected['sha256']
shutil.copyfile(source, root / expected['path'])
PY
```

The current manifest includes the absolute dataroot. Byte reproducibility was
verified twice at the recorded location; relocation changes that hash. Do not
silently replace a locked hash to run acceptance. A relocated dataset needs a new
manifest/selection version and explicit provenance, or the original manifest
restored from a local copy with evaluator `--dataroot` overriding file access.

The recorded `git_commit` remains the HEAD at measurement time. Uncommitted PR0
source was identified by per-file/script hashes. Committing this snapshot does not
retroactively change the historical measurement identity.
