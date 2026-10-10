# Dataset backups

Archived copies of the mapping CSVs, kept so a trained model can be matched to the data it
actually saw after that data has been rebuilt.

Only this file is tracked by git. The archives themselves are ignored — they are large and
regenerable in principle, but regenerating one costs hours and the result is not guaranteed
to be byte-identical, which is the whole reason for keeping them.

## When to make one

**Before anything overwrites a mapping CSV.** In practice that means before re-running
`10_precompute_rt60.py`, `11_precompute_rir_metrics.py`, `05d_void_log.py`, or either
`02_preprocess_*` script, whenever the change alters values an already-trained model
depends on. A new column that nothing reads yet does not need one; a changed `rt60`, a
changed filter column, or a rebuilt row set does.

## How a model is matched to its data

Every run records the mapping's md5 in `run_config.json` under `inputs.mapping_csv.md5`.
Evaluation compares it against the file on disk and says so when they differ. The table
below maps each md5 to the archive holding that version, so the answer to "what did this
model train on" is a lookup rather than a guess.

The date in a filename is for humans. **The md5 is what identifies a version.**

## Making one

From the project root, inside the container:

```bash
SRC=data/matterport3d/soundspaces_matterport3d_mapping.csv
DST=data/backups/$(basename "${SRC%.csv}").$(date +%F).csv.gz

md5sum "$SRC"                      # record this in the table
wc -l < "$SRC"                     # and this
gzip -c "$SRC" > "$DST"
ls -lh "$DST"
```

Then add a row to the table and say in **Reason** what was about to change it. A row
without a reason is nearly useless a month later.

## Restoring one

```bash
gunzip -c data/backups/<archive>.csv.gz > data/matterport3d/soundspaces_matterport3d_mapping.csv
md5sum data/matterport3d/soundspaces_matterport3d_mapping.csv    # must match the table
```

Restoring does **not** restore the scene stats or analysis outputs derived from that
mapping. Re-run `12_dataset_analytics.py` if those matter.

## Archives

| Date | File | md5 | Rows | Size (gz) | Reason |
|------|------|-----|------|-----------|--------|
| 2026-10-10 | `soundspaces_matterport3d_mapping.2026-10-10.csv.gz` | `f3023ba5891161d919a40b77b414eac5` | 9,926,878 | 205 MB | Last state before the `rt60` column was rebuilt over the `[-15,-40]` window. This is the version the two Oct 6 Matterport3D runs were trained against, and the only copy of it once `10_precompute_rt60.py` is re-run. |
