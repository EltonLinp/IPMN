# Celeb-DF v2 Split & Subject Policy

All data is sourced from `E:\CUHK\Industrial_Project\Celeb-DF-v2`.  
Every video is labelled by its parent folder (`Celeb-synthesis` → fake, `Celeb-real`/`YouTube-real` → real) and assigned a stable **subject identifier** using the following priority:

1. **Metadata override** – if ancillary metadata provides `subject` / `person_id`, use it.
2. **Filename pattern** – detect tokens like `id1234` (case-insensitive) and normalise to `id1234`.
3. **Hash fallback** – when the filename exposes no ID, hash the filename (SHA1, first 10 hex chars) and prefix with `hash_`.

Splitting is **stratified 80/20** by label:

- `tools/generate_celebdf_split.py` groups videos by label, shuffles with `seed=42`, then allocates 80 % of each label to `train.csv` and the remaining 20 % to `val.csv`.  
- This preserves the real/fake ratio across both splits while maintaining reproducibility.  
- The combined manifest (`celebdf_manifest.csv`) records `video_path`, `label`, `subject`, and the assigned `split`.

Downstream components (preprocessing, training) rely on these manifests to ensure consistent label balance between training and validation.
