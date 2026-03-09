# eKYC Quick Eval

This script runs a minimal contrast evaluation against the live `/api/ekyc/evaluate` endpoint.
It does not train or modify models. It only sends HTTP requests and aggregates results.

## Data Layout

```
data/quick_eval/
  person_001/
    id.jpg
    selfie.jpg
    video.mp4
  person_002/
    id.jpg
    selfie.jpg
    video.mp4
```

Only `person_*` directories with all three files are used.

## Run

Start the backend first, then run:

```bash
python scripts/quick_eval_ekyc.py --data_dir data/quick_eval --out_dir outputs --base_url http://127.0.0.1:8000
```

Outputs:
- `outputs/quick_eval_results.csv`
- Summary counts printed to console

## What It Generates

For each person `i`:
- **pos**: `(id_i, selfie_i, video_i)`
- **neg_id**: `(id_i, selfie_j, video_j)`
- **neg_selfie**: `(id_j, selfie_i, video_j)`
- **neg_video**: `(id_j, selfie_j, video_i)`

where `j = (i + 1) mod N`.

## Notes

- The script uses Python standard library only (no extra dependencies).
- Failures are logged in the CSV via `error_code` and `error_message`.
- If you have large videos, the multipart body is built in memory; keep clips short.
