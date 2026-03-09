from __future__ import annotations

from pathlib import Path

from training import train_audio as _audio


_ORIG_PARSE_ARGS = _audio.parse_args


def _patched_parse_args():
    args = _ORIG_PARSE_ARGS()
    args.use_sync_fusion = True
    if args.save_path == Path("wavlm_audio.pt"):
        args.save_path = Path("wavlm_audio_sync.pt")
    if getattr(args, "resume_from", None) in (None, Path("wavlm_audio.pt")):
        args.resume_from = args.save_path
    if args.sync_checkpoint is None:
        args.sync_checkpoint = Path("res/sync_module.pt")
    if args.sync_vit_path is None:
        args.sync_vit_path = Path("/root/autodl-tmp/Industrial_Project/vit_model")
    return args


_audio.parse_args = _patched_parse_args


def main():
    _audio.train()


if __name__ == "__main__":
    main()
