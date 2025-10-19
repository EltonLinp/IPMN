import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustfusion.core.pipeline import build_default_system


def test_pipeline_run_smoke():
    system = build_default_system()
    video_tensor = torch.rand(1, 3, 8, 32, 32)
    audio_tensor = torch.rand(1, 1, 64, 8)
    lip_feats = torch.rand(1, 8, 64)
    audio_feats = torch.rand(1, 8, 64)
    rppg_trace = torch.rand(1, 8)
    audio_energy = torch.rand(1, 8)

    result = system.run(
        video_tensor=video_tensor,
        audio_tensor=audio_tensor,
        lip_feats=lip_feats,
        audio_feats=audio_feats,
        rppg_trace=rppg_trace,
        audio_energy=audio_energy,
        sample_rate=160,
        generate_report=False,
    )

    assert result.risk_assessment.trust_index >= 0.0
    assert "json" not in result.report_paths

    summary = system.dataset_summary()
    assert "root_dir" in summary
    assert {"total", "real", "fake", "unknown"}.issubset(summary.keys())
