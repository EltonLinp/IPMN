"""
Trust certificate generation utilities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    REPORTLAB_AVAILABLE = False
    A4 = None  # type: ignore
    mm = None  # type: ignore
    canvas = None  # type: ignore

from ..config import ReportConfig
from ..utils.logger import get_logger
from .risk_analysis import RiskAssessment

LOGGER = get_logger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class TrustReportGenerator:
    """
    Render machine-readable and human-readable trust certificates.
    """

    def __init__(self, config: ReportConfig) -> None:
        self.config = config
        _ensure_dir(self.config.output_dir)

    def _build_payload(self, assessment: RiskAssessment, session_id: str) -> Dict[str, object]:
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            "session_id": session_id,
            "issued_at": timestamp,
            "trust_index": assessment.trust_index,
            "risk_level": assessment.risk_level,
            "modality_contributions": assessment.modality_contributions,
            "alerts": assessment.alerts,
            "issuer": self.config.issuer,
            "organisation": self.config.organisation,
        }

    def write_json(self, assessment: RiskAssessment, session_id: str) -> Path:
        payload = self._build_payload(assessment, session_id)
        json_path = self.config.output_dir / f"{session_id}_certificate.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        LOGGER.info("Saved trust certificate JSON to %s", json_path)
        return json_path

    def write_pdf(self, assessment: RiskAssessment, session_id: str) -> Path:
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab 未安装，无法生成 PDF 证书。")
        pdf_path = self.config.output_dir / f"{session_id}_certificate.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        margin = 20 * mm
        y = height - margin

        def draw_line(text: str) -> None:
            nonlocal y
            c.drawString(margin, y, text)
            y -= 12

        draw_line(f"TrustFusion Trust Certificate")
        draw_line(f"Session ID: {session_id}")
        draw_line(f"Issued At: {datetime.now(timezone.utc).isoformat()}")
        draw_line(f"Issuer: {self.config.issuer} @ {self.config.organisation}")
        draw_line(f"Trust Index: {assessment.trust_index:.3f}")
        draw_line(f"Risk Level: {assessment.risk_level.upper()}")
        draw_line("")
        draw_line("Modality Contributions:")
        for modality, score in assessment.modality_contributions.items():
            draw_line(f" - {modality}: {score:.3f}")
        if assessment.alerts:
            draw_line("")
            draw_line("Alerts:")
            for alert in assessment.alerts:
                draw_line(f" * {alert}")
        c.showPage()
        c.save()
        LOGGER.info("Saved trust certificate PDF to %s", pdf_path)
        return pdf_path

    def generate(self, assessment: RiskAssessment, session_id: str) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}
        if self.config.include_json:
            outputs["json"] = self.write_json(assessment, session_id)
        if self.config.include_pdf:
            outputs["pdf"] = self.write_pdf(assessment, session_id)
        return outputs
