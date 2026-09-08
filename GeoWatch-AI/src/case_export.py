from __future__ import annotations

import io
import json
import zipfile

from .case_service import CaseStore
from .image_store import AnalysisImageStore
from .reporting import DISCLAIMER
from .review_service import ReviewStore
from .temporal_store import TemporalStore


def export_case_bundle(
    case_id: str,
    *,
    cases: CaseStore,
    analyses: ReviewStore,
    timelines: TemporalStore,
    images: AnalysisImageStore,
) -> bytes:
    """Build a portable, reviewable case bundle from persisted local evidence."""
    case = cases.load(case_id)
    comparison_ids = timelines.comparisons_for_case(case_id)
    comparisons = [timelines.load(item) for item in comparison_ids]
    analysis_ids = list(case.analysis_ids)
    for comparison in comparisons:
        for analysis_id in (comparison.before_analysis_id, comparison.after_analysis_id):
            if analysis_id not in analysis_ids:
                analysis_ids.append(analysis_id)

    loaded_analyses = []
    missing_analyses = []
    for analysis_id in analysis_ids:
        try:
            loaded_analyses.append(analyses.load_by_analysis(analysis_id))
        except KeyError:
            missing_analyses.append(analysis_id)

    manifest = {
        "schema": "geowatch-case-bundle-v1",
        "human_review_required": True,
        "case_id": case.case_id,
        "analysis_ids": analysis_ids,
        "comparison_ids": comparison_ids,
        "missing_analysis_ids": missing_analyses,
        "disclaimer": DISCLAIMER,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("case.json", case.model_dump_json(indent=2))
        for comparison in comparisons:
            archive.writestr(
                f"comparisons/{comparison.comparison_id}.json",
                json.dumps(comparison.to_dict(), ensure_ascii=False, indent=2),
            )
        for analysis in loaded_analyses:
            archive.writestr(f"analyses/{analysis.analysis_id}.json", analysis.model_dump_json(indent=2))
            preview = images.get(analysis.analysis_id)
            if preview is not None:
                archive.write(preview, f"evidence/{analysis.analysis_id}.png")
    return output.getvalue()
