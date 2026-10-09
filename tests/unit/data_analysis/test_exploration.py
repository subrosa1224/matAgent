from __future__ import annotations

from pathlib import Path

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.exploration import (
    DataExplorationService,
    render_exploration_markdown,
)
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService


def test_one_click_exploration_is_read_only_and_recommends_next_steps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment.csv"
    source.write_text(
        "sample,group,x,y,constant\n"
        "a,A,1,2,yes\n"
        "b,A,2,4,yes\n"
        "b,A,2,4,yes\n"
        "c,B,3,,yes\n"
        "d,B,4,8,yes\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    store = DatasetStore(tmp_path / "private")
    dataset = store.register_file(
        source, source_artifact_id="artifact-data-exploration"
    )
    bundle = DataExplorationService(
        DataAnalysisService(store), DataStatisticsService(store)
    ).explore(dataset.dataset_id)
    rendered = render_exploration_markdown(bundle)

    assert bundle.descriptive is not None
    assert bundle.correlation is not None
    assert len(bundle.analysis_ids) == 2
    assert "缺失" in rendered
    assert "重复" in rendered
    assert "相关性不代表因果关系" in rendered
    assert source.read_bytes() == original
    assert store.get(dataset.dataset_id).parent_dataset_id is None
