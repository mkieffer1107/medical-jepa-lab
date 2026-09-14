from pathlib import Path

import pytest

from medjepa.config import load_config


@pytest.mark.parametrize(
    "name,algorithm",
    [
        ("ijepa_bloodmnist_128.yaml", "ijepa"),
        ("lejepa_bloodmnist_128.yaml", "lejepa"),
        ("smoke_ijepa_synthetic.yaml", "ijepa"),
        ("smoke_lejepa_synthetic.yaml", "lejepa"),
    ],
)
def test_configs_load(name: str, algorithm: str) -> None:
    cfg = load_config(Path("configs") / name)
    assert cfg.algorithm == algorithm
    assert cfg.data.image_size % cfg.model.patch_size == 0


def test_dotted_override() -> None:
    cfg = load_config(
        "configs/smoke_ijepa_synthetic.yaml",
        ["data.batch_size=3", "experiment.run_name=override-test"],
    )
    assert cfg.data.batch_size == 3
    assert cfg.experiment.run_name == "override-test"
