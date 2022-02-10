"""Contains tests for the workflow."""

from pathlib import Path

from e_model_packages.sscx2020.utils import (
    LocalTargetCustom,
)


def test_localtargetcustom():
    """Test exist function of localtargetcustom."""
    example_dir = Path("tests") / "sscx" / "data" / "output_example"

    filename = "empty_file.Step_150.soma.v.dat"
    target = LocalTargetCustom(example_dir / filename)
    assert target.exists()

    filename = "not_existent_file.Step_150.soma.v.dat"
    target = LocalTargetCustom(example_dir / filename)
    assert not target.exists()

    filename = "*.Step_150.soma.v.dat"
    target = LocalTargetCustom(example_dir / filename)
    assert target.exists()

    filename = "not_*.Step_150.soma.v.dat"
    target = LocalTargetCustom(example_dir / filename)
    assert not target.exists()
