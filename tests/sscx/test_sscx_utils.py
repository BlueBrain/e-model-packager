"""Contains tests for the workflow."""
"""
Copyright 2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

from e_model_packager.sscx2020.utils import (
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
