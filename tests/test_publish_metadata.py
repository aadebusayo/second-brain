from pathlib import Path
import tomllib


def test_pyproject_contains_publish_metadata():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "secondbrain"
    assert project["version"] == "0.1.0a1"
    assert project["readme"] == "README.md"
    assert project["license"] == "MIT"
    assert project["urls"]["Repository"].endswith("second-brain")
