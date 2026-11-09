import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    artifact_dir: Path
    model_id: str
    model_revision: str = "main"
    seed: int = 2027
    max_length: int = 512
    atol: float = 1e-4
    rtol: float = 1e-4


def load_config(path: Path) -> Config:
    path = path.resolve()
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    values["artifact_dir"] = (path.parent / values["artifact_dir"]).resolve()
    config = Config(**values)
    if not 2 <= config.max_length <= 512:
        raise ValueError("max_length must be between 2 and 512, including special tokens")
    if config.atol < 0 or config.rtol < 0:
        raise ValueError("Output tolerances must be nonnegative")
    return config
