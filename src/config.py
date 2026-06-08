from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any

@dataclass
class ExperimentConfig:
    n_iterations_global: int = 10
    n_iterations_graph: int = 2
    n_iterations_masks: int = 500
    mask_prob: float = 0.5


@dataclass
class ModelConfig:
    input_dim: int = 784
    n_epochs_mnist: int = 100
    lr_mnist: float = 1e-3
    batch_size_mnist: int = 256


@dataclass
class SurrogateConfig:
    n_epochs_surrogate: int = 3000
    lr_surrogate: float = 1e-3
    n_epochs_fcn: int = 1000
    lr_fcn: float = 1e-3
    fcn_dims: list[int] = field(default_factory=lambda: [10])
    batch_size_surrogate: int = 32


@dataclass
class DataConfig:
    data_dir: str = "./data"
    num_workers: int = 8


@dataclass
class OutputConfig:
    output_dir: str = "histories"


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # "auto" resolves to cuda if available, else cpu
    device: str = "auto"
    # set to an integer for reproducibility, None to disable
    seed: int | None = None


EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "small": {"n_iterations_graph": 2},     # ~15 edges
    "medium": {"n_iterations_graph": 5},    # ~30 edges
    "large": {"n_iterations_graph": 14},    # ~75 edges
    "xlarge": {"n_iterations_graph": 30},   # ~150 edges
}


def _parse_value(text: str) -> Any:
    """Parse a CLI string into an appropriate Python type."""
    lowered = text.lower()
    if lowered in ("null", "none"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    # list syntax, e.g. [10] or [10,20]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part.strip()) for part in inner.split(",")]

    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _set_override(cfg: Config, dotted_key: str, value: Any) -> None:
    """Set a (possibly nested) attribute on the config from a dotted key."""
    parts = dotted_key.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"Unknown config section: {dotted_key!r}")
        obj = getattr(obj, part)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"Unknown config key: {dotted_key!r}")
    setattr(obj, leaf, value)


def to_dict(cfg: Config) -> dict[str, Any]:
    """Recursively convert the dataclass config into a plain dict."""
    if is_dataclass(cfg):
        return asdict(cfg)
    return cfg  # pragma: no cover


def parse_args(argv: list[str] | None = None) -> Config:
    """
    Build a Config from hydra-style CLI overrides.

    Supported syntax:
        experiment=large
        experiment.n_iterations_global=3
        device=cpu
        seed=42
        surrogate.fcn_dims=[10,20]
    """
    if argv is None:
        argv = sys.argv[1:]

    cfg = Config()

    for token in argv:
        if "=" not in token:
            raise ValueError(
                f"Invalid override {token!r}. Expected form key=value."
            )
        key, raw_value = token.split("=", 1)
        key = key.strip()

        # Special case: experiment preset selection (experiment=<name>)
        if key == "experiment":
            preset = EXPERIMENT_PRESETS.get(raw_value)
            if preset is None:
                raise ValueError(
                    f"Unknown experiment preset {raw_value!r}. "
                    f"Choices: {sorted(EXPERIMENT_PRESETS)}"
                )
            for sub_key, sub_val in preset.items():
                setattr(cfg.experiment, sub_key, sub_val)
            continue

        _set_override(cfg, key, _parse_value(raw_value))

    return cfg
