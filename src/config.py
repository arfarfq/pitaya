"""Load processing defaults from the repository's configuration file."""

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    DEFAULTS = yaml.safe_load(config_file)
