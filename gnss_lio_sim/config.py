"""YAML yapılandırma yükleyici (nokta erişimli sözlük)."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


class Cfg(dict):
    """`cfg.imu.rate` biçiminde erişime izin veren sözlük."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __deepcopy__(self, memo):
        return Cfg(copy.deepcopy(dict(self), memo))


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None):
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            _merge(data, yaml.safe_load(f) or {})
    if overrides:
        _merge(data, overrides)
    return _wrap(data)


def to_plain(cfg):
    if isinstance(cfg, dict):
        return {k: to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [to_plain(v) for v in cfg]
    return cfg
