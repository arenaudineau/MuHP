import numpy as np
from numpy import ma
import json

from typing import Any, Callable, Iterable, Sized, Sequence, Literal
from types import MappingProxyType
from pathlib import Path
import os
import shutil
import subprocess
from functools import partial
from inspect import signature

__all__ = ["MuHP"]


class MuHP:
    def __init__(
        self, name: str | None = None, config: dict[str, Any] | Literal["load"] = {}
    ):
        self.name = name or input("Please give a name for the run: ")

        self.path = Path(f"./muhp/{self.name}")

        self._lapse_metrics: dict[str, Any] = {}
        self._metrics: dict[str, Sequence] = {}
        self._latest_stored_metrics: dict[str, int] = {}

        self._lapse_idx: int = 0
        self._lapse_count: int | None = None
        self.save_frequency: int = 0

        if config == "load":
            if (
                not self.path.exists()
                or not (self.path / "_completed_sentinel").exists()
            ):
                raise ValueError(f"Trying to load non-completed run '{self.name}'")

            with open(self.path / "config.json", "r") as f:
                self.config = json.load(f)
            self._lapse_idx = -1  # Already completed run
        else:
            if self.path.exists() and not self.path.stem.startswith("!"):
                if (self.path / "_completed_sentinel").exists():
                    raise ValueError(f"Already completed run '{self.name}'")

                content = [
                    (self.path / elem, self.path / "_backup" / elem)
                    for elem in os.listdir(self.path)
                ]

                (self.path / "_backup").mkdir()
                for src, dst in content:
                    if src.is_dir():
                        dst.mkdir(parents=True)
                    shutil.move(src, dst)

            self.path.mkdir(parents=True, exist_ok=True)

            if Path(".git").exists():
                git_commit_hash = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .decode("ascii")
                    .strip()
                )
                git_diff = subprocess.check_output(["git", "diff"]).decode()

                with open(self.path / "gitdiff.patch", "w") as f:
                    f.write("# current-commit: " + git_commit_hash + "\n\n")
                    f.write(git_diff)

            self.config = config

    def log(self, name, value, *, list_to_multiple_logs=False):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log(
                    name + "." + str(k),
                    v,
                    list_to_multiple_logs=list_to_multiple_logs + 1,
                )

            return

        elif list_to_multiple_logs and isinstance(value, list):
            if isinstance(list_to_multiple_logs, int):
                list_to_multiple_logs -= 1

            for k, v in enumerate(value):
                self.log(
                    name + "." + str(k), v, list_to_multiple_logs=list_to_multiple_logs
                )

            return

        if self._lapse_metrics.get(name) is not None:
            raise ValueError(
                f"Multiple '{name}' for the same lapse. Previously stored value: {self._lapse_metrics[name]}"
            )

        value = np.asarray(value)
        self._lapse_metrics[name] = value
        self._latest_stored_metrics[name] = self._lapse_idx

        if name not in self._metrics:
            self._metrics[name] = (
                []
                if self._lapse_count is None
                else ma.array(
                    np.empty((self._lapse_count,) + value.shape, dtype=value.dtype),
                    mask=True,
                )
            )

    def lapse(self):
        if self._lapse_idx < 0:
            raise ValueError(
                "Experiment already ran, please create a new MuHP instance"
            )

        for k in self._lapse_metrics.keys():
            if self._lapse_metrics[k] is None:
                continue

            v = self._metrics[k]
            if isinstance(v, np.ndarray):
                v[self._lapse_idx] = self._lapse_metrics[k]
            else:
                v.append(self._lapse_metrics[k])

            self._lapse_metrics[k] = None

        self._lapse_idx += 1

        if self._lapse_count is not None and self._lapse_idx == self._lapse_count:
            self.finalize()

        elif self.save_frequency != 0 and self._lapse_idx % self.save_frequency == 0:
            self._save_metrics(self.path / "metrics_lapse_running")

    def lapsed(self, iter: Iterable, *, size=None, step_size=1, with_init=False):
        return _MuHPGenerator(
            self, iter, size=size, step_size=step_size, with_init=with_init
        )

    def finalize(self):
        self._save_metrics(self.path / "metrics_lapse_final")
        self._lapse_idx = -1
        with open(self.path / "_completed_sentinel", "w"):
            pass

    @property
    def metrics(self):
        return MappingProxyType(self._metrics)

    @property
    def lapse_metrics(self):
        return MappingProxyType(self._lapse_metrics)

    @property
    def latest_metrics(self):
        def _get(k):
            if self._lapse_metrics[k] is None:
                v = self._metrics[k]
                if isinstance(v, np.ndarray):
                    v = v[~v.mask]

                if len(v) == 0:
                    v = None
                else:
                    v = np.asarray(v[-1])
            else:
                v = self._lapse_metrics[k]
            return v

        return MappingProxyType({k: _get(k) for k in self._lapse_metrics.keys()})

    @property
    def config(self):
        return MappingProxyType(self._config)

    @config.setter
    def config(self, conf):
        self._config = conf
        self._update_json_config()

    @property
    def lapse_count(self):
        return self._lapse_count

    def _save_metrics(self, path):
        def _unmask(x):
            if isinstance(x, ma.MaskedArray):
                if np.issubdtype(x.dtype, np.floating):
                    return x.filled(np.nan)
                elif not x.mask.any():
                    return np.asarray(x)
                return x
            else:
                return x

        np.savez_compressed(
            path, **{k: _unmask(v) for k, v in self.metrics.items()}, allow_pickle=True
        )

    def _update_json_config(self):
        def value_to_printable(v):
            if isinstance(v, type):
                return v.__qualname__
            elif isinstance(v, partial):
                inner = v.func.__qualname__
                args = signature(v.func).bind_partial(*v.args, **v.keywords).arguments
                args_str = [f"{k}={v}" for k, v in args.items()]
                return f"{inner}({', '.join(args_str)})"
            elif isinstance(v, Callable):
                return v.__qualname__
            else:
                return v

        if self._lapse_idx != 0:
            raise ValueError("Cannot change the configuration once the run has started")

        printable_config = {k: value_to_printable(v) for k, v in self._config.items()}
        with open(self.path / "config.json", "w") as f:
            json.dump(printable_config, f, indent=2)

    def __getitem__(self, x):
        return self.config.get(x)

    def __setitem__(self, name, v):
        self._config[name] = v
        self._update_json_config()

    def __getattr__(self, x):
        return self[x]

    def __setattr__(self, name, v):
        if name in ["config", "path", "name", "save_frequency"] or name[0] == "_":
            super().__setattr__(name, v)
        else:
            self[name] = v


class _MuHPGenerator:
    def __init__(
        self, instance: MuHP, it: Iterable, size=None, step_size=1, with_init=False
    ):
        self.instance = instance
        self.iter = enumerate(it)

        if self.instance._lapse_idx < 0:
            raise ValueError(
                "Experiment already ran, please create a new MuHP instance"
            )

        if isinstance(it, Sized):
            self.instance._lapse_count = len(it) // step_size + int(with_init)
            self.size = len(it)
        elif size is not None:
            self.instance._lapse_count = size // step_size + int(with_init)
            self.size = size

        self.step_size = step_size
        self.with_init = with_init

    def __iter__(self):
        for i, x in self.iter:
            yield x

            if (i + 1) % self.step_size == 0 or (self.with_init and i == 0):
                self.instance.lapse()

        self._lapse_count = None

    def __len__(self):
        return self.size
