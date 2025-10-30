import numpy as np
import json

from typing import Any, Iterable, Sized, Sequence
from types import MappingProxyType
from pathlib import Path
import os
import shutil
import subprocess


class MuHP:
    _config: dict[str, Any] = {}

    _lapse_metrics: dict[str, Any] = {}
    _metrics: dict[str, Sequence] = {}

    _lapse_idx: int = 0
    _lapse_count: int | None = None
    _save_frequency: int = 0

    def __init__(self, name: str | None = None, config: dict[str, Any] = {}):
        self.name = name or input("Please give a name for the run: ")

        self.path = Path(f"./muhp/{self.name}")
        if self.path.exists() and self.name != "!dbg":
            if (self.path / "_completed_sentinel").exists():
                raise ValueError(f"Already completed run {self.name}")

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
            git_diff = subprocess.check_output(["git", "diff"]).decode("ascii")

            with open(self.path / "gitdiff.patch", "w") as f:
                f.write("# current-commit: " + git_commit_hash + "\n\n")
                f.write(git_diff)

        self.config = config

    def log(self, name, value):
        if self._lapse_metrics.get(name) is not None:
            raise ValueError(
                f"Multiple '{name}' for the same lapse. Previously stored value: {self._lapse_metrics[name]}"
            )

        value = np.asarray(value)
        self._lapse_metrics[name] = value

        if name not in self._metrics:
            self._metrics[name] = (
                []
                if self._lapse_count is None
                else np.empty((self._lapse_count,) + value.shape, dtype=value.dtype)
            )

    def lapse(self):
        if self._lapse_idx < 0:
            raise ValueError(
                "Experiment already ran, please create a new MuHP instance"
            )

        for k in self._lapse_metrics.keys():
            v = self._metrics[k]
            if isinstance(v, np.ndarray):
                v[self._lapse_idx] = self._lapse_metrics[k]
            else:
                v.append(self._lapse_metrics[k])

            self._lapse_metrics[k] = None

        self._lapse_idx += 1

        if self._lapse_count is not None and self._lapse_idx == self._lapse_count:
            self.finalize()

        elif self._save_frequency != 0 and self._lapse_idx % self._save_frequency == 0:
            fmt = (
                str(self._lapse_idx)
                if self._lapse_count is None
                else f"{self._lapsed_idx}-{self._lapse_count}"
            )
            np.savez_compressed(
                self.path / f"metrics_lapse_{fmt}",
                **self.metrics,
                allow_pickle=False,
            )

    def lapsed(self, iter: Iterable, *, size=None, step_size=1, with_init=False):
        return _MuHPGenerator(
            self, iter, size=size, step_size=step_size, with_init=with_init
        )

    def finalize(self):
        np.savez_compressed(
            self.path / "metrics_lapse_final",
            **self.metrics,
            allow_pickle=False,
        )
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
    def config(self):
        return MappingProxyType(self._config)

    @config.setter
    def config(self, conf):
        self._config = conf
        self._update_json_config()

    def _update_json_config(self):
        if self._lapse_idx != 0:
            raise ValueError("Cannot change the configuration once the run has started")

        printable_config = {
            k: v.__qualname__ if isinstance(v, type) else v
            for k, v in self._config.items()
        }
        with open(self.path / "config.json", "w") as f:
            json.dump(printable_config, f, indent=2)

    def __getattr__(self, x):
        return self.config.get(x)

    def __setattr__(self, name, v):
        if name in ["config", "path", "name"] or name[0] == "_":
            super().__setattr__(name, v)
        else:
            self._config[name] = v
            self._update_json_config()


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
