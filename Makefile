# Hide-and-seek — common workflows. Run inside a Python 3.10+ env with the
# package installed (`pip install -e .`). Override PY to point at a specific one.
PY ?= python
CONFIG ?= configs/default.yaml
RUN ?= outputs/latest

.PHONY: help setup smoke train match watch video test tb clean

help:
	@echo "setup    install the package (editable) into the active env"
	@echo "smoke    tiny self-play run (pipeline sanity, ~1 min)"
	@echo "train    full self-play training      (CONFIG=$(CONFIG))"
	@echo "match    run matches, report blue/red win rates   (RUN=path)"
	@echo "watch    watch matches in the interactive viewer  (RUN=path)"
	@echo "video    record a match mp4 (headless)            (RUN=path)"
	@echo "test     run the test suite"
	@echo "tb       tensorboard on outputs/"

setup:
	$(PY) -m pip install -e ".[dev]"

smoke:
	$(PY) -m hideseek.train --smoke

train:
	$(PY) -m hideseek.train --config $(CONFIG)

match:
	$(PY) -m hideseek.match --run $(RUN) --matches 20

watch:
	$(PY) -m hideseek.visualize --run $(RUN)$(if $(SEED), --seed $(SEED))$(if $(PRESET), --preset $(PRESET))

video:
	MUJOCO_GL=egl $(PY) -m hideseek.visualize --run $(RUN) --video $(RUN)/match.mp4

test:
	$(PY) -m pytest

tb:
	tensorboard --logdir outputs

clean:
	rm -rf outputs/* **/__pycache__ .pytest_cache
