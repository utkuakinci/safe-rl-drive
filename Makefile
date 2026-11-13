# Convenience targets (Linux / macOS / WSL). Every target is a thin wrapper around a documented CLI.
PY ?= python
SCENARIOS ?= highway merge intersection
RUN ?= runs/highway/rppo/seed_0

.PHONY: install lint test test-all smoke train train-pomdp sweep eval explain deploy shield-vectors ros2-build ros2-run aif metadrive carla

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[tracking,deploy,active-inference,dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest -m "not slow"

test-all:
	pytest

smoke:
	$(PY) -m safe_rl_drive.training.train --config configs/train/smoke.yaml --out runs_smoke --wandb-mode disabled
	$(PY) -m safe_rl_drive.training.train --config configs/train/smoke_rppo.yaml --out runs_smoke --wandb-mode disabled
	$(PY) -m safe_rl_drive.deploy.export_onnx --model runs_smoke/highway/smoke_rppo/seed_0 --out models/smoke_rppo.onnx

# Week 1: PPO vs RecurrentPPO baselines, 5 seeds each, all scenarios
train:
	for s in $(SCENARIOS); do for a in ppo rppo; do \
	  $(PY) -m safe_rl_drive.training.train --config configs/train/$${s}_$${a}.yaml --all-seeds; done; done

# Partially observable variant (velocities hidden): where recurrence should matter
train-pomdp:
	$(PY) -m safe_rl_drive.training.train --config configs/train/highway_ppo_pomdp.yaml --all-seeds
	$(PY) -m safe_rl_drive.training.train --config configs/train/highway_rppo_pomdp.yaml --all-seeds

sweep:
	wandb sweep configs/sweeps/highway_ppo.yaml

# Week 2: robustness suite x shield on/off, CIs across seeds
eval:
	for s in $(SCENARIOS); do \
	  $(PY) -m safe_rl_drive.evaluation.evaluate --runs runs/$$s --suite robustness --shield both --episodes 20 --out results/$$s; done

explain:
	$(PY) -m safe_rl_drive.explain.visualize --model $(RUN) --scenario highway --shield --out results/explain_highway.mp4

# Week 3: ONNX -> TensorRT -> benchmark table
deploy:
	bash scripts/deploy_pipeline.sh $(RUN) highway

shield-vectors:
	$(PY) scripts/gen_shield_vectors.py

ros2-build:
	docker build -f docker/ros2.Dockerfile -t safe-rl-drive/ros2 .

ros2-run:
	docker run --rm -v $(PWD)/models:/models -v $(PWD)/results:/app/results safe-rl-drive/ros2 \
	  ros2 launch rl_policy_node drive.launch.py model_path:=/models/rppo_highway.onnx episodes:=20

# Week 4
aif:
	$(PY) -m safe_rl_drive.active_inference.experiment --out results/active_inference

# Sim-to-sim transfer without a CARLA server (Docker, CPU only)
metadrive:
	docker build -f docker/metadrive.Dockerfile -t safe-rl-drive/metadrive .
	for m in ppo rppo; do docker run --rm -v $(PWD)/models:/app/models -v $(PWD)/results:/app/results \
	  safe-rl-drive/metadrive python sim2sim_demo/metadrive_demo.py --onnx models/$${m}_highway.onnx \
	  --episodes 20 --out results/metadrive/$$m; done

carla:
	$(PY) carla_demo/run_demo.py --onnx models/rppo_highway.onnx --shield --out results/carla_demo.mp4
