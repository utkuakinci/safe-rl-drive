# CARLA demo (sim-to-sim transfer)

The policy is trained only in highway-env. This demo re-creates the observation contract
(`safe_rl_drive/envs/spec.py`, also embedded in the ONNX metadata) from CARLA ground truth, runs the
**same ONNX artefact** that the ROS2 node uses, applies the safety shield, and maps discrete
meta-actions to a pure-pursuit + PI controller.

```bash
# terminal 1 - CARLA 0.9.15 server
./CarlaUE4.sh -RenderOffScreen -quality-level=Low

# terminal 2
pip install -e ".[deploy,carla]"
python carla_demo/run_demo.py --onnx models/rppo_highway.onnx --shield --seconds 90 --npc 60 \
    --out results/carla_demo.mp4
```

Output: `results/carla_demo.mp4` (chase camera + policy action, executed action, shield reasons) and
`results/carla_demo.json` (decisions, interventions, collisions).

Interface details worth knowing:

* highway-env gives the **ego row in absolute coordinates** (x clipped to +1 after ~100 m, y = lane
  index x 4 m). The demo reproduces exactly that; other vehicles are relative and expressed in the
  lane-aligned frame (highway-env roads are aligned with the x-axis).
* CARLA's left-handed frame (y to the right) matches highway-env's y-down convention, so the lateral
  sign of LANE_LEFT / LANE_RIGHT is preserved.
* Decisions run at 2 Hz on 20 Hz physics, as in training.
