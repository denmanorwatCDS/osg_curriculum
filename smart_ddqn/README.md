# Simplified Habitat + skrl DDQN

This package retains only the essential pipeline:

1. Habitat vector environment produces observations.
2. `HabitatGymnasiumAdapter` exposes a Gymnasium-compatible discrete environment.
3. skrl DDQN stores the full observation in `RandomMemory` and trains the Q-network.
4. The Q-network uses image, graph, goal and **predicted** orientation.
5. `teacher_orientation` is ignored by the policy but remains in replay memory.
6. The auxiliary trainer samples replay states and trains the online perception module against GT yaw.

## Required observation contract

After `CLIPWrapper`, every worker must expose values equivalent to:

```python
{
    "clip": float array [img_dim],
    "goal": float array [goal_dim],
    "graph": float array [graph_dim],
    "orientation": float array/scalar [1],  # yaw in radians
}
```

The exact source names and dimensions are configured in `config.json`.

## Run

From the project root:

```bash
python path/to/simple_habitat_ddqn/main.py \
  --config path/to/simple_habitat_ddqn/config.json
```

## Mandatory checks

Search for `# TODO:` in the Python files. The critical checks are:

- actual imports and constructor signatures of `ObjRLNav`, `CurriculumVectorEnv`, and `CLIPWrapper`;
- actual observation keys and flattened dimensions;
- exact discrete action IDs/payloads;
- whether the vector environment auto-resets completed workers;
- whether Habitat distinguishes timeout truncation in `info`.

## Deliberate simplifications

- one Q-network class, instantiated once for online and once for target roles;
- no PyG/GAT scene graph model;
- no image normalization/preprocessor;
- no curriculum-specific logic in the trainer;
- no custom metric aggregation;
- no separate teacher dataset: teacher targets come directly from replayed observations.
