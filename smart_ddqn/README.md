# Simplified Habitat + skrl DDQN

This package uses two strictly separated optimization paths.

```text
Habitat observation
        |
        +------------------------------+
        |                              |
        v                              v
External perception               DDQN network
- GraphEmbeddingModule            - DDQN image encoder
- OrientationModule               - goal encoder
        |                          - Q head
        | detached features             |
        +------------------------------> Q-values

Replay states
    -> GT teacher orientation
    -> auxiliary loss
    -> graph optimizer + orientation optimizer
```

## Hard separation guarantee

`GraphEmbeddingModule` and `OrientationModule` are not children of either DDQN
Q-network. `DDQNQNetwork` keeps only a weak reference to the external
`AuxPerceptionModules` object and requests features under `torch.no_grad()`.

At startup `trainer.py` checks parameter identities and prints:

```text
[SEPARATION] ... overlap=0
```

Any overlap raises `RuntimeError` before training begins.

Consequences:

- DDQN TD-loss cannot update graph embedding or orientation parameters;
- auxiliary optimizers cannot update DDQN parameters;
- the DDQN checkpoint does not contain perception;
- the perception checkpoint does not contain DDQN;
- graph/orientation features supplied to both online and target Q-networks are
  detached.

## What DDQN trains

Only:

- `DDQNQNetwork.img_encoder`;
- `DDQNQNetwork.goal_encoder`;
- `DDQNQNetwork.q_head`.

The online and target DDQN networks are separate instances of this one class.

## What auxiliary learning trains

Only:

- `GraphEmbeddingModule` via `graph_optimizer`;
- `OrientationModule` via `orientation_optimizer`.

Both receive gradients from the supervised orientation loss sampled from replay
states. The GT yaw is stored in `observation["teacher_orientation"]`.

## GT or predicted orientation

Set `model.orientation_source`:

- `"gt"`: DDQN receives `[sin(gt_yaw), cos(gt_yaw)]` from the observation;
- `"pred"`: DDQN receives the detached prediction of `OrientationModule`.

In both modes the graph embedding is external and detached. Auxiliary training
may remain active in both modes.

## Separate checkpoints

DDQN checkpoint:

```json
"run": {
  "agent_checkpoint": "/path/to/ddqn_checkpoint.pt"
}
```

Perception checkpoint:

```json
"aux": {
  "resume_from": "/path/to/perception_latest.pt"
}
```

For evaluation with `orientation_source="pred"`, load both matching checkpoints.
Old combined perception checkpoints are intentionally rejected because their
parameter ownership is ambiguous.

## Required observation contract

After Habitat wrappers, every worker must expose equivalents of:

```python
{
    "clip": float_array[img_dim],
    "goal": float_array[goal_dim],
    "graph": float_array[graph_dim],
    "orientation": float_or_array[1],  # yaw in radians
}
```

Exact source names and dimensions are configured in `config.json`.

## Run

```bash
python path/to/simple_habitat_ddqn/main.py \
  --config path/to/simple_habitat_ddqn/config.json
```

## Remaining Habitat TODOs

Search for `# TODO:`. The unresolved integration points are:

- actual `ObjRLNav`, `CurriculumVectorEnv`, and `CLIPWrapper` imports/signatures;
- actual observation keys and dimensions;
- exact discrete action IDs and payload format;
- autoreset behavior of completed vector workers;
- timeout/truncation information.
