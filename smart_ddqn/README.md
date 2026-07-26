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
                                   - target-object encoder
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
- `DDQNQNetwork.emb_obj_encoder`;
- `DDQNQNetwork.q_head`.

The online and target DDQN networks are separate instances of this one class.

## What auxiliary learning trains

Only:

- `GraphEmbeddingModule` via `graph_optimizer`;
- `OrientationModule` via `orientation_optimizer`.

Both receive gradients from the supervised orientation loss sampled from replay
states. The GT angle is stored in `observation["angle_to_goal"]`.

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
    "observation": float_array[512],          # CLIP image embedding
    "absolute_goal_position": float_array[2], # Habitat X-Z goal position
    "goal_description": float_array[512],     # CLIP target-object embedding
    "knowledge_graph": float_array[186],      # 31 nodes x 6 fields
    "angle_to_goal": float_or_array[1],       # relative angle in radians
}
```

Exact source names and dimensions are configured in `config.json`.

## Run

```bash
python path/to/simple_habitat_ddqn/main.py \
  --config path/to/simple_habitat_ddqn/config.json
```

## Time policy inference on 100 episodes

`time_policy_inference.py` evaluates the policy on exactly 100 completed
episodes. It separately reports policy forward-pass time and
end-to-end rollout time; environment/model startup, checkpoint loading, and the
initial reset are excluded. Flushed stage messages show startup progress, and
episode progress is printed from `0/100` through `100/100`; a heartbeat is
printed every 10 seconds when no episode completes. The final console and JSON
summaries include Habitat's mean SPL (Success weighted by Path Length) over all
100 episodes.

From the repository root:

```bash
python smart_ddqn/time_policy_inference.py \
  --config smart_ddqn/config_chpt.json \
  --json-out policy_timing.json
```

The default `stage2` profile matches `smart_ddqn/main.py`: it uses the debug
split, the same fixated-target filter, 32 shuffled workers, and the configured
final epsilon. It automatically selects the latest step for which both a DDQN
and perception checkpoint exist. Use `--checkpoint-step 10000` to evaluate an
older matching pair.

The timing run is policy-only: it does not substitute curriculum
expert/controller actions. Also note that the training
`policy_success_rate` is a rolling window, while this script reports a fresh
100-episode sample. Executed action counts are included to expose policies that
rarely or never issue `stop`.

For a deterministic evaluation on the unfiltered held-out split, use:

```bash
python smart_ddqn/time_policy_inference.py \
  --profile heldout \
  --epsilon 0
```

Override worker count with `--num-envs`; per-worker episode quotas always sum to
exactly 100. Explicit checkpoint overrides must provide both
`--agent-checkpoint` and `--perception-checkpoint`, and their numbered steps
must match.

## Remaining Habitat TODOs

Search for `# TODO:`. The unresolved integration points are:

- actual `ObjRLNav`, `CurriculumVectorEnv`, and `CLIPWrapper` imports/signatures;
- actual observation keys and dimensions;
- exact discrete action IDs and payload format;
- autoreset behavior of completed vector workers;
- timeout/truncation information.
