# SFT data generation

Turn agent rollout trajectories into the SFT data format used by
[Jianwen/SkillRL-SFT-Data](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data),
for ALFWorld, WebShop, and Search-agent.

The rollout step is **not** included — run your own model on the target
environment, dump trajectories to disk, then run the pipeline here.

## Pipeline

```
your trajectory dump (txt)
  │
  ▼
[1] preprocess/    parse + dedupe              → processed_trajectories.json
[2] skill_memory/  per-traj memories + aggregate → skill_bank.json
[3] distillation/  DeepSeek V4 Pro reasoning + (ALFWorld) synthetic done step
[4] postprocess/   sharegpt → alpaca pairs + (WebShop) validate
  │
  ▼
{instruction, output} JSON  (alpaca, matches HF schema)
```

## Quick start

```bash
pip install openai
export DEEPSEEK_API_KEY=...
export ROLLOUT_DIR=/path/to/your/rollout/txts
export WORK_DIR=/path/to/output

bash run_alfworld.sh   # or run_webshop.sh / run_search.sh
```

## Per-environment differences

| | ALFWorld | WebShop | Search |
|---|---|---|---|
| Success filter | `Reward: 10.000` | `Reward: 10.000` (`click[buy now]` matches target) | `Reward: 1.0` |
| Synthetic `done` step | **Yes** | No (`click[buy now]` terminates env) | No (`<answer>` terminates) |
| Skill categories | 6 task types (pick / clean / heat / cool / examine / two-obj) | 3 (general / apparel / electronics) | 2 (direct_retrieval / multi_hop) |
| Postprocess validation | not needed | normalize action vocabulary | not needed |

## Input rollout txt format (ALFWorld / WebShop)

The parsers expect one .txt per rollout, organized as:

```
<ROLLOUT_DIR>/
  env000/
    test1.txt
    test2.txt
    ...
  env001/
    ...
```

Each .txt has one block per step:

```
=== Trajectory for Test 0, Env 5 ===
Step -1 | Action: None | Reward: 0.000 | Done: False
Obs: -= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a cabinet 1, ...

Your task is to: put a clean plate in box

Your admissible actions of the current situation are: [open cabinet 1, go to countertop 1, ...].

Now it's your turn to take an action. ...

Step 00 | Action: go to countertop 1 | Reward: 0.000 | Done: False
Obs: You arrive at countertop 1. ...
Your admissible actions of the current situation are: [...].

...

Step 12 | Action: put plate 1 in box 1 | Reward: 10.000 | Done: True
Obs: You put the object down successfully.
Your admissible actions of the current situation are: [...].
```

Conventions:

- `Step -1` is the initial state (`Action: None`).
- `Step N` (N ≥ 0) is the Nth action; its `Obs` is the env response
  **after** taking the action plus the admissible actions for the new
  state.
- The winning step has `Reward: 10.000` and `Done: True`.

A reference rollout writer is in `verl-agent/examples/prompt_agent/qwen_alfworld.py`.

## Input rollout txt format (Search)

```
Your question: when was i can only imagine movie released?

History:
Step 0:<search>I Can Only Imagine film release date</search> <information>...</information>
Step 1:<search>"I Can Only Imagine" theatrical release</search> <information>...</information>

Now it's your turn to take an action.
Action 2: <answer>March 16, 2018</answer>

Reward: 1.0
```

File numbering is used to infer the QA dataset (`nq`, `popqa`,
`hotpotqa`, etc.) — adjust `infer_data_source` in `parse_search.py` if
your numbering differs.

## Bring your own JSON

If your rollout produces a different on-disk format, skip
`preprocess/parse_*.py` and feed your own JSON to stage 2 directly. The
schema downstream stages expect:

```jsonc
[
  {
    "env_id": "env001",
    "task": "put a clean plate in box",
    "type": "all_success",   // or "all_fail"
    "trajectories": [
      [
        { "step_id": "Step -1", "action": null,
          "observation": "<initial state>",
          "admissible_actions": [...], "reward": 0.0, "done": false },
        { "step_id": "Step 00", "action": "go to countertop 1",
          "observation": "<state after action>",
          "admissible_actions": [...], "reward": 0.0, "done": false },
        ...
      ],
      ...
    ]
  }
]
```

## Output schema

```jsonc
[
  {
    "instruction": "You are an expert agent operating in the ALFRED Embodied Environment.\nYour task is to: ...\n\n## Retrieved Relevant Experience\n\n### General Principles\n- ...\n\n### <Category> Skills\n- ...\n\n### Mistakes to Avoid\n- ...\n\n## Current Progress\nYour current observation is: ...\nYour admissible actions of the current situation are: [...]\n\nNow it's your turn to take an action.\n...",
    "output": "<think>...</think>\n<action>...</action>"
  }
]
```

Same column layout as the HF dataset (`instruction`, `output`); can be
loaded directly by alpaca-style SFT trainers.

## Layout

```
sft_data_generation/
├── README.md
├── run_alfworld.sh / run_webshop.sh / run_search.sh   # entry points
│
├── preprocess/        # [1] txt → JSON
│   ├── parse_alfworld.py
│   ├── parse_webshop.py
│   ├── parse_search.py
│   └── dedupe_repetitions.py
│
├── skill_memory/      # [2] per-traj memories + aggregate (DeepSeek V4 Pro)
│   ├── generate_memory_alfworld.py
│   ├── generate_memory_webshop.py
│   ├── generate_memory_search.py
│   └── aggregate_skills.py
│
├── distillation/      # [3] DeepSeek V4 Pro reasoning + (ALFWorld) synthetic done
│   ├── skill_retrieval.py
│   ├── distill_alfworld.py
│   ├── distill_webshop.py
│   └── distill_search.py
│
└── postprocess/       # [4] sharegpt → alpaca + (WebShop) validate
    ├── sharegpt_to_pairs.py
    └── validate_and_fix_webshop.py
```
