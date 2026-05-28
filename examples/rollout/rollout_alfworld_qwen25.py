"""
Generate ALFWorld base-model rollout txt files for SkillRL SFT data generation.

This script fills the gap before ``examples/sft_data_generation/run_alfworld.sh``:
it runs a Qwen Instruct model in ALFWorld and writes one txt file per rollout in
the exact format expected by ``preprocess/parse_alfworld.py``.

Example:
    CUDA_VISIBLE_DEVICES=0,1 python -m examples.rollout.rollout_alfworld_qwen25 \
        --model-path Qwen/Qwen2.5-3B-Instruct \
        --model-dir /home/sunzhengyu/SkillRL/models/Qwen2.5-3B-Instruct \
        --output-dir runs/rollouts/alfworld_qwen25_3b \
        --total-envs 200 \
        --env-num 16 \
        --num-rollouts-per-env 3 \
        --tensor-parallel-size 2
"""

"""
CUDA_VISIBLE_DEVICES=3 python -m examples.rollout.rollout_alfworld_qwen25 \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --model-dir /home/sunzhengyu/SkillRL/models/Qwen2.5-3B-Instruct \
  --output-dir runs/rollouts/alfworld_qwen25_3b_fixed \
  --total-envs 200 \
  --env-num 16 \
  --num-rollouts-per-env 3 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8
"""

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import OmegaConf
from vllm import LLM, SamplingParams

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import (
    alfworld_projection,
    build_alfworld_envs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen base-model rollouts and dump ALFWorld txt trajectories."
    )
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Project-local directory used when --model-path is a HuggingFace repo id.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--total-envs",
        type=int,
        default=None,
        help="Total number of distinct env ids to generate. Defaults to --env-num.",
    )
    parser.add_argument(
        "--env-num",
        type=int,
        default=200,
        help="Number of envs run in parallel per chunk.",
    )
    parser.add_argument("--num-rollouts-per-env", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--eval-dataset",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
        default="eval_in_distribution",
    )
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-cpus-per-env", type=float, default=0.05)
    return parser.parse_args()


def build_env_manager(
    args: argparse.Namespace,
    env_num: int,
    seed: int,
) -> AlfWorldEnvironmentManager:
    repo_root = Path(__file__).resolve().parents[2]
    alf_config_path = (
        repo_root
        / "agent_system"
        / "environments"
        / "env_package"
        / "alfworld"
        / "configs"
        / "config_tw.yaml"
    )
    env_config = OmegaConf.create(
        {
            "env": {
                "env_name": "alfworld/AlfredTWEnv",
                "history_length": args.history_length,
                "use_skills_only_memory": False,
                "rollout": {"n": 1},
                "resources_per_worker": {
                    "num_cpus": args.num_cpus_per_env,
                    "num_gpus": 0.0,
                },
                "alfworld": {"eval_dataset": args.eval_dataset},
            },
            "data": {"train_batch_size": env_num, "val_batch_size": env_num},
        }
    )
    envs = build_alfworld_envs(
        str(alf_config_path),
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=False,
        env_kwargs={"eval_dataset": args.eval_dataset},
        resources_per_worker=OmegaConf.to_container(
            env_config.env.resources_per_worker, resolve=True
        ),
    )
    return AlfWorldEnvironmentManager(envs, alfworld_projection, env_config)


def resolve_model_path(args: argparse.Namespace) -> str:
    candidate = Path(args.model_path).expanduser()
    if candidate.exists():
        return str(candidate)

    repo_root = Path(__file__).resolve().parents[2]
    model_dir = (
        Path(args.model_dir).expanduser()
        if args.model_dir
        else Path("/home/sunzhengyu/SkillRL/models") / args.model_path.split("/")[-1]
    )
    if (model_dir / "config.json").exists():
        return str(model_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download HF model ids. "
            "Install it or pass --model-path as an existing local model directory."
        ) from exc

    # 将模型固定下载到项目目录，避免默认 HF cache 分散到用户主目录中。
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.model_path, local_dir=str(model_dir))
    return str(model_dir)


def build_model(args: argparse.Namespace) -> tuple[LLM, SamplingParams]:
    local_model_path = resolve_model_path(args)
    llm = LLM(
        model=local_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    return llm, sampling_params


def apply_chat_template(tokenizer: Any, observation: str) -> str:
    # 与训练侧 TrajectoryCollector 保持一致：Qwen Instruct 看到的是 chat 格式。
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": observation}],
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_actions(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    active_mask: np.ndarray,
) -> list[str]:
    tokenizer = llm.get_tokenizer()
    active_prompts = [
        apply_chat_template(tokenizer, prompt)
        for prompt, active in zip(prompts, active_mask)
        if active
    ]
    if not active_prompts:
        return [""] * len(prompts)

    outputs = llm.generate(active_prompts, sampling_params)
    active_texts = [item.outputs[0].text.strip() for item in outputs]

    actions = []
    active_iter = iter(active_texts)
    for active in active_mask:
        actions.append(next(active_iter) if active else "")
    return actions


def build_inactive_action(admissible_actions: list[str] | None) -> str:
    # 已结束环境仍会被 vector env step；给一个格式合法的动作，避免无标签文本污染投影逻辑。
    if admissible_actions:
        action = next(
            (item for item in admissible_actions if item != "help"),
            admissible_actions[0],
        )
    else:
        action = "look"
    return f"<think>The episode has already finished.</think>\n<action>{action}</action>"


def project_actions(raw_actions: list[str], admissible_actions: list[list[str]]) -> list[str]:
    # parser 后续只需要环境动作本身；原始 <think>/<action> 文本不写入 Action 字段。
    projected, _ = alfworld_projection(raw_actions.copy(), admissible_actions)
    return projected


def format_step(
    step_id: str,
    action: str | None,
    reward: float,
    done: bool,
    observation: str,
) -> str:
    action_text = "None" if action is None else action
    return (
        f"{step_id} | Action: {action_text} | Reward: {reward:.3f} | Done: {done}\n"
        f"Obs: {observation.strip()}\n"
    )


def write_rollout(path: Path, test_idx: int, env_idx: int, steps: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"=== Trajectory for Test {test_idx}, Env {env_idx} ===\n")
        for step in steps:
            f.write(step)
            if not step.endswith("\n\n"):
                f.write("\n")


def run_once(
    args: argparse.Namespace,
    env_manager: AlfWorldEnvironmentManager,
    llm: LLM,
    sampling_params: SamplingParams,
    test_idx: int,
    env_num: int,
    env_offset: int,
) -> None:
    obs, _ = env_manager.reset(kwargs={})
    done_mask = np.zeros(env_num, dtype=bool)
    trajectories = [
        [
            format_step(
                step_id="Step -1",
                action=None,
                reward=0.0,
                done=False,
                observation=obs["text"][env_idx],
            )
        ]
        for env_idx in range(env_num)
    ]

    for step_idx in range(args.max_steps):
        active_mask = np.logical_not(done_mask)
        if not active_mask.any():
            break

        current_admissible = env_manager.envs.get_admissible_commands
        raw_actions = generate_actions(llm, sampling_params, obs["text"], active_mask)
        for env_idx, active in enumerate(active_mask):
            if not active:
                raw_actions[env_idx] = build_inactive_action(current_admissible[env_idx])

        projected_actions = project_actions(raw_actions, current_admissible)

        # EnvManager 会再次做 projection；这里保留 raw_actions 作为输入以保持运行路径一致。
        next_obs, rewards, dones, _ = env_manager.step(raw_actions)
        rewards = np.asarray(rewards, dtype=float)
        dones = np.asarray(dones, dtype=bool)

        for env_idx in range(env_num):
            if done_mask[env_idx]:
                continue
            trajectories[env_idx].append(
                format_step(
                    step_id=f"Step {step_idx:02d}",
                    action=projected_actions[env_idx],
                    reward=float(rewards[env_idx]),
                    done=bool(dones[env_idx]),
                    observation=next_obs["text"][env_idx],
                )
            )

        done_mask = np.logical_or(done_mask, dones)
        obs = next_obs
        success_rate = np.mean(rewards[active_mask] >= 10.0)
        print(
            f"[test {test_idx}] step={step_idx:02d} "
            f"envs={env_offset:03d}-{env_offset + env_num - 1:03d} "
            f"done={done_mask.sum()}/{env_num} step_success={success_rate:.3f}",
            flush=True,
        )

    output_root = Path(args.output_dir)
    for env_idx, steps in enumerate(trajectories):
        global_env_idx = env_offset + env_idx
        write_rollout(
            output_root / f"env{global_env_idx:03d}" / f"test{test_idx}.txt",
            test_idx=test_idx,
            env_idx=global_env_idx,
            steps=steps,
        )


def main() -> None:
    args = parse_args()
    llm, sampling_params = build_model(args)
    total_envs = args.total_envs or args.env_num

    for env_offset in range(0, total_envs, args.env_num):
        chunk_env_num = min(args.env_num, total_envs - env_offset)
        chunk_seed = args.seed + env_offset

        for test_idx in range(args.num_rollouts_per_env):
            # ALFWorld reset 会推进到下一批 game。为保证同一 envXXX 下的 test*.txt
            # 表示同一个任务的多次采样，这里每个 test 重建同 seed 的 env chunk。
            env_manager = build_env_manager(
                args,
                env_num=chunk_env_num,
                seed=chunk_seed,
            )
            try:
                run_once(
                    args,
                    env_manager,
                    llm,
                    sampling_params,
                    test_idx,
                    env_num=chunk_env_num,
                    env_offset=env_offset,
                )
            finally:
                env_manager.close()

    print(f"Done. Rollouts written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
