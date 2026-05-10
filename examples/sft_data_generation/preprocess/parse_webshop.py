"""
Parse raw WebShop rollout trajectories (txt format) into structured JSON.

Same parser as ALFWorld (the rollout writer in 01_rollout uses the same
`Step N | Action: ... | Reward: ... | Done: ...` header format), but uses
WebShop-flavored bookkeeping:

- Successful trajectories: identified by 'Reward: 10.000' (WebShop only
  emits a 10.0 reward when 'click[buy now]' lands on a perfectly matching
  product).
- The winning action is `click[buy now]`; no synthetic terminal step is
  needed (unlike ALFWorld's `done`).

Usage:
    python parse_webshop.py \\
        --input_dir /path/to/trajectories_qwen2.5-sft_webshop \\
        --output_file processed_trajectories_webshop.json
"""
import argparse

# Re-use the parser implementation from parse_alfworld.
from parse_alfworld import parse_trajectory_file  # noqa: F401
from parse_alfworld import main as alfworld_main


def main():
    # parse_alfworld's main is generic enough for WebShop because
    # the rollout txt format is identical. We just rename for clarity.
    alfworld_main()


if __name__ == "__main__":
    main()
