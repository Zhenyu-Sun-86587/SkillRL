#!/usr/bin/env python3
"""Fix Qwen tokenizer_config.json compatibility with newer transformers.

Some SFT exporters keep ``extra_special_tokens`` as an empty list.  Newer
transformers versions expect this field to be a dict and fail during tokenizer
initialization.  The fix is metadata-only and does not touch model weights.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        help="Local model directory containing tokenizer_config.json.",
    )
    args = parser.parse_args()

    config_path = Path(args.model_dir) / "tokenizer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    value = config.get("extra_special_tokens")
    if isinstance(value, list):
        # 空 list 和空 dict 都表示没有额外特殊 token；这里仅修正字段类型。
        config["extra_special_tokens"] = {}
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Fixed extra_special_tokens in {config_path}")
    else:
        print(f"No change needed for {config_path}")


if __name__ == "__main__":
    main()
