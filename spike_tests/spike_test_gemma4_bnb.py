"""Spike test: does google/gemma-4-31B + bitsandbytes work in the pinned
vLLM version? The official vLLM Gemma4 recipe documents only W4A16 (QAT) and
int8-per-channel (MoE-variant-only) quantization paths -- NOT bitsandbytes.
gemma-4-31B is a brand-new dense architecture (Gemma4ForConditionalGeneration),
so generic bnb nn.Linear-replacement should work in principle, but is
unverified. Run BEFORE trusting any gemma4-31b rows in the full run_matrix.

Loads via benchmark.engine.build_llm (not a raw vllm.LLM(..., hf_overrides=...)
call) -- vLLM's hf_overrides={"quantization_config": {...}} mechanism is
confirmed broken (weight-shape AssertionError, reproduced across every vLLM
version from 0.9.2 to 0.25.1; see docs/spike_test_error_report.md) and is no
longer what the real benchmark run uses. Pulling the real gemma4-31b/
int8_bnb entry out of configs/run_matrix.yaml via find_entries() (which
already carries trust_remote_code and the limit_mm_per_prompt vllm_extra_arg
-- build_llm applies both generically) means this spike test exercises the
exact same path Phase 3 will use, so its PASS/FAIL is representative.

PASS -> proceed with configs/run_matrix.yaml as-is.
FAIL -> drop bnb for this model and use vLLM's officially documented gemma4
        quantization path instead (W4A16/int8-per-channel), noting the
        substitution explicitly in the report's limitations section since it
        is no longer an apples-to-apples bnb comparison with the other models.

Usage:
    python spike_tests/spike_test_gemma4_bnb.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `benchmark` is importable regardless of cwd

import yaml

from benchmark.engine import build_llm
from benchmark.run_one_combo import find_entries

MODEL_ID = "gemma4-31b"
QUANT_LEVEL = "int8_bnb"
RUN_MATRIX_PATH = REPO_ROOT / "configs" / "run_matrix.yaml"


def main() -> None:
    from vllm import SamplingParams

    with open(RUN_MATRIX_PATH) as f:
        run_matrix = yaml.safe_load(f)
    model_entry, run_entry = find_entries(run_matrix, MODEL_ID, QUANT_LEVEL)
    defaults = run_matrix["defaults"]

    print(f"Loading {model_entry['hf_repo']} ({MODEL_ID}/{QUANT_LEVEL}) via build_llm ...")
    try:
        llm = build_llm(model_entry, run_entry, defaults)
    except Exception as e:
        print(f"FAIL: model load raised an exception: {e}", file=sys.stderr)
        print(
            "If this looks like an unsupported-quantization error, gemma4 may only "
            "support W4A16/int8-per-channel per the official vLLM recipe -- drop bnb "
            "for this model and use that documented path instead, noting the "
            "substitution in the report's limitations section."
        )
        sys.exit(1)

    try:
        outputs = llm.generate(
            ["What is the capital of France? Answer in one word."],
            SamplingParams(max_tokens=16, temperature=0),
        )
        text = outputs[0].outputs[0].text.strip()
    except Exception as e:
        print(f"FAIL: generation raised an exception: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated: {text!r}")
    if not text:
        print("FAIL: generation produced empty output.")
        sys.exit(1)

    print("PASS: gemma-4-31B + bitsandbytes loads and generates in this vLLM version.")


if __name__ == "__main__":
    main()
