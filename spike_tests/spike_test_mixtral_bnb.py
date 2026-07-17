"""Spike test: does Mixtral-8x7B (MoE) + bitsandbytes actually work in the
pinned vLLM version? Historically unstable -- MoE in-flight bnb support was
added incrementally (vllm-project/vllm PR #20061, tracking issue #20480),
and vLLM has an open RFC (#39583) proposing to deprecate bnb entirely. Run
this BEFORE trusting any mixtral-8x7b rows in the full run_matrix.

Loads via benchmark.engine.build_llm (not a raw vllm.LLM(..., hf_overrides=...)
call) -- vLLM's hf_overrides={"quantization_config": {...}} mechanism is
confirmed broken (weight-shape AssertionError, reproduced across every vLLM
version from 0.9.2 to 0.25.1; see spike_test_error_report.md) and is no
longer what the real benchmark run uses. Pulling the real mixtral-8x7b/
int8_baseline entry out of configs/run_matrix.yaml via find_entries() means
this spike test exercises the exact same path Phase 3 will use, so its
PASS/FAIL is representative.

PASS -> proceed with configs/run_matrix.yaml as-is.
FAIL -> apply the fallback noted in run_matrix.yaml's `known_risk` field for
        mixtral-8x7b: use HF transformers + BitsAndBytesConfig for the
        Mixtral leg only, and flag in the report that Mixtral's
        throughput/latency numbers are then not directly comparable to the
        vLLM-served numbers for the other 3 models.

Usage:
    python spike_tests/spike_test_mixtral_bnb.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `benchmark` is importable regardless of cwd

import yaml

from benchmark.engine import build_llm
from benchmark.run_one_combo import find_entries

MODEL_ID = "mixtral-8x7b"
QUANT_LEVEL = "int8_baseline"
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
        print("See configs/run_matrix.yaml known_risk field for the transformers-native fallback plan.")
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
        print("FAIL: generation produced empty output (possible silently broken MoE+bnb kernel).")
        sys.exit(1)

    print("PASS: Mixtral-8x7B + bitsandbytes loads and generates in this vLLM version.")
    print("NOTE: this only checks load+generate succeed, not output quality -- spot-check")
    print("a few real Task A/B/C outputs manually after the first Mixtral combo completes.")


if __name__ == "__main__":
    main()
