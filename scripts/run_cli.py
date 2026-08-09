import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _print_result(result: dict) -> None:
    response = result["response"]
    print("=" * 72)
    print(f"Classification : {response['classification']}")
    print(f"Answer         : {response['answer']}")
    print(f"Confidence     : {response['confidence']}")
    print(f"Requires human : {response['requires_human']}")
    print(f"Reason         : {response['reason']}")
    if response.get("clarification_question"):
        print(f"Clarify        : {response['clarification_question']}")
    if response["sources"]:
        print("Sources:")
        for source in response["sources"]:
            print(f"  - {source['source_id']}: {source['passage']}")
    print(f"Node trace     : {' -> '.join(result['trace'])}")
    print(f"Schema valid   : {result['schema_valid']}")
    print(f"Latency (s)    : {result['total_latency_seconds']:.2f}")
    print(json.dumps(response, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="OrbitDesk local support agent")
    parser.add_argument("--question", type=str, help="Ask a single ad-hoc question")
    parser.add_argument("--samples", action="store_true", help="Run every question in sample_questions.json")
    parser.add_argument(
        "--debug-force-verification-failure",
        action="store_true",
        help="Force the first generation attempt to fail verification, to demonstrate the retry path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write results when running --samples",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip Hugging Face network checks and load models only from the local cache "
        "(use this once models have already been downloaded at least once)",
    )
    args = parser.parse_args()

    if not args.question and not args.samples:
        parser.error("Provide --question '...' or --samples")

    if args.offline:
        _enable_offline_mode()

    from orbitdesk_agent import config
    from orbitdesk_agent.pipeline import SupportAgentPipeline

    if args.output is None:
        args.output = str(config.OUTPUTS_DIR / "sample_runs.json")

    pipeline = SupportAgentPipeline()

    if args.question:
        result = pipeline.run(
            args.question,
            question_id="adhoc",
            debug_force_failure=args.debug_force_verification_failure,
        )
        _print_result(result)
        return

    payload = json.loads(config.SAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))
    all_results = []
    for item in payload["questions"]:
        result = pipeline.run(item["question"], question_id=item["question_id"])
        _print_result(result)
        all_results.append({"question_id": item["question_id"], "question": item["question"], **result})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} results to {output_path}")


if __name__ == "__main__":
    main()