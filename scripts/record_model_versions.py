import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orbitdesk_agent import config
from orbitdesk_agent.models import GenerationModel
from orbitdesk_agent.retrieval import EmbeddingModel


def main() -> None:
    embedding_model = EmbeddingModel()
    generation_model = GenerationModel()

    info = {
        "embedding_model": {
            "name": embedding_model.model_name,
            "load_time_seconds": round(embedding_model.load_time_seconds, 3),
        },
        "generation_model": {
            "name": generation_model.model_name,
            "revision": generation_model.revision_hash,
            "device": generation_model.device,
            "load_time_seconds": round(generation_model.load_time_seconds, 3),
        },
    }

    output_path = config.OUTPUTS_DIR / "model_versions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(json.dumps(info, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
