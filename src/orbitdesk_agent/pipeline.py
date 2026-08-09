import json
import time

from . import config
from .graph import build_graph
from .logging_utils import get_logger
from .models import GenerationModel
from .retrieval import EmbeddingModel, RetrievalIndex

logger = get_logger(__name__)


class SupportAgentPipeline:
    def __init__(self) -> None:
        self.embedding_model = EmbeddingModel()
        self.retrieval_index = RetrievalIndex(self.embedding_model)
        self.generation_model = GenerationModel()
        self.app = build_graph(self.embedding_model, self.retrieval_index, self.generation_model)
        self.schema = json.loads(config.OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))

    def run(self, question: str, question_id: str = "adhoc", debug_force_failure: bool = False) -> dict:
        from jsonschema import ValidationError, validate

        started = time.perf_counter()
        initial_state = {
            "question_id": question_id,
            "question": question,
            "debug_force_failure": debug_force_failure,
            "generation_attempts": 0,
            "trace": [],
            "timings": {},
        }
        result = self.app.invoke(initial_state, config={"recursion_limit": config.RECURSION_LIMIT})
        response = result["final_response"]

        try:
            validate(instance=response, schema=self.schema)
            schema_valid = True
            schema_error = None
        except ValidationError as exc:
            schema_valid = False
            schema_error = str(exc.message)
            response.setdefault("warnings", []).append("schema_validation_failed")

        total_latency = time.perf_counter() - started
        logger.info(
            "Completed %s in %.2fs (classification=%s, schema_valid=%s)",
            question_id,
            total_latency,
            response.get("classification"),
            schema_valid,
        )

        return {
            "response": response,
            "trace": result.get("trace", []),
            "timings": result.get("timings", {}),
            "schema_valid": schema_valid,
            "schema_error": schema_error,
            "total_latency_seconds": total_latency,
        }
