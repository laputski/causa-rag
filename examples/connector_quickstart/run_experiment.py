"""End-to-end: register toy_rag (already running via serve_app.py),
upload a tiny golden dataset, run an experiment, and print the results —
no curl, no hand-written FastAPI.

Prerequisites:
  1. The platform's gateway is running (`make up` / `make bootstrap`).
  2. serve_app.py is running in another terminal:
       python3 -m examples.connector_quickstart.serve_app

Run:
    python3 -m examples.connector_quickstart.run_experiment
"""
from causa_rag_client import RagPlatformClient

PLATFORM_URL = "http://localhost:8081"
RAG_URL = "http://localhost:8800/"


def main() -> None:
    client = RagPlatformClient(PLATFORM_URL)

    rag = client.register_rag(
        name="toy_rag_quickstart",
        url=RAG_URL,
        description="connector_quickstart example RAG",
    )
    print(f"Registered: {rag['id']} ({rag['name']})")

    probe = client.test_rag(rag["id"])
    print(f"Probe: ok={probe['ok']} answer_preview={probe.get('answer_preview')!r}")

    client.upload_dataset(
        rag["id"],
        filename="quickstart.jsonl",
        realm_id="quickstart",
        questions=[
            {
                "id": "q1", "question": "annual leave", "reference_answer": "24 calendar days",
                "article_refs": ["DEMO/1"], "answerability": "answerable",
            },
            {
                "id": "q2", "question": "working week", "reference_answer": "40 hours",
                "article_refs": ["DEMO/2"], "answerability": "answerable",
            },
        ],
    )

    run = client.run_experiment(
        name="connector_quickstart_run",
        dataset_name="quickstart.jsonl",
        rag_id=rag["id"],
    )
    print(f"Run started: {run['run_id']} (status={run['status']})")

    result = client.wait_for_completion(run["run_id"], poll_interval=1.0, timeout=120)
    print("Done. Aggregate metrics:")
    for key, value in result.get("aggregate_metrics", {}).items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
