import csv
import time
import requests
import logging
import sys
import statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8000"

# Repo-specific query sets — swap TEST_QUERIES to benchmark different codebases
SQLMODEL_QUERIES = [
    "How does SQLModel combine Pydantic models and SQLAlchemy ORM functionality?",
    "What happens internally when a class inherits from SQLModel and sets table=True?",
    "How is the database metadata generated and where are tables registered?",
    "How does Session.exec() differ from standard SQLAlchemy query execution?",
    "How does SQLModel determine which fields become database columns?",
    "What is the role of Field() in SQLModel and how is it translated into SQLAlchemy column definitions?",
    "How are relationships between tables implemented in SQLModel?",
    "Trace the execution flow when SQLModel.metadata.create_all(engine) is called.",
    "How does SQLModel support type hints and convert them into SQL column types?",
    "Where and how does SQLModel integrate with Pydantic validation during object creation?"
]

FASTAPI_QUERIES = [
    "How does authentication work?",
    "How are routes registered?",
    "How are database sessions managed?",
    "How does dependency injection work?",
    "How are request bodies validated?",
    "How does error handling work?",
    "How are response models used?",
    "How does application startup work?",
    "How are path parameters processed?",
    "How is middleware configured?"
]

LANGCHAIN_QUERIES = [
    "How does the Agent class execute a chain of tools?",
    "What is the difference between a Tool and a ToolUse object?",
    "How does the memory system store and retrieve conversation history?",
    "How are prompt templates processed and formatted?",
    "What happens when a model response is parsed and validated?",
    "How does the retriever integrate with the language model?",
    "What is the role of callbacks in tracing LLM execution?",
    "How does the document loader chunk and split text?",
    "What are embeddings and how are they used for similarity search?",
    "How does the vector store handle semantic search queries?"
]

# Choose which query set to use
TEST_QUERIES = SQLMODEL_QUERIES  # Change to FASTAPI_QUERIES or LANGCHAIN_QUERIES as needed


def verify_api_ready():
    """Verify API is running and a repo has been indexed."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("repo_indexed"):
            logger.error("No repo has been indexed yet. Run /ingest first.")
            sys.exit(1)
        
        logger.info(f"✓ API healthy. Repo indexed: {data['repo_indexed']}")
        return True
    
    except requests.ConnectionError:
        logger.error(f"Cannot connect to API at {API_BASE}. Is it running?")
        sys.exit(1)
    except requests.Timeout:
        logger.error(f"API at {API_BASE} timed out.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error verifying API: {e}")
        sys.exit(1)


results = []

logger.info("Running benchmark...\n")
verify_api_ready()

for i, query in enumerate(TEST_QUERIES, start=1):

    logger.info(f"[{i}/{len(TEST_QUERIES)}] {query}")
    start = time.perf_counter()
    error_msg = None

    try:
        resp = requests.post(
            f"{API_BASE}/ask",
            json={
                "question": query,
                "n_results": 5
            },
            timeout=120
        )

        elapsed = time.perf_counter() - start
        
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning(f"Query failed: {error_msg}")
            results.append({
                "question": query,
                "success": False,
                "latency": round(elapsed, 2),
                "answer_words": 0,
                "source_count": 0,
                "answer": None,
                "sources": None,
                "error": error_msg,
                "manual_grade": None
            })
            continue

        data = resp.json()
        answer = data["answer"]
        sources = data["sources"]

        results.append({
            "question": query,
            "success": True,
            "latency": round(elapsed, 2),
            "answer_words": len(answer.split()),
            "source_count": len(sources),
            "answer": answer[:5000],
            "sources": " | ".join(sources),
            "error": None,
            "manual_grade": None
        })

    except requests.Timeout:
        elapsed = time.perf_counter() - start
        error_msg = "Request timeout (120s exceeded)"
        logger.error(error_msg)
        results.append({
            "question": query,
            "success": False,
            "latency": round(elapsed, 2),
            "answer_words": 0,
            "source_count": 0,
            "answer": None,
            "sources": None,
            "error": error_msg,
            "manual_grade": None
        })

    except requests.ConnectionError as e:
        elapsed = time.perf_counter() - start
        error_msg = f"Connection error: {str(e)[:100]}"
        logger.error(error_msg)
        results.append({
            "question": query,
            "success": False,
            "latency": round(elapsed, 2),
            "answer_words": 0,
            "source_count": 0,
            "answer": None,
            "sources": None,
            "error": error_msg,
            "manual_grade": None
        })

    except Exception as e:
        elapsed = time.perf_counter() - start
        error_msg = f"Unexpected error: {type(e).__name__}: {str(e)[:100]}"
        logger.error(error_msg)
        results.append({
            "question": query,
            "success": False,
            "latency": round(elapsed, 2),
            "answer_words": 0,
            "source_count": 0,
            "answer": None,
            "sources": None,
            "error": error_msg,
            "manual_grade": None
        })

successes = [r for r in results if r["success"]]
failures = [r for r in results if not r["success"]]

avg_latency = (
    sum(r["latency"] for r in successes) / len(successes)
    if successes else 0
)

avg_answer_words = (
    sum(r["answer_words"] for r in successes) / len(successes)
    if successes else 0
)

avg_sources = (
    sum(r["source_count"] for r in successes) / len(successes)
    if successes else 0
)

success_rate = (
    len(successes) / len(results) * 100 if results else 0
)

# Percentile latency (P50 = median, P95 = 95th percentile)
latencies = [r["latency"] for r in successes if r["latency"] > 0]
p50 = statistics.median(latencies) if latencies else 0
p95 = sorted(latencies)[int(len(latencies)*0.95) - 1] if len(latencies) > 1 else (latencies[0] if latencies else 0)

logger.info("\n" + "="*60 + " BENCHMARK RESULTS " + "="*60)
logger.info(f"Total Queries: {len(results)}")
logger.info(f"Success Rate: {success_rate:.1f}% ({len(successes)}/{len(results)})")

if successes:
    logger.info(f"Average Latency: {avg_latency:.2f}s")
    logger.info(f"P50 Latency: {p50:.2f}s")
    logger.info(f"P95 Latency: {p95:.2f}s")
    logger.info(f"Average Answer Length: {avg_answer_words:.1f} words")
    logger.info(f"Average Sources Returned: {avg_sources:.1f}")

if failures:
    logger.warning(f"Failed Qukeries: {len(failures)}")
    for i, fail in enumerate(failures, 1):
        logger.warning(f"  {i}. {fail['question'][:60]}... → {fail.get('error', 'Unknown error')}")

with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
    if results:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

logger.info("✓ Saved benchmark_results.csv")
logger.info("  → Open the CSV and fill in 'manual_grade' column (Good/Partial/Bad) for quality evaluation")
logger.info("="*120)