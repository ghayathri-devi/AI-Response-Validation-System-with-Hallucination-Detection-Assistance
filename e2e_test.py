import requests

BASE_URL = "http://127.0.0.1:8000"

results = []  # (test_name, passed: bool, detail: str)


def record(test_name, passed, detail=""):
    results.append((test_name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {test_name} — {detail}")


def test_single_evaluation_workflow():
    payload = {
        "question": "What sits on top of the Main Building at Notre Dame?",
        "answer": "A golden statue of the Virgin Mary sits atop the Main Building.",
        "reference_answer": "A golden statue of the Virgin Mary",
    }
    try:
        resp = requests.post(f"{BASE_URL}/evaluate", data=payload, timeout=60)
        ok = resp.status_code == 200
        data = resp.json() if ok else {}
        has_fields = ok and "evaluation" in data and "retrieved_context" in data
        eval_obj = data.get("evaluation", {})
        has_all_scores = all(
            k in eval_obj for k in
            ["relevance", "accuracy", "completeness", "hallucination_risk", "overall_score", "verdict_label"]
        )
        record(
            "Single evaluation workflow", ok and has_fields and has_all_scores,
            f"status={resp.status_code}, verdict={eval_obj.get('verdict_label')}"
        )
    except Exception as e:
        record("Single evaluation workflow", False, f"Exception: {e}")



def test_batch_evaluation_workflow():
    csv_content = (
        "question,answer,reference_answer\n"
        "What is the capital of France?,Paris is the capital of France.,Paris\n"
        "What is 2+2?,2+2 equals 5.,4\n"
    )
    files = {"csv_file": ("test_e2e.csv", csv_content, "text/csv")}
    try:
        resp = requests.post(f"{BASE_URL}/batch-evaluate", files=files, timeout=120)
        ok = resp.status_code == 200
        data = resp.json() if ok else {}
        has_results = ok and len(data.get("results", [])) == 2
        has_aggregate = ok and "aggregate" in data and data["aggregate"].get("count") == 2
        record(
            "Batch evaluation workflow", ok and has_results and has_aggregate,
            f"status={resp.status_code}, rows={data.get('row_count')}"
        )
        return data if ok else None
    except Exception as e:
        record("Batch evaluation workflow", False, f"Exception: {e}")
        return None


def test_dashboard_updates():
    try:
        before = requests.get(f"{BASE_URL}/analytics", timeout=30).json()
        before_total = before.get("total_evaluations", 0)

        requests.post(f"{BASE_URL}/evaluate", data={
            "question": "What is a prime number?",
            "answer": "A prime number is a natural number greater than 1 with no divisors other than 1 and itself.",
        }, timeout=60)

        after = requests.get(f"{BASE_URL}/analytics", timeout=30).json()
        after_total = after.get("total_evaluations", 0)

        passed = after_total > before_total
        record(
            "Dashboard updates after new evaluation", passed,
            f"before={before_total}, after={after_total}"
        )
    except Exception as e:
        record("Dashboard updates after new evaluation", False, f"Exception: {e}")



def test_report_generation(batch_data):
    if not batch_data:
        record("Report generation (PDF export)", False, "Skipped — batch workflow test did not produce data")
        return
    try:
        resp = requests.post(f"{BASE_URL}/export-report", json={
            "results": batch_data["results"],
            "aggregate": batch_data["aggregate"],
        }, timeout=60)
        ok = resp.status_code == 200
        is_pdf = ok and resp.headers.get("content-type", "").startswith("application/pdf")
        has_content = ok and len(resp.content) > 1000
        record(
            "Report generation (PDF export)", ok and is_pdf and has_content,
            f"status={resp.status_code}, size={len(resp.content) if ok else 0} bytes"
        )
    except Exception as e:
        record("Report generation (PDF export)", False, f"Exception: {e}")



def test_rag_retrieval():
    try:
        from retrieval_agent import retrieve_context
        chunks = retrieve_context("What sits on top of the Main Building at Notre Dame?", top_k=2)
        passed = isinstance(chunks, list) and len(chunks) > 0 and all(isinstance(c, str) for c in chunks)
        record("RAG retrieval returns relevant chunks", passed, f"{len(chunks)} chunk(s) retrieved")
    except Exception as e:
        record("RAG retrieval returns relevant chunks", False, f"Exception: {e}")



def test_agent_scoring():
    try:
        from relevance_agent import judge_relevance
        from accuracy_agent import judge_accuracy
        from completeness_agent import judge_completeness
        from hallucination_agent import detect_hallucination

        question = "What is the capital of France?"
        answer = "Paris is the capital of France."
        reference = "Paris"
        contexts = ["The capital of France is Paris, a major European city."]

        rel = judge_relevance(question, answer)
        acc = judge_accuracy(answer, reference, contexts)
        comp = judge_completeness(question, answer)
        halluc = detect_hallucination(answer, contexts, reference)

        record("Agent scoring — Relevance", 0.0 <= rel["relevance_score"] <= 1.0 and bool(rel["reasoning"]),
               f"score={rel['relevance_score']}")
        record("Agent scoring — Accuracy", 0.0 <= acc["accuracy_score"] <= 1.0 and bool(acc["evidence"]),
               f"score={acc['accuracy_score']}")
        record("Agent scoring — Completeness", 0.0 <= comp["completeness_score"] <= 1.0 and bool(comp["reasoning"]),
               f"score={comp['completeness_score']}")
        record("Agent scoring — Hallucination",
               halluc["hallucination_risk"] in ("Low", "Medium", "High") and bool(halluc["reasoning"]),
               f"risk={halluc['hallucination_risk']}")
    except Exception as e:
        record("Agent scoring", False, f"Exception: {e}")


def test_verdict_generation():
    try:
        from verdict_agent import build_verdict

        relevance_result = {"relevance_score": 1.0}
        accuracy_result = {"accuracy_score": 1.0}
        completeness_result = {"completeness_score": 1.0, "missing_aspects": []}
        hallucination_result = {"hallucination_risk": "Low", "flagged_statements": []}

        verdict = build_verdict(relevance_result, accuracy_result, completeness_result, hallucination_result)

        expected_score = round(1.0 * 0.20 + 1.0 * 0.35 + 1.0 * 0.15 + 1.0 * 0.30, 2)
        score_correct = verdict["overall_score"] == expected_score
        label_correct = verdict["verdict_label"] == "Excellent"

        record("Verdict generation — weighted math", score_correct,
               f"expected={expected_score}, got={verdict['overall_score']}")
        record("Verdict generation — label banding", label_correct,
               f"label={verdict['verdict_label']}")
    except Exception as e:
        record("Verdict generation", False, f"Exception: {e}")


def test_error_handling():
    # Malformed CSV (wrong columns) should return a clean 400, not a 500 crash
    try:
        bad_csv = "not,a,valid,header\nfoo,bar,baz,qux\n"
        files = {"csv_file": ("bad.csv", bad_csv, "text/csv")}
        resp = requests.post(f"{BASE_URL}/batch-evaluate", files=files, timeout=30)
        record("Error handling — malformed batch CSV", resp.status_code == 400, f"status={resp.status_code}")
    except Exception as e:
        record("Error handling — malformed batch CSV", False, f"Exception: {e}")

    # Oversized batch (>20 rows) should be rejected cleanly, not silently truncated
    try:
        rows = "question,answer\n" + "".join([f"Q{i},A{i}\n" for i in range(25)])
        files = {"csv_file": ("oversized.csv", rows, "text/csv")}
        resp = requests.post(f"{BASE_URL}/batch-evaluate", files=files, timeout=30)
        record("Error handling — batch row limit exceeded", resp.status_code == 400, f"status={resp.status_code}")
    except Exception as e:
        record("Error handling — batch row limit exceeded", False, f"Exception: {e}")



def test_invalid_input_handling():
    # Missing required 'answer' field
    try:
        resp = requests.post(f"{BASE_URL}/evaluate", data={"question": "What is AI?"}, timeout=30)
        record("Invalid input — missing required field", resp.status_code in (400, 422), f"status={resp.status_code}")
    except Exception as e:
        record("Invalid input — missing required field", False, f"Exception: {e}")

    # Empty question and answer — should not crash (500) even if handled leniently
    try:
        resp = requests.post(f"{BASE_URL}/evaluate", data={"question": "", "answer": ""}, timeout=30)
        record("Invalid input — empty question/answer", resp.status_code != 500, f"status={resp.status_code}")
    except Exception as e:
        record("Invalid input — empty question/answer", False, f"Exception: {e}")

    # Non-CSV file uploaded to batch endpoint
    try:
        files = {"csv_file": ("not_a_csv.txt", "just some text", "text/plain")}
        resp = requests.post(f"{BASE_URL}/batch-evaluate", files=files, timeout=30)
        record("Invalid input — non-CSV file to batch endpoint", resp.status_code == 400, f"status={resp.status_code}")
    except Exception as e:
        record("Invalid input — non-CSV file to batch endpoint", False, f"Exception: {e}")



def main():
    print("=" * 80)
    print("END-TO-END TEST SUITE — AI Response Quality Evaluator")
    print("=" * 80)
    print("NOTE: requires the FastAPI server running locally before starting.\n")

    test_single_evaluation_workflow()
    batch_data = test_batch_evaluation_workflow()
    test_dashboard_updates()
    test_report_generation(batch_data)
    test_rag_retrieval()
    test_agent_scoring()
    test_verdict_generation()
    test_error_handling()
    test_invalid_input_handling()

    print("\n" + "=" * 80)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print(f"SUMMARY: {passed_count}/{total_count} tests passed")
    print("=" * 80)

    failed = [r for r in results if not r[1]]
    if failed:
        print("\nFailed tests:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")

    with open("e2e_test_report.txt", "w") as f:
        f.write("END-TO-END TEST REPORT\n")
        f.write("AI Response Quality Evaluator\n")
        f.write("=" * 80 + "\n\n")
        for name, passed, detail in results:
            status = "PASS" if passed else "FAIL"
            f.write(f"[{status}] {name}\n       {detail}\n\n")
        f.write(f"\nSUMMARY: {passed_count}/{total_count} tests passed\n")

    print("\nFull report saved to e2e_test_report.txt")


if __name__ == "__main__":
    main()