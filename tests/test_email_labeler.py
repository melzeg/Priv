from worker.workflows.email_labeler import (
    ThreadSummary,
    build_classification_prompt,
    build_search_query,
    extract_thread_summaries,
    normalize_category,
)


def test_build_search_query_excludes_existing_labels():
    query = build_search_query(3)
    assert query == (
        "newer_than:3d -label:wijn -label:muziek -label:zeilen "
        "-in:sent -in:chats -in:spam -in:trash"
    )


def test_build_search_query_clamps_days_back():
    assert "newer_than:1d" in build_search_query(0)
    assert "newer_than:1d" in build_search_query(-5)


def test_normalize_category_maps_case_and_whitespace():
    assert normalize_category(" Wijn ") == "Wijn"
    assert normalize_category("MUZIEK") == "Muziek"
    assert normalize_category("zeilen") == "Zeilen"
    assert normalize_category("geen") == "geen"
    assert normalize_category("something else") == "geen"


def test_extract_thread_summaries_from_mcp_content():
    import json

    result = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "threads": [
                            {
                                "id": "t1",
                                "subject": "Nieuwe vintage",
                                "messages": [
                                    {
                                        "id": "m1",
                                        "from": "wijnhuis@example.com",
                                        "subject": "Nieuwe vintage",
                                        "plaintextBody": "Probeer onze nieuwe rode wijn.",
                                    }
                                ],
                            }
                        ],
                        "nextPageToken": "tok",
                    }
                ),
            }
        ]
    }

    threads = extract_thread_summaries(result)
    assert len(threads) == 1
    thread = threads[0]
    assert thread.thread_id == "t1"
    assert thread.subject == "Nieuwe vintage"
    assert thread.sender == "wijnhuis@example.com"
    assert thread.message_ids == ["m1"]
    assert "rode wijn" in thread.text


def test_extract_thread_summaries_skips_threads_without_id():
    result = {"threads": [{"subject": "no id"}, {"id": "t2"}]}
    threads = extract_thread_summaries(result)
    assert [t.thread_id for t in threads] == ["t2"]


def test_extract_thread_summaries_truncates_long_text():
    long_body = "x" * 10000
    result = {
        "threads": [
            {
                "id": "t1",
                "messages": [{"id": "m1", "plaintextBody": long_body}],
            }
        ]
    }
    threads = extract_thread_summaries(result)
    assert len(threads[0].text) <= 6000


def test_build_classification_prompt_contains_rules_and_content():
    thread = ThreadSummary(
        thread_id="t1",
        subject="Zeilweekend",
        sender="club@example.com",
        text="We organiseren een zeilweekend.",
        message_ids=["m1"],
    )
    prompt = build_classification_prompt(thread)
    assert "wijn" in prompt and "muziek" in prompt and "zeilen" in prompt
    assert "Zeilweekend" in prompt
