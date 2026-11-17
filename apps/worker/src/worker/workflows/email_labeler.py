"""Daily Gmail labeling workflow.

Fetches recent unlabeled threads from Gmail, classifies each one with a
Mistral model into Wijn, Muziek or Zeilen, and applies the matching Gmail
label. Designed to run unattended on a daily schedule, so the Gmail
connector resolves the deployment's service credentials.
"""

from __future__ import annotations

import json
from typing import Any

import pydantic
import structlog

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

logger = structlog.get_logger(__name__)

gmail_connector = connector("gmail", run_as="deployment")

LABEL_NAMES = ("Wijn", "Muziek", "Zeilen")
MODEL = "mistral-small-latest"
DEFAULT_DAYS_BACK = 3
DEFAULT_MAX_THREADS = 25
MAX_CHARS_PER_EMAIL = 2000
MAX_CHARS_PER_THREAD = 6000

CATEGORY_KEYS = {"wijn": "Wijn", "muziek": "Muziek", "zeilen": "Zeilen"}


class LabelerInput(pydantic.BaseModel):
    dry_run: bool = False
    days_back: int = DEFAULT_DAYS_BACK
    max_threads: int = DEFAULT_MAX_THREADS


class ThreadSummary(pydantic.BaseModel):
    thread_id: str
    subject: str = ""
    sender: str = ""
    text: str = ""
    message_ids: list[str] = pydantic.Field(default_factory=list)


class Classification(pydantic.BaseModel):
    category: str


class ThreadResult(pydantic.BaseModel):
    thread_id: str
    subject: str
    category: str
    labeled: bool


class LabelerSummary(pydantic.BaseModel):
    threads_seen: int
    labeled: int
    dry_run: bool
    details: list[ThreadResult] = pydantic.Field(default_factory=list)


def build_search_query(days_back: int) -> str:
    excluded = " ".join(f"-label:{name.lower()}" for name in LABEL_NAMES)
    return f"newer_than:{max(1, days_back)}d {excluded} -in:sent -in:chats -in:spam -in:trash"


def normalize_category(raw: str) -> str:
    return CATEGORY_KEYS.get(raw.strip().lower(), "geen")


def _first_str(obj: Any, *keys: str) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tool_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts)
        structured = result.get("structuredContent")
        if structured is not None:
            return json.dumps(structured)
    return json.dumps(result)


def _tool_json(result: Any) -> Any:
    text = _tool_text(result)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _extract_threads(result: Any) -> list[dict[str, Any]]:
    payload = _tool_json(result)
    if isinstance(payload, dict):
        for key in ("threads", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _extract_labels(result: Any) -> list[dict[str, Any]]:
    payload = _tool_json(result)
    if isinstance(payload, dict):
        value = payload.get("labels")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _message_text(message: dict[str, Any]) -> str:
    body = _first_str(message, "plaintextBody", "plainTextBody", "snippet", "body", "text")
    if not body:
        body = _first_str(message, "htmlBody")
    return body[:MAX_CHARS_PER_EMAIL]


def extract_thread_summaries(result: Any) -> list[ThreadSummary]:
    summaries: list[ThreadSummary] = []
    for raw in _extract_threads(result):
        thread_id = _first_str(raw, "id", "threadId")
        if not thread_id:
            continue
        messages = raw.get("messages")
        messages = messages if isinstance(messages, list) else []
        message_dicts = [m for m in messages if isinstance(m, dict)]
        subject = _first_str(raw, "subject", "title")
        sender = _first_str(raw, "from", "sender")
        if not message_dicts:
            text = _first_str(raw, "snippet", "plaintextBody", "body")
        else:
            subject = subject or _first_str(message_dicts[0], "subject")
            sender = sender or _first_str(message_dicts[0], "from", "sender", "fromField")
            parts = [_message_text(m) for m in message_dicts]
            text = ""
            for part in parts:
                if len(text) + len(part) > MAX_CHARS_PER_THREAD:
                    text += part[: MAX_CHARS_PER_THREAD - len(text)]
                    break
                text += part
        message_ids = [
            mid
            for mid in (_first_str(m, "id", "messageId") for m in message_dicts)
            if mid
        ]
        summaries.append(
            ThreadSummary(
                thread_id=thread_id,
                subject=subject,
                sender=sender,
                text=text,
                message_ids=message_ids,
            )
        )
    return summaries


def build_classification_prompt(thread: ThreadSummary) -> str:
    prompt = (
        "Klassificeer de volgende e-mail in exact één categorie: "
        "'wijn', 'muziek', 'zeilen' of 'geen'.\n"
        "- wijn: wijnaanbiedingen, wijnhuizen, proeverijen, wijnabonnementen, wijnnieuws.\n"
        "- muziek: concerten, festivals, platenuitgaven, muziekdiensten, instrumenten, muzieklessen.\n"
        "- zeilen: zeilboten, zeilvakanties, zeillessen, botenverhuur, maritieme uitrusting, zeilclubs.\n"
        "- geen: alles wat niet bij een van de drie categorieën hoort.\n\n"
        f"Afzender: {thread.sender}\n"
        f"Onderwerp: {thread.subject}\n\n"
        f"Inhoud:\n{thread.text or '(geen inhoud)'}"
    )
    return prompt


@workflows.activity()
async def classify_thread(thread: ThreadSummary) -> Classification:
    from mistralai.workflows.plugins.mistralai import get_mistral_client

    client = get_mistral_client()
    response = await client.chat.complete_async(
        model=MODEL,
        messages=[{"role": "user", "content": build_classification_prompt(thread)}],
        temperature=0.0,
    )
    content = ""
    if response.choices and response.choices[0].message:
        content = response.choices[0].message.content or ""
    return Classification(category=normalize_category(content))


@workflows.activity()
async def ensure_labels(gmail: ToolCallClient = Depends(gmail_connector)) -> dict[str, str]:
    result = await gmail.call_tool("list_labels", {})
    existing: dict[str, str] = {}
    for label in _extract_labels(result):
        label_id = _first_str(label, "labelId", "id")
        name = _first_str(label, "name", "displayName")
        if label_id and name:
            existing[name] = label_id

    label_ids: dict[str, str] = {}
    for name in LABEL_NAMES:
        if name in existing:
            label_ids[name] = existing[name]
            continue
        created = await gmail.call_tool("create_label", {"displayName": name})
        payload = _tool_json(created)
        label_id = _first_str(payload, "labelId", "id") if isinstance(payload, dict) else ""
        if not label_id:
            raise RuntimeError(f"Failed to create Gmail label {name!r}")
        label_ids[name] = label_id
    return label_ids


@workflows.activity()
async def find_unlabeled_threads(
    days_back: int,
    max_threads: int,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> list[ThreadSummary]:
    query = build_search_query(days_back)
    summaries: list[ThreadSummary] = []
    page_token: str | None = None
    while len(summaries) < max_threads:
        arguments: dict[str, Any] = {"query": query, "pageSize": min(20, max_threads - len(summaries))}
        if page_token:
            arguments["pageToken"] = page_token
        result = await gmail.call_tool("search_threads", arguments)
        summaries.extend(extract_thread_summaries(result))
        payload = _tool_json(result)
        next_token = _first_str(payload, "nextPageToken", "next_page_token") if isinstance(payload, dict) else ""
        if not next_token:
            break
        page_token = next_token
    return summaries[:max_threads]


@workflows.activity()
async def apply_label(
    thread_id: str,
    message_ids: list[str],
    label_id: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> None:
    try:
        await gmail.call_tool("label_thread", {"threadId": thread_id, "labelIds": [label_id]})
        return
    except Exception as thread_error:
        logger.warning(
            "label_thread failed, falling back to per-message labels",
            thread_id=thread_id,
            error=str(thread_error),
        )
    for message_id in message_ids:
        await gmail.call_tool("label_message", {"messageId": message_id, "labelIds": [label_id]})


@workflows.workflow.define(name="email-labeler", workflow_display_name="Email Labeler")
@uses_connectors(gmail_connector)
class EmailLabelerWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: LabelerInput | None = None) -> LabelerSummary:
        params = input or LabelerInput()
        label_ids = await ensure_labels()
        threads = await find_unlabeled_threads(days_back=params.days_back, max_threads=params.max_threads)

        results: list[ThreadResult] = []
        for thread in threads:
            classification = await classify_thread(thread)
            category = classification.category
            labeled = False
            if category in CATEGORY_KEYS and not params.dry_run:
                await apply_label(
                    thread_id=thread.thread_id,
                    message_ids=thread.message_ids,
                    label_id=label_ids[category],
                )
                labeled = True
            results.append(
                ThreadResult(
                    thread_id=thread.thread_id,
                    subject=thread.subject,
                    category=category,
                    labeled=labeled,
                )
            )

        return LabelerSummary(
            threads_seen=len(results),
            labeled=sum(1 for r in results if r.labeled),
            dry_run=params.dry_run,
            details=results,
        )
