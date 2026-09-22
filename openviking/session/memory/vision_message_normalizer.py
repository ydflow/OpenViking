# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Normalize image messages into extraction-friendly text messages."""

import base64
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional

from openviking.message import Message
from openviking.message.part import ImagePart, TextPart

logger = logging.getLogger(__name__)

IMAGE_DESCRIPTION_PROMPT = (
    "Describe this image for later memory extraction. Focus on durable, user-relevant "
    "details such as visible people, objects, places, actions, text, dates, and other "
    "facts that may matter in future conversations. Return only the description."
)

# Sources the VLM can fetch or already holds as bytes. Anything else is a
# writer-side filesystem path that must be inlined before it leaves this host.
_REMOTE_IMAGE_PREFIXES = ("http://", "https://", "data:")

# Mirrors the extension table in the VLM backends' ``_prepare_image``; unknown
# extensions fall back to PNG exactly as they do there.
_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def message_has_image_part(message: Message) -> bool:
    return any(isinstance(part, ImagePart) for part in getattr(message, "parts", []))


def _local_image_to_data_uri(url: str) -> Optional[str]:
    """Read a local image file and inline it as a base64 data URI.

    Returns ``None`` when the file cannot be read, which is the expected case
    when the writer that captured the path runs on a different host from this
    server: there are no bytes to send, and emitting the path verbatim would
    only reach the model as an unusable ``image_url`` target.
    """
    try:
        data = Path(url).read_bytes()
    except OSError as exc:
        logger.warning("Skipping unreadable image path %s: %s", url, exc)
        return None
    mime_type = _IMAGE_MIME_BY_SUFFIX.get(Path(url).suffix.lower(), "image/png")
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_part_to_openai_content(part: ImagePart) -> Optional[Dict[str, Any]]:
    """Build an OpenAI image content block for an ``ImagePart``.

    ``http(s)`` and ``data:`` sources pass through unchanged. Any other value is
    treated as a local path and inlined as a data URI, because the VLM is called
    with ``messages=`` and its own ``images=`` conversion never runs on this
    path. Returns ``None`` when a local path cannot be read.
    """
    if part.url.startswith(_REMOTE_IMAGE_PREFIXES):
        image_url: Dict[str, Any] = {"url": part.url}
        if part.detail is not None:
            image_url["detail"] = part.detail
        return {"type": "image_url", "image_url": image_url}

    data_uri = _local_image_to_data_uri(part.url)
    if data_uri is None:
        return None
    image_url = {"url": data_uri}
    if part.detail is not None:
        image_url["detail"] = part.detail
    return {"type": "image_url", "image_url": image_url}


def build_vision_description_messages(message: Message) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": IMAGE_DESCRIPTION_PROMPT}]
    for part in getattr(message, "parts", []):
        if isinstance(part, TextPart) and part.text:
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            block = image_part_to_openai_content(part)
            if block is not None:
                content.append(block)
    return [{"role": "user", "content": content}]


def fallback_image_description(message: Message) -> str:
    return ""


def normalize_vision_response(response: Any) -> str:
    content = getattr(response, "content", response)
    return str(content or "").strip()


def _original_text_parts(message: Message) -> List[TextPart]:
    return [
        TextPart(text=part.text)
        for part in getattr(message, "parts", [])
        if isinstance(part, TextPart) and part.text
    ]


async def describe_image_message(
    message: Message,
    *,
    vlm: Any,
    logger: Any = None,
) -> str:
    if vlm is None:
        return fallback_image_description(message)

    try:
        response = await vlm.get_vision_completion_async(
            messages=build_vision_description_messages(message),
            thinking=False,
        )
        description = normalize_vision_response(response)
        return description or fallback_image_description(message)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to describe image message %s: %s", message.id, exc)
        return fallback_image_description(message)


async def replace_image_parts_with_descriptions(
    messages: List[Message],
    *,
    get_vlm: Callable[[], Any],
    logger: Any = None,
) -> List[Message]:
    prepared_messages: List[Message] = []
    for message in messages:
        if not message_has_image_part(message):
            prepared_messages.append(message)
            continue

        description = await describe_image_message(
            message,
            vlm=get_vlm(),
            logger=logger,
        )
        parts = _original_text_parts(message)
        if description:
            parts.append(TextPart(text=f"[Image description]: {description}"))
        if parts:
            prepared_messages.append(
                Message(
                    id=message.id,
                    role=message.role,
                    parts=parts,
                    peer_id=message.peer_id,
                    created_at=message.created_at,
                )
            )
    return prepared_messages
