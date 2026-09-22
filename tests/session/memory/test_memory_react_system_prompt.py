# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Test that provider instruction correctly instructs LLM.
"""

import base64

from openviking.message import ImagePart, Message, TextPart, ToolPart
from openviking.session.memory.session_extract_context_provider import SessionExtractContextProvider
from openviking.session.memory.vision_message_normalizer import (
    IMAGE_DESCRIPTION_PROMPT,
    image_part_to_openai_content,
)


class TestProviderInstruction:
    """Test the provider instruction contains correct instructions."""

    def test_instruction_contains_read_before_edit_instructions(self):
        """Test that instruction explicitly tells LLM to read files before editing."""
        # Create provider with mock messages
        mock_messages = []
        provider = SessionExtractContextProvider(messages=mock_messages)

        instruction = provider.instruction()

        # Check for critical instructions
        assert (
            "Before editing ANY existing memory file, you MUST first read its complete content"
            in instruction
        )
        assert (
            "ONLY read URIs that are explicitly listed in ls/search tool results, returned by previous tool calls"
            in instruction
        )

    def test_instruction_includes_extraction_and_maintenance_objective(self):
        provider = SessionExtractContextProvider(messages=[])

        instruction = provider.instruction()

        assert (
            "You are a memory extraction and maintenance agent. Analyze the conversation "
            "together with the pre-fetched existing memories, then produce a single atomic "
            "plan to create, update, delete, or reorganize memories so that the final memory "
            "collection conforms to all enabled memory schemas." in instruction
        )

    def test_instruction_contains_output_language(self):
        """Test that instruction includes the output language setting."""
        mock_messages = []
        provider = SessionExtractContextProvider(messages=mock_messages)

        instruction = provider.instruction()

        # Check that output language instruction is present
        assert "Target Output Language" in instruction
        assert "All memory content MUST be written in" in instruction

    def test_instruction_explains_peer_memory_routing(self):
        provider = SessionExtractContextProvider(messages=[])

        instruction = provider.instruction()

        assert "Peer Memory" in instruction
        assert "Message role is authoritative for newly extracted facts" in instruction
        assert "attribute each fact to the speaker whose" in instruction
        assert "follow each enabled memory type's own schema rules" in instruction
        assert "Do not infer ownership from neighboring messages" in instruction
        assert "Facts already stored in pre-fetched or explicitly read memories" in instruction
        assert "remain valid sources for" in instruction
        assert "maintenance and may be preserved" in instruction
        assert "or moved between enabled memory" in instruction
        assert "types when required by their schemas" in instruction
        assert "same identity under the active memory schema" in instruction
        assert "shared category, or an overlapping topic" in instruction
        assert "If identity is uncertain" in instruction
        assert "compact" in instruction.lower()
        assert "only duplicate wording" in instruction.lower()

    def test_instruction_omits_resource_uri_handling_without_resource_uri(self):
        provider = SessionExtractContextProvider(
            messages=[Message(id="m1", role="user", parts=[TextPart("我喜欢越前龙马。")])]
        )

        instruction = provider.instruction()

        assert "Resource URI Handling" not in instruction
        assert "Affected memory URIs" not in instruction

    def test_instruction_includes_resource_uri_handling_for_user_scoped_resource_uri(self):
        provider = SessionExtractContextProvider(
            messages=[
                Message(
                    id="m1",
                    role="user",
                    parts=[
                        TextPart(
                            "这张图是越前龙马："
                            "viking://user/ryoma/peers/fuji/resources/images/yueqian_jpeg"
                        )
                    ],
                )
            ]
        )

        instruction = provider.instruction()

        assert "Resource URI Handling" in instruction
        assert "viking://user/{user_id}/resources/..." in instruction
        assert "viking://user/{user_id}/peers/{peer_id}/resources/..." in instruction
        assert (
            "system-generated `## Resource Deletion` block's `Affected memory URIs`" in instruction
        )


class TestSessionConversationToolFiltering:
    def test_session_conversation_omits_skill_tool_call(self):
        messages = [
            Message(
                id="m1",
                role="assistant",
                parts=[
                    TextPart("Running a skill."),
                    ToolPart(
                        tool_id="tool_1",
                        tool_name="read",
                        tool_uri="viking://session/test/tools/tool_1",
                        skill_uri="viking://user/skills/create_presentation",
                        tool_input={"file_path": "/skills/ppt/SKILL.md"},
                        tool_output="ok",
                        tool_status="completed",
                        duration_ms=123,
                    ),
                ],
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)

        conversation = provider._assemble_conversation(messages)

        assert "ToolCall:" not in conversation
        assert "create_presentation" not in conversation
        assert "Running a skill." in conversation

    def test_session_conversation_omits_regular_tool_call(self):
        messages = [
            Message(
                id="m1",
                role="assistant",
                parts=[
                    TextPart("Running a tool."),
                    ToolPart(
                        tool_id="tool_1",
                        tool_name="read",
                        tool_uri="viking://session/test/tools/tool_1",
                        tool_input={"file_path": "README.md"},
                        tool_output="ok",
                        tool_status="completed",
                        duration_ms=123,
                    ),
                ],
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)

        conversation = provider._assemble_conversation(messages)

        assert "ToolCall:" not in conversation
        assert "tool_name=read" not in conversation
        assert "Running a tool." in conversation

    def test_agent_provider_conversation_includes_tool_call_evidence(self):
        from openviking.session.memory.agent_trajectory_context_provider import (
            AgentTrajectoryContextProvider,
        )

        messages = [
            Message(
                id="m1",
                role="assistant",
                parts=[
                    TextPart("Checking the reservation."),
                    ToolPart(
                        tool_id="tool_1",
                        tool_name="get_reservation_details",
                        tool_input={"reservation_id": "EHGLP3"},
                        tool_output="available",
                        tool_status="completed",
                    ),
                ],
            )
        ]
        provider = AgentTrajectoryContextProvider(messages=messages)

        conversation = provider._assemble_conversation(messages)

        assert "ToolCall: tool_name=get_reservation_details" in conversation
        assert "input={'reservation_id': 'EHGLP3'}" in conversation
        assert "output=available" in conversation

    def test_assemble_conversation_uses_peer_id_when_present(self):
        messages = [
            Message(
                id="m1",
                role="user",
                parts=[TextPart("My invoice is still missing.")],
                peer_id="web-visitor-alice",
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)

        conversation = provider._assemble_conversation(messages)

        assert "[0][user][web-visitor-alice]" in conversation
        assert "[0][user][default]" not in conversation

    def test_detect_language_only_uses_text_parts(self):
        messages = [
            Message(
                id="m1",
                role="assistant",
                parts=[TextPart("Please keep the memory in English.")],
            ),
            Message(
                id="m2",
                role="assistant",
                parts=[
                    ToolPart(
                        tool_id="tool_1",
                        tool_name="read",
                        tool_uri="viking://session/test/tools/tool_1",
                        tool_input={"file_path": "README.md"},
                        tool_output="这是中文工具输出",
                        tool_status="completed",
                    )
                ],
            ),
        ]

        provider = SessionExtractContextProvider(messages=messages)

        assert provider._detect_language() == "en"

    def test_detect_language_prefers_user_text_over_assistant_text(self):
        messages = [
            Message(
                id="m1",
                role="user",
                parts=[TextPart("请把记忆保持为中文，继续优化。")],
            ),
            Message(
                id="m2",
                role="assistant",
                parts=[TextPart("한국어 응답이 섞였습니다")],
            ),
        ]

        provider = SessionExtractContextProvider(messages=messages)

        assert provider._detect_language() == "zh-CN"

    async def test_prepare_extraction_messages_replaces_image_part_with_vlm_description(self):
        class FakeVisionVLM:
            def __init__(self):
                self.messages = None

            async def get_vision_completion_async(self, **kwargs):
                self.messages = kwargs.get("messages")
                return "A high level image description."

        messages = [
            Message(
                id="m1",
                role="user",
                parts=[
                    TextPart("Please remember this image."),
                    ImagePart(url="https://example.com/image.png", detail="auto"),
                ],
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)
        fake_vlm = FakeVisionVLM()
        provider._vision_vlm = fake_vlm

        await provider.prepare_extraction_messages()
        prompt_message = provider._build_conversation_message()

        assert "Please remember this image." in prompt_message["content"]
        assert "A high level image description." in prompt_message["content"]
        assert "https://example.com/image.png" not in prompt_message["content"]
        assert fake_vlm.messages == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": IMAGE_DESCRIPTION_PROMPT,
                    },
                    {"type": "text", "text": "Please remember this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.png",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

    async def test_prepare_extraction_messages_skips_image_only_without_description(self):
        messages = [
            Message(
                id="m1",
                role="user",
                parts=[ImagePart(url="https://example.com/private-family-photo.png")],
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)
        provider._vision_vlm = None

        await provider.prepare_extraction_messages()
        prompt_message = provider._build_conversation_message()

        assert "[Image description]" not in prompt_message["content"]
        assert "https://example.com/private-family-photo.png" not in prompt_message["content"]

    async def test_prepare_extraction_messages_keeps_text_when_image_is_undescribed(self):
        messages = [
            Message(
                id="m1",
                role="user",
                parts=[
                    TextPart("Please remember the surrounding text."),
                    ImagePart(url="https://example.com/private-family-photo.png"),
                ],
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)
        provider._vision_vlm = None

        await provider.prepare_extraction_messages()
        prompt_message = provider._build_conversation_message()

        assert "Please remember the surrounding text." in prompt_message["content"]
        assert "[Image description]" not in prompt_message["content"]
        assert "https://example.com/private-family-photo.png" not in prompt_message["content"]

    async def test_prepare_extraction_messages_does_not_replace_caller_message_list(self):
        messages = [
            Message(
                id="m1",
                role="user",
                parts=[
                    TextPart("Please remember this image."),
                    ImagePart(url="https://example.com/image.png"),
                ],
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)
        provider._vision_vlm = None

        await provider.prepare_extraction_messages()

        assert len(messages) == 1
        assert any(isinstance(part, ImagePart) for part in messages[0].parts)
        assert provider.messages is not messages


class TestImagePartUrlNormalization:
    """A local ``ImagePart.url`` must reach the VLM as a data URI, not a raw path.

    ``get_vision_completion_async(messages=...)`` takes the branch that leaves
    image content untouched, so the normalizer is the only place that can turn a
    writer-side filesystem path into bytes the model can actually see.
    """

    PNG_BYTES = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    )

    def _write(self, tmp_path, name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    async def test_local_png_path_is_sent_as_data_uri(self, tmp_path):
        path = self._write(tmp_path, "sample.png", self.PNG_BYTES)

        content = image_part_to_openai_content(ImagePart(url=path))

        assert content == {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(self.PNG_BYTES).decode()}"
            },
        }
        assert path not in content["image_url"]["url"]

    async def test_local_jpeg_path_uses_jpeg_mime(self, tmp_path):
        path = self._write(tmp_path, "sample.jpg", b"\xff\xd8\xff\xe0" + b"0" * 32)

        content = image_part_to_openai_content(ImagePart(url=path))

        assert content["image_url"]["url"].startswith("data:image/jpeg;base64,")

    async def test_unknown_extension_falls_back_to_png_mime(self, tmp_path):
        path = self._write(tmp_path, "sample.xyz", self.PNG_BYTES)

        content = image_part_to_openai_content(ImagePart(url=path))

        assert content["image_url"]["url"].startswith("data:image/png;base64,")

    async def test_https_url_passes_through_with_detail(self):
        content = image_part_to_openai_content(
            ImagePart(url="https://example.com/image.png", detail="auto")
        )

        assert content == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png", "detail": "auto"},
        }

    async def test_data_uri_passes_through_unchanged(self):
        data_uri = "data:image/png;base64,iVBORw0KGgo="

        content = image_part_to_openai_content(ImagePart(url=data_uri))

        assert content == {"type": "image_url", "image_url": {"url": data_uri}}

    async def test_unreadable_local_path_drops_the_image_part(self, tmp_path, caplog):
        missing = str(tmp_path / "does-not-exist.png")

        content = image_part_to_openai_content(ImagePart(url=missing))

        assert content is None
        assert "does-not-exist.png" in caplog.text

    async def test_unreadable_local_path_keeps_text_and_skips_image(self, tmp_path):
        """The text part survives; only the unusable image part is dropped."""
        missing = str(tmp_path / "does-not-exist.png")

        captured = {}

        class FakeVisionVLM:
            async def get_vision_completion_async(self, **kwargs):
                captured["messages"] = kwargs.get("messages")
                return "A description."

        messages = [
            Message(
                id="m1",
                role="user",
                parts=[TextPart("Please remember this image."), ImagePart(url=missing)],
            )
        ]
        provider = SessionExtractContextProvider(messages=messages)
        provider._vision_vlm = FakeVisionVLM()

        await provider.prepare_extraction_messages()
        prompt_message = provider._build_conversation_message()

        assert "Please remember this image." in prompt_message["content"]
        assert "A description." in prompt_message["content"]
        assert missing not in prompt_message["content"]
        # The unusable image part never reaches the VLM request.
        assert captured["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                    {"type": "text", "text": "Please remember this image."},
                ],
            }
        ]


def test_session_provider_empty_messages_still_uses_environment_fallback(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    provider = SessionExtractContextProvider(messages=[])

    assert provider.get_output_language() == "zh-CN"
