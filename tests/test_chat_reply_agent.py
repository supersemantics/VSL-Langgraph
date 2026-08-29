import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "chat-conversation"))

import chat_fake_providers as fake_providers  # noqa: E402
from chat_reply_agent import build_agent  # noqa: E402

_MESSAGE = "My order hasn't arrived and it's been two weeks, what happens now?"


async def test_resolved_reply_is_sent():
    fake_providers.set_scenario("resolved")
    result = await build_agent().ainvoke({"customer_message": _MESSAGE})
    assert result.get("sent_reply") is not None
    assert result.get("guarantee_denial") is None
    assert result.get("confidence_denial") is None
    assert result.get("handed_off") is None


async def test_unverified_guarantee_is_handed_off_not_sent():
    fake_providers.set_scenario("unverified_guarantee")
    result = await build_agent().ainvoke({"customer_message": _MESSAGE})
    assert result.get("sent_reply") is None
    assert result.get("guarantee_denial") is not None
    assert result.get("handed_off") is True


async def test_low_confidence_is_handed_off_not_sent():
    fake_providers.set_scenario("low_confidence")
    result = await build_agent().ainvoke({"customer_message": _MESSAGE})
    assert result.get("sent_reply") is None
    assert result.get("confidence_denial") is not None
    assert result.get("handed_off") is True
