import pytest

import asyncio
import types

from enkibot.utils import news_channels
from enkibot.utils.news_channels import extract_channel_usernames


def test_extract_channel_usernames_from_links_only_and_sorts():
    html = """
    <img src="avatar@2x.jpg">
    <a href="https://tlgrm.ru/channels/@AlphaNews">Alpha</a>
    <a href='tg://resolve?domain=Beta_updates'>Beta</a>
    <a href="https://tlgrm.ru/channels/@AlphaNews">duplicate</a>
    """
    assert extract_channel_usernames(html) == ["AlphaNews", "Beta_updates"]


def test_fetch_channel_usernames_handles_pagination(monkeypatch):
    page1 = (
        '<div id="vue-channels-list" data-current-page="1" '
        'data-last-page="2" data-path="https://tlgrm.ru/channels/news"></div>'
        '<a href="https://tlgrm.ru/channels/@Alpha">Alpha</a>'
    )
    page2 = '<a href="https://tlgrm.ru/channels/@Beta">Beta</a>'

    class DummyResponse:
        def __init__(self, text: str):
            self.text = text
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self._pages = [page1, page2]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            return DummyResponse(self._pages.pop(0))

    monkeypatch.setattr(
        news_channels,
        "httpx",
        types.SimpleNamespace(AsyncClient=DummyAsyncClient),
    )
    # Limit to a single category to exercise pagination logic
    monkeypatch.setattr(
        news_channels,
        "CHANNEL_CATEGORY_URLS",
        [news_channels.NEWS_CHANNELS_URL],
    )

    result = asyncio.run(news_channels.fetch_channel_usernames())
    assert result == ["Alpha", "Beta"]


def test_fetch_channel_usernames_multiple_categories(monkeypatch):
    tech_page = '<a href="https://tlgrm.ru/channels/@Tech">Tech</a>'
    news_page = '<a href="https://tlgrm.ru/channels/@News">News</a>'

    class DummyResponse:
        def __init__(self, text: str):
            self.text = text
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            # First the news page, then technology page
            self._pages = [news_page, tech_page]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            return DummyResponse(self._pages.pop(0))

    monkeypatch.setattr(
        news_channels,
        "httpx",
        types.SimpleNamespace(AsyncClient=DummyAsyncClient),
    )
    monkeypatch.setattr(
        news_channels,
        "CHANNEL_CATEGORY_URLS",
        [news_channels.NEWS_CHANNELS_URL, news_channels.TECH_CHANNELS_URL],
    )

    result = asyncio.run(news_channels.fetch_channel_usernames())
    assert result == ["News", "Tech"]
