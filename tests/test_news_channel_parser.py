import pytest

from enkibot.utils.news_channels import extract_channel_usernames


def test_extract_channel_usernames_from_links_only_and_sorts():
    html = """
    <img src="avatar@2x.jpg">
    <a href="https://tlgrm.ru/channels/@AlphaNews">Alpha</a>
    <a href='tg://resolve?domain=Beta_updates'>Beta</a>
    <a href="https://tlgrm.ru/channels/@AlphaNews">duplicate</a>
    """
    assert extract_channel_usernames(html) == ["AlphaNews", "Beta_updates"]
