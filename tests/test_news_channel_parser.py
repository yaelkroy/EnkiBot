import pytest

from enkibot.utils.news_channels import extract_channel_usernames


def test_extract_channel_usernames_deduplicates_and_sorts():
    html = """
    <div>@AlphaNews</div>
    <span>@Beta_updates</span>
    <p>Duplicate @AlphaNews mention</p>
    """
    assert extract_channel_usernames(html) == ["AlphaNews", "Beta_updates"]
