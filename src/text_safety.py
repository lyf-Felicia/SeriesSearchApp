import html
import re


def clean_html_tags(text: object) -> str:
    if not text:
        return ""
    without_tags = re.sub(r"<[^>]+>", "", str(text))
    return html.escape(without_tags)