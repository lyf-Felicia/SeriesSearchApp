from src.text_safety import clean_html_tags


def test_clean_html_tags_removes_markup_and_escapes_entities():
    value = '<script>alert("x")</script> Fish & Chips'

    cleaned = clean_html_tags(value)

    assert "<script>" not in cleaned
    assert cleaned == 'alert(&quot;x&quot;) Fish &amp; Chips'


def test_clean_html_tags_handles_empty_values():
    assert clean_html_tags(None) == ""