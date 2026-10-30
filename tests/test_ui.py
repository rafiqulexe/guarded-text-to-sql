"""UI page: config rendered as data, untrusted text never parsed as HTML."""

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.config import get_settings
from app.main import app

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_index_renders_badge_cutoffs_from_config(client: httpx.AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    match = re.search(r'<script id="config" type="application/json">(.*?)</script>', response.text)
    assert match
    config = json.loads(match.group(1))
    settings = get_settings()
    assert config["confidenceHigh"] == settings.confidence_high
    assert config["confidenceThreshold"] == settings.confidence_threshold
    assert config["maxQuestionChars"] == settings.max_question_chars
    assert f'maxlength="{settings.max_question_chars}"' in response.text


async def test_index_names_the_product_in_tab_and_header(client: httpx.AsyncClient) -> None:
    html = (await client.get("/")).text
    assert "<title>Guarded Text-to-SQL</title>" in html
    assert '<h1 class="page-title">Guarded Text-to-SQL</h1>' in html


async def test_security_headers_forbid_inline_and_foreign_scripts(
    client: httpx.AsyncClient,
) -> None:
    for path in ("/", "/static/app.js", "/health"):
        csp = (await client.get(path)).headers["content-security-policy"]
        assert "script-src 'self'" in csp and "unsafe-inline" not in csp
        assert "object-src 'none'" in csp


async def test_static_files_are_served(client: httpx.AsyncClient) -> None:
    assert (await client.get("/static/app.js")).status_code == 200
    assert (await client.get("/static/styles.css")).status_code == 200


def test_js_never_parses_untrusted_text_as_html() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "createContextualFragment",
        "DOMParser",
        'setAttribute("on',
    ):
        assert sink not in js, sink
    assert "textContent" in js


def test_template_has_no_unescaped_output_or_inline_script() -> None:
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert "|safe" not in html.replace(" ", "") and "autoescape false" not in html
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", html, re.S)
    for attrs, body in scripts:
        # only data (JSON config) or an external same-origin file
        assert 'type="application/json"' in attrs or (
            body.strip() == "" and 'src="/static/' in attrs
        )
    assert not re.search(r"\son\w+=", html)  # no inline event handlers


def test_css_hex_colors_only_in_token_blocks() -> None:
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    without_tokens = re.sub(r":root[^{]*\{[^}]*\}", "", css)
    without_tokens = re.sub(r"@media \(prefers-color-scheme: light\) \{\s*\}", "", without_tokens)
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", without_tokens)


def test_css_defines_every_tokens_md_color() -> None:
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for token, value in {
        "--bg": "#0E1116",
        "--surface": "#161B22",
        "--surface-2": "#1F2630",
        "--border": "#2D3540",
        "--text": "#E6EDF3",
        "--text-muted": "#8B96A5",
        "--accent": "#3FB68B",
        "--ok": "#3FB68B",
        "--warn": "#E3B341",
        "--danger": "#F06A6A",
        "--info": "#58A6FF",
        "--sql-keyword": "#FF7B72",
    }.items():
        assert re.search(rf"{re.escape(token)}:\s*{value};", css), token


# --- live progress while a question is running ----------------------------------------------------

STEP_LABELS = {
    "RECEIVED": "Checking the question",
    "INPUT_CHECK": "Checking the question",
    "GENERATING": "Writing SQL (LLM)",
    "AST_CHECK": "Running safety checks",
    "SCHEMA_CHECK": "Running safety checks",
    "REWRITING": "Running safety checks",
    "DRY_RUN": "Running safety checks",
    "EXECUTING": "Running the query (read-only)",
    "VALIDATING": "Checking the answer (LLM judge)",
    "SCORING": "Scoring confidence",
}


async def test_index_has_a_hidden_progress_element(client: httpx.AsyncClient) -> None:
    html = (await client.get("/")).text
    progress = re.search(r'<div id="progress"[^>]*>(.*?)</div>', html, re.S)
    assert progress and "hidden" in progress.group(0) and 'role="status"' in progress.group(0)
    for element in ('id="progress-step"', 'id="progress-timer"', 'id="progress-slow"'):
        assert element in progress.group(1)
    slow = re.search(r'<p id="progress-slow"[^>]*>(.*?)</p>', html, re.S)
    assert slow and "hidden" in slow.group(0)
    assert slow.group(1) == "The model is responding slowly; retries are automatic."


def test_js_maps_every_running_state_to_its_step_label() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const STEP_LABELS = \{(.*?)\};", js, re.S)
    assert block
    mapping = dict(re.findall(r'(\w+): "([^"]+)"', block.group(1)))
    assert mapping == STEP_LABELS


def test_js_polls_the_run_with_a_client_id_and_flags_slow_steps() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "crypto.randomUUID" in js and "run_id: runId" in js
    assert "/api/runs/${encodeURIComponent(progress.runId)}" in js
    assert re.search(r"const POLL_MS = (\d+);", js).group(1) == "1000"  # type: ignore[union-attr]
    assert re.search(r"const SLOW_STEP_MS = (\d+);", js).group(1) == "15000"  # type: ignore[union-attr]
    # a late poll must never draw over the final result
    assert "token !== progress.token" in js


def test_progress_styles_use_tokens_only() -> None:
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    rules = re.findall(r"\.progress[\w-]*\s*\{([^}]*)\}", css)
    assert rules
    for body in rules:
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", body)
        for value in re.findall(r"(\d+)px", body):
            assert value == "1", f"size literal {value}px outside :root"
