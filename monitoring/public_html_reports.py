"""Safe, self-contained views of historical HTML reports.

Imported source coverage pages can contain useful line-level evidence even
when their source paths no longer exist. Parse that evidence from the saved
page and render it with a small fixed HTML vocabulary instead of replaying
historical scripts, links, paths, or process-only metrics.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from core.public_metrics import public_text


_OLD_WINDOWS_PATH = re.compile(r"(?i)(?:file:/+)?/?[a-z]:[\\/][^\s<>\"'`]+")
_SOURCE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PROCESS_ACRONYMS = {"sfc", "nrr", "tsq", "sfq", "csg", "pv", "dvs", "tic", "rcs"}
_ROW_TONES = {"covered", "uncovered", "neutral", "partial"}
_PUBLIC_EVIDENCE = {"branch", "assertion", "mutation"}
_IGNORED_ELEMENTS = {"script", "style", "iframe", "object", "embed", "svg", "math", "form"}


def _visible_text(value: str) -> str:
    return _OLD_WINDOWS_PATH.sub("历史路径已隐藏", public_text(value))


def _source_text(value: str) -> str:
    """Keep ordinary code identifiers while hiding standalone metric labels."""
    preserved: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        identifier = match.group(0)
        projected = public_text(identifier)
        if projected == identifier or identifier.casefold() in _PROCESS_ACRONYMS:
            return identifier
        marker = f"\ue000{len(preserved)}\ue001"
        preserved[marker] = identifier
        return marker

    masked = _SOURCE_IDENTIFIER.sub(protect, value)
    cleaned = public_text(masked)
    for marker, identifier in preserved.items():
        cleaned = cleaned.replace(marker, identifier)
    cleaned = cleaned.replace("内部过程指标已隐藏", "…")
    return _OLD_WINDOWS_PATH.sub("历史路径已隐藏", cleaned)


class _SourceRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[int, str, list[str], str]] = []
        self._ignored: list[str] = []
        self._in_body = False
        self._row: dict[str, object] | None = None
        self._cell = -1
        self._evidence_metric: str | None = None
        self._evidence_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _IGNORED_ELEMENTS:
            self._ignored.append(tag)
            return
        if self._ignored:
            return
        attributes = dict(attrs)
        if tag == "tbody":
            self._in_body = True
        elif tag == "tr" and self._in_body:
            line = attributes.get("data-line") or ""
            if not line.isdecimal():
                return
            classes = set((attributes.get("class") or "").split())
            tone = next((name for name in ("covered", "uncovered", "partial", "neutral") if name in classes), "neutral")
            self._row = {"line": int(line), "tone": tone, "evidence": [], "source": []}
            self._cell = -1
        elif tag == "td" and self._row is not None:
            self._cell += 1
        elif tag == "span" and self._row is not None and self._cell == 2:
            metric = attributes.get("data-metric")
            if metric in _PUBLIC_EVIDENCE:
                self._evidence_metric = metric
                self._evidence_text = []

    def handle_data(self, data: str) -> None:
        if self._ignored or self._row is None:
            return
        if self._cell == 3:
            source = self._row["source"]
            assert isinstance(source, list)
            source.append(data)
        elif self._cell == 2 and self._evidence_metric:
            self._evidence_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._ignored:
            if tag == self._ignored[-1]:
                self._ignored.pop()
            return
        if tag == "span" and self._evidence_metric and self._row is not None:
            original = "".join(self._evidence_text).strip()
            if original and public_text(original) == original:
                evidence = self._row["evidence"]
                assert isinstance(evidence, list)
                evidence.append(_visible_text(original))
            self._evidence_metric = None
            self._evidence_text = []
        elif tag == "tr" and self._row is not None:
            source = self._row["source"]
            evidence = self._row["evidence"]
            assert isinstance(source, list) and isinstance(evidence, list)
            self.rows.append((
                int(self._row["line"]),
                str(self._row["tone"]),
                evidence,
                _source_text("".join(source)),
            ))
            self._row = None
            self._cell = -1
            self._evidence_metric = None
        elif tag == "tbody":
            self._in_body = False


def render_public_source_coverage_html(content: str) -> str:
    """Preserve saved source rows while excluding historical active content."""
    parser = _SourceRows()
    parser.feed(content)
    parser.close()

    rows = []
    for line, tone, evidence, source in parser.rows:
        if tone not in _ROW_TONES:
            tone = "neutral"
        status = {"covered": "已覆盖", "uncovered": "未覆盖", "partial": "部分覆盖", "neutral": "不适用"}[tone]
        labels = " ".join(f"<span>{html.escape(label)}</span>" for label in evidence) or "无定位证据"
        rows.append(
            f'<tr class="{tone}" data-line="{line}"><td>{line}</td><td>{status}</td>'
            f'<td>{labels}</td><td><code>{html.escape(source)}</code></td></tr>'
        )
    body = "".join(rows) if rows else '<tr><td colspan="4">暂无可展示的源码行</td></tr>'
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>源码覆盖报告</title><style>'
        'body{font-family:system-ui,sans-serif;color:#1b2638;background:#f5f7fb;margin:24px}'
        'main{max-width:1400px;margin:auto}h1{font-size:24px}p{color:#516173}'
        'table{width:100%;border-collapse:collapse;background:white}th,td{border:1px solid #dce3ed;'
        'padding:7px 10px;text-align:left;vertical-align:top}th{background:#e9eef7;position:sticky;top:0}'
        'tr.covered{background:#e9f7ee}tr.uncovered{background:#fff0ef}'
        'tr.partial{background:#fff7e3}td:first-child{width:4rem}td:nth-child(2){width:6rem}'
        'td:nth-child(3){width:18rem}td span{display:inline-block;margin:2px;padding:2px 6px;'
        'background:#e9eef7;border-radius:4px}code{white-space:pre-wrap;overflow-wrap:anywhere}'
        '</style></head><body><main><h1>源码覆盖报告</h1>'
        '<p>以下行级覆盖与测试证据来自已保存的实验报告。</p>'
        '<table><thead><tr><th>行</th><th>行覆盖</th><th>证据标签</th><th>源码</th></tr></thead>'
        f'<tbody>{body}</tbody></table></main></body></html>'
    )
