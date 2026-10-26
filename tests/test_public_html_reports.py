"""Public rendering of imported source coverage reports."""

from __future__ import annotations

import re

from monitoring.public_html_reports import _source_text, render_public_source_coverage_html


HISTORICAL_REPORT = """<!doctype html><html><head>
<script>alert('old report script')</script>
</head><body onload="steal()">
<h1><DATASETS>\\datasets\\TestEval\\sample.py</h1>
<div>SFC 80% / NRR 75% / TSQ 60% / SFQ 55% / CSG 90% / PV 50%</div>
<table><thead><tr><th>行</th><th>行覆盖</th><th>证据标签</th><th>源码</th></tr></thead>
<tbody>
<tr class="covered" data-row data-line="1" title="SFC 80%" onclick="steal()">
  <td>1</td><td><span data-metric="line">已执行</span></td>
  <td><span class="tag branch ok" data-metric="branch">分支覆盖</span>
      <span class="tag defect" data-metric="risk">高关注 100%</span></td>
  <td><code>if value &gt; 0:  # SFC D:\\wn\\secret\\source.py</code></td>
</tr>
<tr class="uncovered" data-row data-line="2">
  <td>2</td><td><span data-metric="line">未执行</span></td>
  <td><span data-metric="assertion">强断言 2</span><span data-metric="mutation">变异存活</span></td>
  <td><code>    return "&lt;script&gt;alert(1)&lt;/script&gt;"</code></td>
</tr>
</tbody></table><iframe src="file:///D:/wn/secret"></iframe>
</body></html>"""


def test_preserves_line_heatmap_evidence_without_internal_metrics_or_old_paths():
    rendered = render_public_source_coverage_html(HISTORICAL_REPORT)

    assert 'data-line="1"' in rendered
    assert 'data-line="2"' in rendered
    assert 'class="covered"' in rendered
    assert 'class="uncovered"' in rendered
    assert "分支覆盖" in rendered
    assert "强断言 2" in rendered
    assert "变异存活" in rendered
    assert "if value &gt; 0:" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "高关注 100%" not in rendered
    assert not re.search(r"(?i)(?<![A-Za-z0-9])(?:sfc|nrr|tsq|sfq|csg|pv)(?![A-Za-z0-9])", rendered)
    assert "D:\\wn" not in rendered
    assert "file:///" not in rendered
    assert "<script" not in rendered.lower()
    assert "<iframe" not in rendered.lower()
    assert "onclick=" not in rendered.lower()
    assert "onload=" not in rendered.lower()
    assert "href=" not in rendered.lower()


def test_malformed_report_does_not_pass_through_untrusted_html():
    rendered = render_public_source_coverage_html("<script>alert(1)</script><a href='file:///D:/wn/secret'>old</a>")

    assert "暂无可展示的源码行" in rendered
    assert "<script" not in rendered.lower()
    assert "<a " not in rendered.lower()
    assert "D:/wn" not in rendered


def test_source_projection_preserves_normal_identifiers_without_process_placeholders():
    projected = _source_text("private Integer priority; def calculate_quality_gain(): # SFC 80%")
    assert "priority" in projected
    assert "calculate_quality_gain" in projected
    assert "SFC" not in projected
    assert "内部过程指标已隐藏" not in projected


def test_preserves_source_identifiers_while_hiding_standalone_metric_names():
    saved = (
        '<table><tbody><tr class="covered" data-row data-line="1">'
        '<td>1</td><td>已执行</td><td></td>'
        '<td><code>def calculate_quality_gain():  # SFC 80%, NRR 50%</code></td>'
        '</tr></tbody></table>'
    )

    rendered = render_public_source_coverage_html(saved)

    assert "def calculate_quality_gain():" in rendered
    assert not re.search(r"(?i)(?<![A-Za-z0-9_])(?:sfc|nrr)(?![A-Za-z0-9_])", rendered)
