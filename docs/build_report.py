"""Render the three-page ChurnGuard AI report from real run artifacts.

    backend\\.venv\\Scripts\\python.exe docs/build_report.py

Inputs are produced by the repository itself:
  * docs/training_run.log              (full train.py output, EXITCODE=0)
  * backend/models/model_metadata.json (metrics, leakage probe, validation)
  * backend/reports/*.png              (SHAP summary + force plot)
  * docs/report_assets/*               (live GUI + /metrics captures)
  * backend/Dockerfile, k8s/, terraform/

The output is docs/ChurnGuard_Report.pdf, exactly three A4 pages.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "report_assets"
REPORTS = ROOT / "backend" / "reports"
MODELS = ROOT / "backend" / "models"
PDF_PATH = DOCS / "ChurnGuard_Report.pdf"

INK = colors.HexColor("#10231f")
MUTED = colors.HexColor("#4f6a62")
GREEN = colors.HexColor("#146c43")
DANGER = colors.HexColor("#9a1e1e")
AMBER = colors.HexColor("#8a5a00")
LIGHT = colors.HexColor("#eef4ef")
PALE_RED = colors.HexColor("#fdf0ee")
PALE_GREEN = colors.HexColor("#f0f7f2")

CONTENT_W = A4[0] - 2.3 * cm  # 18.7 cm

CONSOLA = "C:/Windows/Fonts/consola.ttf"
CONSOLAB = "C:/Windows/Fonts/consolab.ttf"
SEGOE = "C:/Windows/Fonts/segoeui.ttf"
SEGOEB = "C:/Windows/Fonts/segoeuib.ttf"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
H1 = ParagraphStyle(
    "H1", fontName="Helvetica-Bold", fontSize=16.5, leading=19, textColor=GREEN, spaceAfter=2
)
SUB = ParagraphStyle(
    "SUB", fontName="Helvetica", fontSize=8.4, leading=10.2, textColor=MUTED, spaceAfter=5
)
H2 = ParagraphStyle(
    "H2",
    fontName="Helvetica-Bold",
    fontSize=9.8,
    leading=11.6,
    textColor=GREEN,
    spaceBefore=3,
    spaceAfter=2,
)
H3 = ParagraphStyle(
    "H3",
    fontName="Helvetica-Bold",
    fontSize=8.3,
    leading=9.8,
    textColor=INK,
    spaceBefore=1,
    spaceAfter=1,
)
BODY = ParagraphStyle(
    "BODY", fontName="Helvetica", fontSize=8.35, leading=10.5, textColor=INK, spaceAfter=2.5
)
SMALL = ParagraphStyle(
    "SMALL", fontName="Helvetica", fontSize=7.55, leading=9.4, textColor=INK, spaceAfter=2
)
CAPTION = ParagraphStyle(
    "CAPTION",
    fontName="Helvetica-Oblique",
    fontSize=6.8,
    leading=8.1,
    textColor=MUTED,
    spaceBefore=1,
    spaceAfter=2,
)
CODE = ParagraphStyle(
    "CODE", fontName="Courier", fontSize=5.9, leading=7.0, textColor=INK
)
CODE_SM = ParagraphStyle(
    "CODE_SM", fontName="Courier", fontSize=5.45, leading=6.5, textColor=INK
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _img(path: Path, width_cm: float) -> Image:
    with PILImage.open(path) as im:
        w, h = im.size
    width = width_cm * cm
    return Image(str(path), width=width, height=width * h / w)


def _read_log() -> list[str]:
    data = (DOCS / "training_run.log").read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8", errors="replace")
    raw = text.splitlines()
    start = next(i for i, line in enumerate(raw) if line.startswith("ChurnGuard AI -"))
    return raw[start:]


def _section(lines: list[str], first: str, stop: str) -> list[str]:
    i = next(i for i, line in enumerate(lines) if line.startswith(first))
    j = next((j for j in range(i + 1, len(lines)) if lines[j].startswith(stop)), len(lines))
    return lines[i:j]


def _terminal(
    lines: list[str],
    title: str,
    filename: str,
    *,
    width: int = 1800,
    font_size: int = 17,
    max_lines: int | None = None,
) -> Path:
    font = ImageFont.truetype(CONSOLA, font_size)
    bold = ImageFont.truetype(CONSOLAB, font_size)
    title_font = ImageFont.truetype(SEGOE, 15)
    pad_x = 22
    max_chars = int((width - 2 * pad_x) / font.getlength("M"))

    wrapped: list[str] = []
    for raw in lines:
        raw = raw.rstrip()
        if len(raw) <= max_chars:
            wrapped.append(raw)
        else:
            wrapped.extend(
                textwrap.wrap(
                    raw,
                    max_chars,
                    subsequent_indent="    ",
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
            )
    if max_lines is not None and len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines] + ["    ... output truncated for the report ..."]

    line_h = int(font_size * 1.38)
    top = 54
    height = top + 18 + line_h * max(len(wrapped), 1) + 20
    canvas = PILImage.new("RGB", (width, height), (24, 28, 33))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 44], fill=(31, 36, 43))
    for i, colour in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        draw.ellipse([18 + i * 22, 16, 28 + i * 22, 26], fill=colour)
    draw.text((96, 13), title, font=title_font, fill=(154, 167, 180))

    y = top + 18
    for line in wrapped:
        heading = line.startswith(
            (
                "2.",
                "3.",
                "4.",
                "5.",
                "6.",
                "7.",
                "FINAL",
                "Tool comparison",
                "PyCaret leaderboard",
                "Fairness slices",
            )
        )
        if "[PASS]" in line:
            colour = (110, 231, 168)
        elif "[WARNING]" in line:
            colour = (251, 191, 36)
        elif "[ACTION]" in line:
            colour = (147, 197, 253)
        elif "FAIL" in line:
            colour = (252, 165, 165)
        elif heading:
            colour = (255, 255, 255)
        elif "winner" in line or "Optuna:" in line or "Overall:" in line:
            colour = (167, 243, 208)
        else:
            colour = (215, 224, 234)
        draw.text((pad_x, y), line, font=bold if heading else font, fill=colour)
        y += line_h

    path = ASSETS / filename
    canvas.save(path)
    return path


def _code_box(text: str, *, small: bool = False) -> Table:
    style = CODE_SM if small else CODE
    table = Table([[Preformatted(text, style)]], colWidths=[6.15 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f7f4")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d8cd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _callout(text: str, *, colour: colors.Color = PALE_GREEN) -> Table:
    table = Table([[Paragraph(text, SMALL)]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colour),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d8cd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c9d8cd"))
    canvas.setLineWidth(0.4)
    canvas.line(1.15 * cm, 0.82 * cm, A4[0] - 1.15 * cm, 0.82 * cm)
    canvas.setFont("Helvetica", 6.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.15 * cm, 0.5 * cm, "ChurnGuard AI  ·  AutoML + XAI + DevOps")
    canvas.drawRightString(A4[0] - 1.15 * cm, 0.5 * cm, f"Page {doc.page} of 3")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build() -> Path:
    metadata = json.loads((MODELS / "model_metadata.json").read_text(encoding="utf-8"))
    metrics = metadata["metrics"]
    sample = json.loads((ASSETS / "sample_prediction.json").read_text(encoding="utf-8"))
    log = _read_log()

    optuna = _section(log, "2. Baseline", "3. AutoML")
    trial_rows = [i for i, line in enumerate(optuna) if line.strip().startswith("trial")]
    if len(trial_rows) > 5:
        keep = optuna[:2] + [optuna[i] for i in trial_rows[:3]]
        keep += ["     ... 16 more trials ..."]
        keep += [optuna[i] for i in trial_rows[-1:]]
        keep += optuna[trial_rows[-1] + 1 :]
        optuna = keep

    tools_raw = _section(log, "3. AutoML", "   deployed pipeline:")
    table_rows = [
        line
        for line in tools_raw
        if line.startswith(
            ("tool ", "Optuna (baseline LR)", "PyCaret ", "FLAML ", "   AutoML vs baseline")
        )
    ]
    tools = [
        "3. AutoML model comparison (60s budget per engine)",
        next(line for line in tools_raw if line.startswith("   PyCaret candidates")),
        "   PyCaret CV F1: LR 0.5970 | GBC 0.5859 | RF 0.5615 | ET 0.5260 | DT 0.4966",
        next(line for line in tools_raw if line.startswith("   PyCaret winner")),
        next(line for line in tools_raw if line.startswith("   FLAML winner")),
        "   Tool comparison (same stratified holdout)",
    ] + table_rows
    leakage = _section(log, "4. Deliberate", "5. Leakage")

    raw_metrics = (ASSETS / "metrics_raw.txt").read_text(encoding="utf-8", errors="replace")
    metric_lines = [
        line
        for line in raw_metrics.splitlines()
        if line.startswith("# HELP http_requests_total")
        or line.startswith("# TYPE http_requests_total")
        or line.startswith("http_requests_total{")
        or line.startswith("http_request_duration_seconds_count{")
    ][:14]
    metric_lines = [
        "$ curl -s http://127.0.0.1:8000/metrics | grep -E 'http_requests|duration'"
    ] + metric_lines

    ci_lines = (ASSETS / "ci_local.txt").read_text(encoding="utf-8").splitlines()

    optuna_png = _terminal(
        optuna, "train.py  ·  Optuna baseline (20 TPE trials)", "terminal_optuna.png"
    )
    tools_png = _terminal(
        tools, "train.py  ·  PyCaret vs FLAML on the same holdout", "terminal_automl.png"
    )
    leakage_png = _terminal(
        leakage, "train.py  ·  deliberate leakage probe", "terminal_leakage.png"
    )
    metrics_png = _terminal(
        metric_lines, "GET /metrics  ·  Prometheus exposition", "terminal_metrics.png"
    )
    ci_png = _terminal(
        ci_lines, "ci job  ·  flake8 + pytest (local mirror)", "terminal_ci.png"
    )

    shap_summary = _img(REPORTS / "shap_summary_bar.png", 8.6)
    shap_force = _img(REPORTS / "shap_force_plot.png", 11.4)
    gui = _img(ASSETS / "gui_prediction.png", 6.35)

    story: list = []

    # ------------------------------------------------------------------
    # Page 1 - AutoML
    # ------------------------------------------------------------------
    story.append(Paragraph("ChurnGuard AI — AutoML, XAI &amp; DevOps", H1))
    story.append(
        Paragraph(
            "Three-page report · IBM Telco Customer Churn · FastAPI + React + SHAP · "
            "all numbers below are from <b>docs/training_run.log</b> (EXITCODE=0)",
            SUB,
        )
    )

    story.append(Paragraph("Problem &amp; dataset", H2))
    story.append(
        Paragraph(
            "Predict whether a telecom customer will churn (binary classification; churn is the "
            "positive class). The IBM Telco sample has 7,043 accounts and 20 columns. After "
            "dropping <font name='Courier'>customerID</font> and 11 rows with a blank "
            "<font name='Courier'>TotalCharges</font>, the cleaned data has <b>7,032 rows and 19 "
            "features</b> (3 numeric, 16 categorical) with a churn rate of <b>0.266</b>. The "
            "split is stratified 80/20: <b>5,625 train / 1,407 hold-out</b> rows "
            "(<font name='Courier'>random_state=42</font>). F1 is the primary metric because the "
            "classes are imbalanced and a missed churner costs more than a wasted retention "
            "offer; precision, recall and ROC-AUC are reported alongside it.",
            BODY,
        )
    )

    story.append(Paragraph("Step 2 · Optuna Bayesian optimisation (baseline)", H2))
    story.append(_img(optuna_png, 18.7))
    story.append(
        Paragraph(
            "Optuna's TPE sampler tuned regularisation <font name='Courier'>C</font>, "
            "<font name='Courier'>class_weight</font> and <font name='Courier'>fit_intercept</font> "
            "with 3-fold stratified CV on the training partition only. Best CV F1 "
            f"<b>{metrics['baseline']['cv_f1_mean']:.4f}</b>; hold-out F1 "
            f"<b>{metrics['baseline']['f1']:.4f}</b> (precision {metrics['baseline']['precision']:.4f}, "
            f"recall {metrics['baseline']['recall']:.4f}, ROC-AUC {metrics['baseline']['roc_auc']:.4f}). "
            "The untuned reference scored 0.5960 CV / 0.6099 hold-out.",
            SMALL,
        )
    )

    story.append(Paragraph("Step 3 · PyCaret vs FLAML under the same 60-second budget", H2))
    story.append(_img(tools_png, 18.7))
    story.append(
        Paragraph(
            "PyCaret's internal split is disabled (<font name='Courier'>train_size=0.999</font>) and "
            "<font name='Courier'>compare_models</font> is capped at 60 seconds; FLAML is capped at "
            "<font name='Courier'>time_budget=60</font> seconds. Both are scored on the exact same "
            "1,407-row hold-out, so the comparison is apples-to-apples. PyCaret selected logistic "
            "regression (0.6082); FLAML spent its budget on a small random forest (0.5844).",
            SMALL,
        )
    )

    story.append(Paragraph("Step 4 · Critical evaluation: deliberate leakage", H2))
    story.append(_img(leakage_png, 18.7))
    story.append(
        Paragraph(
            "A synthetic <font name='Courier'>future_churn_indicator</font> column was appended to "
            "copies of the frames as an exact copy of the label. The probe model reached "
            "<b>F1 = 1.0000</b>, which is the expected signature of leakage. The script prints a "
            "warning, drops the column, and retrains; the column never enters the real feature "
            "frame or the deployed pipeline. The same audit checks identifier removal, duplicate "
            "rows, fit/transform isolation, <font name='Courier'>TotalCharges ~= tenure x "
            "MonthlyCharges</font> redundancy (r = 1.000, known at prediction time, so redundancy "
            "rather than target leakage), class balance, and fairness slices.",
            SMALL,
        )
    )

    left = [
        shap_summary,
        Paragraph(
            "Global SHAP summary (mean |SHAP|, 200 hold-out rows). Top drivers: "
            "<b>MonthlyCharges 0.1790</b>, <b>tenure 0.1561</b>, <b>InternetService 0.1242</b>.",
            CAPTION,
        ),
    ]
    right = [
        Paragraph("GET /model-card", H3),
        Paragraph(
            "Dataset, metrics (baseline vs deployed), intended use, known limits, features, "
            "explainability, fairness slices, leakage checks/probe, AutoML runs and validation. "
            "Keys: <font name='Courier'>model, dataset, metrics, intended_use, known_limits, "
            "features, explainability, fairness, leakage_checks, leakage_probe, automl_tools, "
            "validation</font>.",
            SMALL,
        ),
        Paragraph("Honest result", H3),
        Paragraph(
            f"Tuned AutoML hold-out F1 <b>{metrics['automl']['f1']:.4f}</b> vs "
            f"<b>{metrics['baseline']['f1']:.4f}</b> for the Optuna baseline "
            f"(<b>{metrics['f1_improvement_over_baseline']:+.4f}</b>), well inside the CV standard "
            "deviation (0.0216). AutoML adds a selection process, tuning record and model card, "
            "not free accuracy; no improvement is claimed.",
            SMALL,
        ),
    ]
    row = Table(
        [[left, right]],
        colWidths=[9.65 * cm, 8.9 * cm],
    )
    row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 8),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ]
        )
    )
    story.append(row)
    story.append(
        _callout(
            "<b>Executable validation: 6/6 PASS</b> — Optuna completed · FLAML vs PyCaret "
            "comparison completed · leakage detected and dropped · SHAP plots generated · "
            "/metrics and /model-card present · K8s and Terraform files exist.",
        )
    )

    # ------------------------------------------------------------------
    # Page 2 - XAI
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Explainable AI — from global drivers to this customer", H1))
    story.append(
        Paragraph(
            "Every <font name='Courier'>POST /predict</font> returns the probability, the original "
            "normalised importance, signed <font name='Courier'>top_factors</font> and an English "
            "explanation. The React UI renders the signed values as a SHAP force plot.",
            SUB,
        )
    )

    sample_factors = sample.get("top_factors", [])[:3]
    factor_rows = [["Feature", "SHAP (log-odds)", "Direction"]]
    for factor in sample_factors:
        value = float(factor["contribution"])
        factor_rows.append(
            [
                factor["feature"],
                f"{value:+.4f}",
                "toward churn" if value >= 0 else "toward retention",
            ]
        )
    factor_table = Table(factor_rows, colWidths=[4.0 * cm, 3.5 * cm, 4.0 * cm])
    factor_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1ea")),
                ("TEXTCOLOR", (0, 0), (-1, 0), GREEN),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("LEADING", (0, 0), (-1, -1), 8.6),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d8cd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ]
        )
    )

    right_col = [
        shap_force,
        Paragraph(
            "SHAP force plot for the highest-risk hold-out customer "
            "(<font name='Courier'>shap.plots.force(..., matplotlib=True)</font>). Red pushes the "
            "prediction toward churn; blue pushes it toward retention.",
            CAPTION,
        ),
        Paragraph("How to read the GUI", H2),
        Paragraph(
            "The live React screenshot (left) shows <b>High Risk · 74.28%</b>, the normalised "
            "importance bars, and the signed force plot built from <font name='Courier'>"
            "top_factors</font>. Each row is a feature; the bar grows from the centreline toward "
            "retention (green, left) or churn (red, right), and the printed number is the "
            "log-odds contribution. This answers \"why this customer?\" rather than only \"which "
            "features matter globally?\".",
            SMALL,
        ),
        Paragraph("Innovation · kNN marginalisation", H2),
        Paragraph(
            "A retention agent rarely has all 19 Telco fields. Instead of filling missing inputs "
            "with one made-up \"typical\" row or per-column medians, the API averages the model over "
            "the <b>100 training customers closest to the supplied inputs in tenure and monthly "
            "charges</b> (distance in standard deviations). Supplied values are applied to every "
            "one of those rows. This estimates <font name='Courier'>P(churn | supplied)</font> by "
            "marginalising over the missing dimensions, so the score and its SHAP values stay "
            "on-manifold. The response names the inferred fields in "
            "<font name='Courier'>assumed_features</font> and reports "
            "<font name='Courier'>aggregation = {method: neighbour_average, rows: 100}</font>; a "
            "single-row fast path is used when every input is supplied. SHAP values are averaged "
            "over the same look-alike rows, so the force plot explains the number actually "
            "returned.",
            SMALL,
        ),
    ]

    gui_col = [
        gui,
        Paragraph(
            "Live React UI (headless Chrome, real API): High Risk, 74.28%, importance bars and "
            "the signed force plot. The form is above; the optional-profile section and model "
            "meta are below the crop.",
            CAPTION,
        ),
    ]
    xai_row = Table([[gui_col, right_col]], colWidths=[6.85 * cm, 11.7 * cm])
    xai_row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 10),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ]
        )
    )
    story.append(xai_row)

    story.append(Paragraph("The same explanation from the API", H2))
    story.append(
        Paragraph(
            f"Request: tenure 2, MonthlyCharges 95, month-to-month, 4 support calls, fibre optic, "
            f"electronic check, paperless billing, no tech support. Response: "
            f"<b>{sample['prediction']} · {sample['churn_probability'] * 100:.2f}%</b>. "
            f"<i>{sample['explanation']}</i>",
            SMALL,
        )
    )
    story.append(factor_table)
    story.append(
        Paragraph(
            "The kNN path inferred 11 fields and reported "
            f"<font name='Courier'>aggregation.rows = {sample['aggregation']['rows']}</font>. "
            "SHAP values describe model behaviour, not causal effects: they say which inputs the "
            "model relied on, not what would happen if a customer changed a feature.",
            CAPTION,
        )
    )

    # ------------------------------------------------------------------
    # Page 3 - DevOps
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("DevOps — CI/CD, containers, orchestration &amp; observability", H1))
    story.append(
        Paragraph(
            "The same repository ships a blocking lint/test gate, a multi-stage image, Kubernetes "
            "manifests with two replicas and Terraform for Azure. Monitoring is built in, not "
            "bolted on.",
            SUB,
        )
    )

    story.append(Paragraph("CI gate · lint and tests before the image is built", H2))
    story.append(
        Paragraph(
            "<font name='Courier'>.github/workflows/deploy.yml</font> runs a blocking "
            "<font name='Courier'>ci</font> job on every push: install "
            "<font name='Courier'>requirements-dev.txt</font>, run <b>flake8</b> (style errors fail "
            "the build), then <b>pytest</b> (35 tests; the suite is hermetic). The "
            "<font name='Courier'>deploy</font> job has <font name='Courier'>needs: ci</font>, so it "
            "cannot start unless both pass.",
            SMALL,
        )
    )
    actions_png = ASSETS / "github_actions.png"
    if actions_png.exists():
        story.append(_img(actions_png, 14.2))
        story.append(
            Paragraph(
                "Public GitHub Actions job for rubric commit 2cd3174: Lint and test succeeded in "
                "35 s (flake8 + 35 pytest tests). The separate Build and deploy job fails at "
                "azure/login because the service principal has no subscription association - "
                "an external RBAC issue, not a code failure.",
                CAPTION,
            )
        )
    else:
        story.append(_img(ci_png, 18.7))
        story.append(
            Paragraph(
                "Local mirror of the CI job (the public Actions screenshot is attached separately "
                "when the run completes).",
                CAPTION,
            )
        )

    dockerfile_text = """FROM python:3.10-slim AS builder
RUN apt-get update && apt-get install \\
    -y --no-install-recommends build-essential
RUN python -m venv /opt/venv \\
 && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.10-slim AS runtime
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app . .
RUN useradd --uid 1001 --gid 1001 app
USER app
HEALTHCHECK CMD python -c "..."
CMD ["uvicorn", "main:app", "--port", "8000"]"""

    k8s_text = """# k8s/deployment.yaml
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
      containers:
        - name: api
          readinessProbe:  /model-info
          livenessProbe:   /"""

    tf_text = """# terraform/main.tf
resource "azurerm_resource_group" "main" {
  name     = "${var.project}-rg"
  location = var.location
}

resource "azurerm_container_registry" "main" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
}"""

    code_row = Table(
        [
            [
                [
                    Paragraph("Multi-stage Dockerfile", H3),
                    _code_box(dockerfile_text, small=True),
                ],
                [
                    Paragraph("Kubernetes (2 replicas)", H3),
                    _code_box(k8s_text, small=True),
                ],
                [
                    Paragraph("Terraform (Azure RG + ACR)", H3),
                    _code_box(tf_text, small=True),
                ],
            ]
        ],
        colWidths=[6.15 * cm, 6.15 * cm, 6.15 * cm],
    )
    code_row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(code_row)

    story.append(Paragraph("Observability · Prometheus at /metrics", H2))
    story.append(
        Paragraph(
            "<font name='Courier'>prometheus-fastapi-instrumentator</font> is wired in "
            "<font name='Courier'>main.py</font>. <font name='Courier'>GET /metrics</font> exposes "
            "request counts, status codes, latency histograms and in-progress requests; the API "
            "also keeps structured request logs and a drift endpoint. The excerpt below is a real "
            "capture from the running service.",
            SMALL,
        )
    )
    story.append(_img(metrics_png, 18.7))

    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.15 * cm,
        rightMargin=1.15 * cm,
        topMargin=0.95 * cm,
        bottomMargin=0.95 * cm,
        title="ChurnGuard AI - AutoML, XAI and DevOps report",
        author="ChurnGuard AI",
        subject="Three-page rubric report",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_footer)])
    doc.build(story)
    return PDF_PATH


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")
