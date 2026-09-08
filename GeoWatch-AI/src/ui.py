from __future__ import annotations

from datetime import datetime
from html import escape


GLOBAL_CSS = """
<style>
:root {
  --gw-bg: #050b0f;
  --gw-sidebar: #081116;
  --gw-panel: #0b151b;
  --gw-panel-2: #0f1c23;
  --gw-panel-hover: #13242c;
  --gw-line: rgba(148, 180, 192, .14);
  --gw-line-strong: rgba(148, 180, 192, .25);
  --gw-text: #f4f8f9;
  --gw-muted: #8fa3ab;
  --gw-dim: #61747c;
  --gw-cyan: #45e6cf;
  --gw-cyan-soft: rgba(69, 230, 207, .10);
  --gw-amber: #f6bd60;
  --gw-red: #ff7272;
  --gw-blue: #69a9ff;
  --gw-radius: 14px;
}

html, body, [class*="css"] {
  font-family: Inter, "Segoe UI Variable", "Segoe UI", sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 78% -10%, rgba(43, 128, 121, .11), transparent 34rem),
    var(--gw-bg);
  color: var(--gw-text);
}

header[data-testid="stHeader"] { background: transparent; height: 0; }
footer { display: none; }
.block-container { max-width: 1560px; padding: 1.55rem 2rem 4rem; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #091319 0%, #071015 100%);
  border-right: 1px solid var(--gw-line);
}
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1rem 2rem; }
[data-testid="stSidebar"] h2 {
  color: var(--gw-dim) !important;
  font-size: .68rem !important;
  font-weight: 650 !important;
  letter-spacing: .14em !important;
  text-transform: uppercase;
  margin: 1.55rem .65rem .45rem;
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: .18rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  min-height: 42px;
  padding: .62rem .72rem;
  border: 1px solid transparent;
  border-radius: 9px;
  color: var(--gw-muted);
  font-size: .88rem;
  transition: background .16s ease, border-color .16s ease, color .16s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,.025);
  color: var(--gw-text);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: var(--gw-cyan-soft);
  border-color: rgba(69, 230, 207, .18);
  color: #dffcf7;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--gw-muted); }

.gw-side-brand {
  display: flex;
  align-items: center;
  gap: .72rem;
  padding: .25rem .55rem 1.05rem;
  border-bottom: 1px solid var(--gw-line);
}
.gw-side-brand__mark {
  position: relative;
  width: 31px;
  height: 31px;
  border: 1px solid rgba(69,230,207,.48);
  border-radius: 9px;
  background: linear-gradient(145deg, rgba(69,230,207,.16), rgba(69,230,207,.02));
}
.gw-side-brand__mark:before,
.gw-side-brand__mark:after {
  content: "";
  position: absolute;
  inset: 8px;
  border: 1px solid var(--gw-cyan);
  border-radius: 50%;
}
.gw-side-brand__mark:after { inset: 13px; background: var(--gw-cyan); }
.gw-side-brand strong { display: block; letter-spacing: .045em; font-size: .9rem; }
.gw-side-brand small { display: block; margin-top: .1rem; color: var(--gw-dim); font-size: .65rem; letter-spacing: .09em; text-transform: uppercase; }

.gw-system {
  display: flex;
  align-items: flex-start;
  gap: .6rem;
  padding: .72rem;
  background: rgba(255,255,255,.022);
  border: 1px solid var(--gw-line);
  border-radius: 10px;
}
.gw-system__dot { width: 7px; height: 7px; margin-top: .32rem; border-radius: 50%; background: var(--gw-cyan); box-shadow: 0 0 0 4px rgba(69,230,207,.08); }
.gw-system strong { display: block; font-size: .75rem; color: #dffcf7; }
.gw-system span { display: block; margin-top: .16rem; font-size: .68rem; line-height: 1.35; color: var(--gw-dim); }

.gw-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 1.25rem;
  margin-bottom: 1.45rem;
  border-bottom: 1px solid var(--gw-line);
}
.gw-breadcrumb { color: var(--gw-dim); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
.gw-title { margin-top: .34rem; color: var(--gw-text); font-size: 1.36rem; line-height: 1.2; font-weight: 660; letter-spacing: -.025em; }
.gw-topbar__right { display: flex; align-items: center; gap: .55rem; }
.gw-chip {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  min-height: 31px;
  padding: 0 .7rem;
  border: 1px solid var(--gw-line);
  border-radius: 999px;
  color: var(--gw-muted);
  font-size: .7rem;
  letter-spacing: .04em;
  white-space: nowrap;
}
.gw-chip--live { color: #a9f4e9; border-color: rgba(69,230,207,.24); background: rgba(69,230,207,.055); }
.gw-chip__dot { width: 6px; height: 6px; border-radius: 50%; background: var(--gw-cyan); box-shadow: 0 0 8px rgba(69,230,207,.65); }

h1, h2, h3 { color: var(--gw-text) !important; letter-spacing: -.025em; }
h1 { font-size: 1.75rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.02rem !important; }
p, li { line-height: 1.55; }

.gw-section-label {
  margin: 1.2rem 0 .7rem;
  color: var(--gw-dim);
  font-size: .68rem;
  font-weight: 650;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.gw-grid { display: grid; gap: .85rem; }
.gw-grid--4 { grid-template-columns: repeat(4, minmax(0,1fr)); }
.gw-kpi {
  position: relative;
  min-height: 124px;
  padding: 1rem 1.05rem;
  overflow: hidden;
  background: linear-gradient(150deg, rgba(15,29,36,.96), rgba(9,19,25,.96));
  border: 1px solid var(--gw-line);
  border-radius: var(--gw-radius);
}
.gw-kpi:after {
  content: "";
  position: absolute;
  width: 74px;
  height: 74px;
  right: -26px;
  top: -30px;
  border: 1px solid rgba(69,230,207,.10);
  border-radius: 50%;
}
.gw-kpi__label { color: var(--gw-muted); font-size: .73rem; }
.gw-kpi__value { margin-top: .72rem; color: var(--gw-text); font-size: 1.72rem; font-weight: 670; letter-spacing: -.04em; }
.gw-kpi__meta { margin-top: .28rem; color: var(--gw-dim); font-size: .68rem; }
.gw-kpi__signal { color: var(--gw-cyan); }
.gw-kpi__warn { color: var(--gw-amber); }

.gw-panel {
  background: linear-gradient(160deg, rgba(13,27,34,.97), rgba(8,18,24,.97));
  border: 1px solid var(--gw-line);
  border-radius: var(--gw-radius);
  overflow: hidden;
}
.gw-panel__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 54px;
  padding: 0 1rem;
  border-bottom: 1px solid var(--gw-line);
}
.gw-panel__title { color: var(--gw-text); font-size: .82rem; font-weight: 620; }
.gw-panel__meta { color: var(--gw-dim); font-size: .68rem; }
.gw-panel__body { padding: 1rem; }
.gw-map-empty {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 285px;
  padding: 1.5rem;
  text-align: center;
  background-color: #07151a;
  background-image:
    linear-gradient(rgba(69,230,207,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(69,230,207,.04) 1px, transparent 1px);
  background-size: 34px 34px;
}
.gw-map-empty:after {
  content: "";
  position: absolute;
  width: 150px;
  height: 150px;
  border: 1px solid rgba(69,230,207,.16);
  border-radius: 50%;
  box-shadow: 0 0 0 38px rgba(69,230,207,.025), 0 0 0 76px rgba(69,230,207,.014);
}
.gw-map-empty__copy { position: relative; z-index: 2; max-width: 300px; }
.gw-map-empty__icon { width: 10px; height: 10px; margin: 0 auto .75rem; border-radius: 50%; background: var(--gw-cyan); box-shadow: 0 0 0 7px rgba(69,230,207,.10); }
.gw-map-empty strong { display: block; font-size: .8rem; }
.gw-map-empty span { display: block; margin-top: .35rem; color: var(--gw-muted); font-size: .72rem; line-height: 1.45; }

.gw-timeline { position: relative; padding-left: 1.35rem; }
.gw-timeline:before { content:""; position:absolute; left:.28rem; top:.45rem; bottom:.45rem; width:1px; background:var(--gw-line-strong); }
.gw-event { position: relative; padding: 0 0 1.02rem .3rem; }
.gw-event:last-child { padding-bottom: 0; }
.gw-event:before { content:""; position:absolute; left:-1.11rem; top:.34rem; width:7px; height:7px; border:2px solid #0d1b22; border-radius:50%; background:var(--gw-cyan); box-shadow:0 0 0 1px rgba(69,230,207,.3); }
.gw-event__top { display:flex; justify-content:space-between; gap:.75rem; }
.gw-event__title { color:#dfeaec; font-size:.75rem; font-weight:600; }
.gw-event__time { color:var(--gw-dim); font-size:.64rem; white-space:nowrap; }
.gw-event__detail { margin-top:.2rem; color:var(--gw-muted); font-size:.68rem; }

.gw-readiness { display:grid; gap:.48rem; }
.gw-readiness__item { display:grid; grid-template-columns:8px 1fr auto; align-items:center; gap:.65rem; padding:.68rem .72rem; border:1px solid var(--gw-line); border-radius:9px; background:rgba(255,255,255,.015); }
.gw-readiness__bar { width:3px; height:26px; border-radius:4px; background:var(--gw-cyan); }
.gw-readiness__bar--wait { background:var(--gw-amber); }
.gw-readiness__copy strong { display:block; color:#dfeaec; font-size:.72rem; }
.gw-readiness__copy span { display:block; margin-top:.16rem; color:var(--gw-dim); font-size:.64rem; }
.gw-readiness__state { color:var(--gw-muted); font-size:.64rem; text-transform:uppercase; letter-spacing:.07em; }

[data-testid="stMetric"] { background: var(--gw-panel); border: 1px solid var(--gw-line); border-radius: var(--gw-radius); padding: 1rem; }
[data-testid="stMetricLabel"] { color: var(--gw-muted); font-size: .72rem; }
[data-testid="stMetricValue"] { color: var(--gw-text); font-weight: 650; }
[data-testid="stFileUploaderDropzone"] { padding: 1.35rem; background: #091820; border: 1px dashed rgba(69,230,207,.28); border-radius: var(--gw-radius); }
[data-testid="stDataFrame"] { overflow: hidden; border: 1px solid var(--gw-line); border-radius: var(--gw-radius); }
[data-testid="stExpander"] { background: var(--gw-panel); border: 1px solid var(--gw-line) !important; border-radius: var(--gw-radius) !important; }
.stButton > button, .stDownloadButton > button { min-height: 42px; border: 1px solid var(--gw-line-strong); border-radius: 9px; font-weight: 610; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: rgba(69,230,207,.42); color: var(--gw-cyan); }
.stButton > button[kind="primary"] { color:#04110f; background:var(--gw-cyan); border-color:var(--gw-cyan); }
[data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--gw-line); }
[data-baseweb="tab"] { border-radius:8px 8px 0 0; }
.stAlert { border-radius:10px; }
.warn { padding:.8rem 1rem; border:1px solid rgba(69,230,207,.18); border-radius:10px; background:rgba(69,230,207,.055); color:#c9f7f0; }
.empty { padding:1.2rem; border:1px dashed var(--gw-line-strong); border-radius:var(--gw-radius); background:rgba(255,255,255,.015); color:var(--gw-muted); }
.stage { padding:.9rem 1rem; border:1px solid var(--gw-line); border-radius:10px; background:var(--gw-panel); margin:.45rem 0; }
.stage-ready { border-left:3px solid var(--gw-cyan); }
.stage-wait { border-left:3px solid var(--gw-amber); }
.change-card { background:var(--gw-panel); border:1px solid var(--gw-line); border-radius:var(--gw-radius); padding:1rem; margin:.5rem 0; }
.change-card span { color:var(--gw-muted); font-size:.82rem; }

.gw-workbench {
  display:grid;
  grid-template-columns:minmax(0,1.15fr) minmax(0,1fr) minmax(0,1fr);
  gap:.65rem;
  margin:.2rem 0 1rem;
}
.gw-workbench__step {
  position:relative;
  padding:.72rem .85rem .72rem 2rem;
  border:1px solid var(--gw-line);
  border-radius:9px;
  background:rgba(255,255,255,.015);
  color:var(--gw-muted);
  font-size:.7rem;
}
.gw-workbench__step:before {
  content:"";
  position:absolute;
  left:.8rem;
  top:50%;
  width:7px;
  height:7px;
  transform:translateY(-50%);
  border-radius:50%;
  background:var(--gw-cyan);
  box-shadow:0 0 0 4px rgba(69,230,207,.08);
}
.gw-workbench__step strong { display:block; color:#dfeaec; font-size:.73rem; margin-bottom:.12rem; }
.gw-evidence {
  padding:1rem;
  border:1px solid rgba(69,230,207,.2);
  border-radius:var(--gw-radius);
  background:linear-gradient(160deg, rgba(16,35,42,.98), rgba(8,18,24,.98));
}
.gw-evidence__eyebrow { color:var(--gw-cyan); font-size:.63rem; letter-spacing:.12em; text-transform:uppercase; }
.gw-evidence__title { margin-top:.55rem; color:var(--gw-text); font-size:1rem; font-weight:650; }
.gw-evidence__grid { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; margin-top:.9rem; }
.gw-evidence__datum { padding:.6rem; border:1px solid var(--gw-line); border-radius:8px; background:rgba(255,255,255,.018); }
.gw-evidence__datum span { display:block; color:var(--gw-dim); font-size:.61rem; text-transform:uppercase; letter-spacing:.07em; }
.gw-evidence__datum strong { display:block; margin-top:.2rem; color:#dfeaec; font-size:.73rem; overflow-wrap:anywhere; }
.gw-evidence__review { margin-top:.8rem; padding:.7rem; border-left:2px solid var(--gw-amber); background:rgba(246,189,96,.055); color:#e9d6ad; font-size:.68rem; line-height:1.45; }
.gw-layer-key { display:flex; flex-wrap:wrap; gap:.45rem; margin:.25rem 0 .75rem; }
.gw-layer-key span { display:inline-flex; align-items:center; gap:.35rem; padding:.3rem .55rem; border:1px solid var(--gw-line); border-radius:999px; color:var(--gw-muted); font-size:.65rem; }
.gw-layer-key i { width:7px; height:7px; border-radius:2px; background:var(--gw-cyan); }
.gw-layer-key .old { background:var(--gw-amber); }
.gw-layer-key .stable { background:var(--gw-blue); }

@media (max-width: 900px) {
  .block-container { padding: 1.2rem 1rem 3rem; }
  .gw-grid--4 { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .gw-topbar { align-items:flex-start; }
  .gw-topbar__right .gw-chip:not(.gw-chip--live) { display:none; }
  .gw-workbench { grid-template-columns:1fr; }
}
@media (max-width: 560px) {
  .gw-grid--4 { grid-template-columns: 1fr; }
  .gw-topbar { display:block; }
  .gw-topbar__right { margin-top:.75rem; }
}
/* Light workspace palette. Native widgets follow .streamlit/config.toml. */
:root {
  --gw-bg: #f7f6f2;
  --gw-sidebar: #ffffff;
  --gw-panel: #ffffff;
  --gw-panel-2: #f1f2f4;
  --gw-panel-hover: #eeece5;
  --gw-text: #0e1c36;
  --gw-line: #deded8;
  --gw-line-strong: #b9bcbf;
  --gw-muted: #58574b;
  --gw-dim: #626775;
  --gw-cyan: #29466e;
  --gw-cyan-soft: #edf1f7;
  --gw-amber: #896019;
  --gw-red: #b83442;
  --gw-blue: #315ba0;
  --gw-radius: 10px;
}
.stApp { background: var(--gw-bg); }
[data-testid="stSidebar"] { background: var(--gw-sidebar); min-width: 250px; }
[data-testid="stSidebar"] h2 { margin: 1rem .4rem .25rem; letter-spacing: .06em !important; }
[data-testid="stSidebar"] [data-testid="stButton"] button {
  justify-content: flex-start; text-align: left; min-height: 34px;
  padding: .35rem .65rem; border-radius: 7px;
}
[data-testid="stSidebar"] button[kind="primary"] {
  background: #f5ecd7; border-color: #ddaf53; color: #0e1c36;
}
[data-testid="stSidebar"] button:focus-visible { outline: 2px solid #29466e; outline-offset: 2px; }
.gw-kpi, .gw-panel, .gw-evidence { background: #fff; box-shadow: 0 2px 7px rgba(14,28,54,.025); }
.gw-kpi:after { display: none; }
.gw-kpi { border-top: 3px solid #ddaf53; }
.gw-side-brand__mark { background: #f5ecd7; border-color: #ddaf53; }
.gw-system strong, .gw-event__title, .gw-readiness__copy strong,
.gw-workbench__step strong, .gw-evidence__datum strong { color: var(--gw-text); }
.gw-map-empty { background-color: #f0f2f4; background-image: linear-gradient(#e1e5e9 1px, transparent 1px), linear-gradient(90deg, #e1e5e9 1px, transparent 1px); }
.gw-map-empty:after { border-color: #d2d8df; box-shadow: none; }
.gw-evidence__review, .warn { background: #faf3e4; color: #654915; border-color: #ddaf53; }
.gw-evidence { border-color: var(--gw-line); }
.gw-event:before { border-color: #fff; box-shadow: none; }
[data-testid="stFileUploaderDropzone"] { background: #fff; border-color: #a6afbd; }
.stButton > button[kind="primary"] { color: #fff; background: #0e1c36; border-color: #0e1c36; }
.gw-chip { background: #fff; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: #f5ecd7; border-color: #ddaf53; color: #0e1c36;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p { color: #0e1c36; }
.gw-title { font-size: 1.7rem; letter-spacing: -.035em; }
.gw-topbar { padding-bottom: 1rem; margin-bottom: 1.25rem; }
header[data-testid="stHeader"] { height: auto; }
@media (max-width: 700px) {
  [data-testid="stSidebar"] { min-width: 0; }
  .block-container { padding-top: 3.5rem; }
}
</style>
"""


def topbar(section: str, title: str, model_version: str) -> str:
    return f"""
    <div class="gw-topbar">
      <div>
        <div class="gw-breadcrumb">{escape(section)}</div>
        <div class="gw-title">{escape(title)}</div>
      </div>
      <div class="gw-topbar__right">
        <span class="gw-chip">Проверка аналитиком</span>
      </div>
    </div>
    """


def kpi_grid(items: list[tuple[str, str, str, str]]) -> str:
    cards = []
    for label, value, meta, tone in items:
        tone_class = "gw-kpi__warn" if tone == "warn" else "gw-kpi__signal" if tone == "signal" else ""
        cards.append(
            '<div class="gw-kpi">'
            f'<div class="gw-kpi__label">{escape(label)}</div>'
            f'<div class="gw-kpi__value {tone_class}">{escape(value)}</div>'
            f'<div class="gw-kpi__meta">{escape(meta)}</div>'
            '</div>'
        )
    return '<div class="gw-grid gw-grid--4">' + ''.join(cards) + '</div>'


def timeline(events: list[tuple[str, str, str]], empty_text: str) -> str:
    if not events:
        return f'<div class="gw-map-empty"><div class="gw-map-empty__copy"><div class="gw-map-empty__icon"></div><strong>Журнал событий пуст</strong><span>{escape(empty_text)}</span></div></div>'
    rows = []
    for title, detail, timestamp in events:
        rows.append(
            '<div class="gw-event">'
            f'<div class="gw-event__top"><span class="gw-event__title">{escape(title)}</span><span class="gw-event__time">{escape(timestamp)}</span></div>'
            f'<div class="gw-event__detail">{escape(detail)}</div>'
            '</div>'
        )
    return '<div class="gw-panel__body"><div class="gw-timeline">' + ''.join(rows) + '</div></div>'


def map_empty() -> str:
    return """
    <div class="gw-map-empty">
      <div class="gw-map-empty__copy">
        <div class="gw-map-empty__icon"></div>
        <strong>Нет снимков с координатами</strong>
        <span>Загрузите GeoTIFF в WGS 84 — сцена появится на оперативной карте без вымышленных координат.</span>
      </div>
    </div>
    """


def readiness(items: list[tuple[str, str, bool]]) -> str:
    rows = []
    for title, detail, ready in items:
        bar = "" if ready else " gw-readiness__bar--wait"
        state = "готово" if ready else "требует внимания"
        rows.append(
            '<div class="gw-readiness__item">'
            f'<i class="gw-readiness__bar{bar}"></i>'
            f'<div class="gw-readiness__copy"><strong>{escape(title)}</strong><span>{escape(detail)}</span></div>'
            f'<span class="gw-readiness__state">{state}</span>'
            '</div>'
        )
    return '<div class="gw-panel__body"><div class="gw-readiness">' + ''.join(rows) + '</div></div>'


def utc_short(value: datetime) -> str:
    return value.strftime("%d %b · %H:%M UTC")


def temporal_workbench() -> str:
    return """
    <div class="gw-workbench">
      <div class="gw-workbench__step"><strong>01 · Источник</strong>Одна территория, два периода</div>
      <div class="gw-workbench__step"><strong>02 · Сопоставление</strong>Качество, геометрия, кандидаты</div>
      <div class="gw-workbench__step"><strong>03 · Решение</strong>Подтверждает аналитик</div>
    </div>
    """


def layer_key() -> str:
    return """
    <div class="gw-layer-key">
      <span><i></i>Появилось</span>
      <span><i class="old"></i>Исчезло</span>
      <span><i class="stable"></i>Сохранилось</span>
    </div>
    """


def evidence_card(
    *,
    event_label: str,
    class_name: str,
    confidence: str,
    overlap: str,
    before_id: str,
    after_id: str,
) -> str:
    return (
        '<div class="gw-evidence">'
        '<div class="gw-evidence__eyebrow">Evidence candidate</div>'
        f'<div class="gw-evidence__title">{escape(event_label)} · {escape(class_name)}</div>'
        '<div class="gw-evidence__grid">'
        f'<div class="gw-evidence__datum"><span>Confidence</span><strong>{escape(confidence)}</strong></div>'
        f'<div class="gw-evidence__datum"><span>IoU пары</span><strong>{escape(overlap)}</strong></div>'
        f'<div class="gw-evidence__datum"><span>До</span><strong>{escape(before_id)}</strong></div>'
        f'<div class="gw-evidence__datum"><span>После</span><strong>{escape(after_id)}</strong></div>'
        '</div>'
        '<div class="gw-evidence__review">Это кандидат модели, а не установленный факт. '
        'Подтвердите или отклоните его в очереди экспертной проверки.</div>'
        '</div>'
    )
