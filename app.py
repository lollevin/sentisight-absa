import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from absa_engine import PROVIDER_MODELS, PROVIDER_INFO, analyze_batch, analyze_review
from evaluator import evaluate_results, load_labeled_data, vader_baseline_metrics
from lang import t, tlist
from traditional import analyze_batch_vader, analyze_vader
from visualizer import (
    aspect_bar_chart,
    aspect_grouped_bar,
    generate_wordcloud,
    sentiment_pie,
    vader_gauge,
)

load_dotenv()

def _get_default_api_key(provider: str = "DeepSeek") -> str:
    """Read API key from .env (local) or st.secrets (Streamlit Cloud)."""
    env_map = {
        "DeepSeek": "DEEPSEEK_API_KEY",
        "OpenAI":   "OPENAI_API_KEY",
        "Groq":     "GROQ_API_KEY",
        "Gemini":   "GEMINI_API_KEY",
    }
    env_var = env_map.get(provider, "DEEPSEEK_API_KEY")
    # 1. Try environment variable (local .env)
    key = os.getenv(env_var, "")
    if not key:  # fallback: any known key
        for v in env_map.values():
            key = os.getenv(v, "")
            if key:
                break
    # 2. Try Streamlit secrets (cloud deployment)
    if not key:
        try:
            key = st.secrets.get(env_var, "")
            if not key:
                for v in env_map.values():
                    key = st.secrets.get(v, "")
                    if key:
                        break
        except Exception:
            pass
    return key or ""

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiSight — AI Marketing Insight Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Language init ─────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; }

.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 70%, #6d28d9 100%);
    border-radius: 16px; padding: 2.2rem 2.5rem 1.6rem;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(139,92,246,0.3);
    box-shadow: 0 8px 40px rgba(109,40,217,0.25);
}
.hero-title   { font-size:2.4rem; font-weight:700; color:#f8fafc; margin:0; letter-spacing:-0.5px; }
.hero-sub-title { font-size:1.3rem; font-weight:600; color:#c4b5fd; margin-top:0.15rem; }
.hero-desc    { font-size:0.95rem; color:#a78bfa; margin-top:0.4rem; }

/* Lang toggle buttons */
.lang-btn-row { display:flex; gap:8px; justify-content:flex-end; margin-bottom:0.5rem; }
.lang-btn {
    background: rgba(255,255,255,0.08); border:1px solid rgba(139,92,246,0.4);
    border-radius:8px; padding:4px 14px; font-size:0.82rem; font-weight:600;
    color:#c4b5fd; cursor:pointer; transition:all .15s;
}
.lang-btn.active { background:rgba(109,40,217,0.6); border-color:#7c3aed; color:#f8fafc; }

.kpi-card {
    background: linear-gradient(135deg,rgba(30,27,75,.8),rgba(49,46,129,.5));
    border:1px solid rgba(139,92,246,.2); border-radius:12px;
    padding:1.2rem 1.5rem; text-align:center;
}
.kpi-value { font-size:2rem; font-weight:700; color:#a78bfa; }
.kpi-label { font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; }

.aspect-card  { border-radius:10px; padding:1rem 1.2rem; margin-bottom:.7rem; border-left:4px solid; }
.aspect-positive { background:rgba(34,197,94,.08);  border-color:#22c55e; }
.aspect-negative { background:rgba(239,68,68,.08);  border-color:#ef4444; }
.aspect-neutral  { background:rgba(148,163,184,.08); border-color:#94a3b8; }
.aspect-name  { font-size:1rem; font-weight:600; color:#e2e8f0; }
.aspect-quote { font-size:.85rem; color:#94a3b8; font-style:italic; margin-top:.3rem; }
.aspect-badge-pos { background:#166534; color:#86efac; border-radius:6px; padding:2px 10px; font-size:.78rem; font-weight:600; }
.aspect-badge-neg { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:2px 10px; font-size:.78rem; font-weight:600; }
.aspect-badge-neu { background:#1e293b; color:#94a3b8; border-radius:6px; padding:2px 10px; font-size:.78rem; font-weight:600; }
.conf-bar  { height:6px; border-radius:3px; background:#1e293b; margin-top:6px; }
.conf-fill { height:6px; border-radius:3px; }

.sug-card {
    background:rgba(30,27,75,.5); border:1px solid rgba(139,92,246,.2);
    border-radius:10px; padding:.9rem 1.2rem; margin-bottom:.5rem;
    color:#e2e8f0; font-size:.92rem;
}
.section-header {
    font-size:1rem; font-weight:600; color:#a78bfa;
    text-transform:uppercase; letter-spacing:.1em;
    border-bottom:1px solid rgba(139,92,246,.2);
    padding-bottom:.4rem; margin-bottom:1rem;
}
.vader-card { background:rgba(15,23,42,.6); border:1px solid rgba(100,116,139,.2); border-radius:12px; padding:1.2rem; text-align:center; }
.vader-label { font-size:.78rem; color:#64748b; text-transform:uppercase; letter-spacing:.1em; }
.vader-value { font-size:1.6rem; font-weight:700; color:#64748b; margin:.3rem 0; }
.vader-limit { font-size:.78rem; color:#475569; margin-top:.5rem; font-style:italic; }
.callout      { background:rgba(34,197,94,.08); border:1px solid rgba(34,197,94,.3); border-radius:10px; padding:1rem 1.5rem; color:#86efac; font-size:.92rem; }
.callout-warn { background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.3); border-radius:10px; padding:1rem 1.5rem; color:#fca5a5; font-size:.92rem; }

#MainMenu { visibility:hidden; } footer { visibility:hidden; } header { visibility:hidden; }

.stTabs [data-baseweb="tab-list"] { background:rgba(30,27,75,.4); border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"]      { border-radius:8px; color:#94a3b8; padding:.5rem 1.2rem; }
.stTabs [aria-selected="true"]    { background:rgba(109,40,217,.5); color:#e2e8f0; }

[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid rgba(139,92,246,.15); }

button[kind="primary"] {
    background:linear-gradient(135deg,#6d28d9,#7c3aed);
    border:none; border-radius:8px; font-weight:600; letter-spacing:.03em;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {t('sidebar_title')}")
    st.markdown("---")
    st.markdown(f"**{t('sidebar_api_config')}**")

    provider = st.selectbox(t("sidebar_provider"), list(PROVIDER_MODELS.keys()), index=0)
    info     = PROVIDER_INFO[provider]
    st.caption(f"ℹ️ {info['note']}")
    model    = st.selectbox(t("sidebar_model"), PROVIDER_MODELS[provider])

    # ── Smart API key: show status badge if key is pre-loaded, else show input ──
    default_key = _get_default_api_key(provider)
    if default_key:
        api_key = default_key
        st.markdown(
            '<div style="background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);'
            'border-radius:8px;padding:.5rem .8rem;display:flex;align-items:center;gap:.5rem;">'
            '<span style="color:#22c55e;font-size:1rem;">✅</span>'
            '<span style="color:#86efac;font-size:.85rem;font-weight:600;">API Key Ready</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Allow override via expander (advanced users)
        with st.expander("🔧 Use a different key", expanded=False):
            override = st.text_input("Override API Key", type="password", placeholder="sk-...")
            if override:
                api_key = override
    else:
        api_key = st.text_input(
            t("sidebar_api_key"),
            type="password",
            placeholder=t("sidebar_api_placeholder"),
        )
        st.markdown(
            f'<a href="{info["key_url"]}" target="_blank" style="font-size:.78rem;color:#a78bfa;">'
            f'🔑 Get {provider} API Key →</a>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(t("sidebar_how_to_get"))
    st.markdown(t("sidebar_deepseek_steps"))
    st.markdown(t("sidebar_openai_steps"))
    st.markdown("---")
    st.markdown(t("sidebar_about"))
    st.caption(t("sidebar_about_text"))

# ─── Helpers ───────────────────────────────────────────────────────────────────
def sentiment_badge(sentiment: str) -> str:
    cls  = {"positive": "aspect-badge-pos", "negative": "aspect-badge-neg"}.get(sentiment, "aspect-badge-neu")
    icon = {"positive": "✅", "negative": "❌"}.get(sentiment, "➖")
    return f'<span class="{cls}">{icon} {sentiment.capitalize()}</span>'

def conf_bar_html(conf: float, sentiment: str) -> str:
    color = {"positive": "#22c55e", "negative": "#ef4444"}.get(sentiment, "#94a3b8")
    return (f'<div class="conf-bar"><div class="conf-fill" '
            f'style="width:{conf*100:.0f}%;background:{color};"></div></div>')

def render_aspect_card(aspect: dict):
    name      = aspect["name"].replace("_", " ").title()
    sentiment = aspect.get("sentiment", "neutral")
    quote     = aspect.get("quote", "")
    conf      = aspect.get("confidence", 0.0)
    badge = sentiment_badge(sentiment)
    bar   = conf_bar_html(conf, sentiment)
    st.markdown(
        f"""
<div class="aspect-card aspect-{sentiment}">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span class="aspect-name">📌 {name}</span>{badge}
  </div>
  {f'<div class="aspect-quote">"{quote}"</div>' if quote else ""}
  {bar}
  <div style="font-size:.75rem;color:#64748b;margin-top:3px;">Confidence: {conf:.0%}</div>
</div>""",
        unsafe_allow_html=True,
    )

def render_suggestions(suggestions: list):
    icons = ["✅", "⚡", "🎯", "📣", "🔧"]
    for i, sug in enumerate(suggestions):
        st.markdown(f'<div class="sug-card">{icons[i % len(icons)]} {sug}</div>', unsafe_allow_html=True)

# ─── Hero Banner + Language Toggle ─────────────────────────────────────────────
col_hero, col_lang = st.columns([6, 1])
with col_lang:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("EN", use_container_width=True,
                     type="primary" if st.session_state["lang"] == "EN" else "secondary"):
            st.session_state["lang"] = "EN"
            st.rerun()
    with c2:
        if st.button("中文", use_container_width=True,
                     type="primary" if st.session_state["lang"] == "ZH" else "secondary"):
            st.session_state["lang"] = "ZH"
            st.rerun()

with col_hero:
    st.markdown(
        f"""
<div class="hero-banner">
  <div class="hero-title">🔮 {t("app_title")}</div>
  <div class="hero-sub-title">{t("app_subtitle")}</div>
  <div class="hero-desc">{t("app_desc")}</div>
</div>""",
        unsafe_allow_html=True,
    )

# ─── Problem Statement ─────────────────────────────────────────────────────────
with st.expander(t("problem_expander"), expanded=False):
    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown(
            f'<div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);'
            f'border-radius:10px;padding:1.2rem;">'
            f'<div style="font-size:.8rem;font-weight:600;color:#fca5a5;text-transform:uppercase;'
            f'letter-spacing:.1em;margin-bottom:.6rem;">{t("problem_title")}</div>'
            f'<div style="color:#e2e8f0;font-size:.9rem;line-height:1.6;">{t("problem_body")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cp2:
        st.markdown(
            f'<div style="background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);'
            f'border-radius:10px;padding:1.2rem;">'
            f'<div style="font-size:.8rem;font-weight:600;color:#86efac;text-transform:uppercase;'
            f'letter-spacing:.1em;margin-bottom:.6rem;">{t("solution_title")}</div>'
            f'<div style="color:#e2e8f0;font-size:.9rem;line-height:1.6;">{t("solution_body")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([t("tab1"), t("tab2"), t("tab3"), t("tab4")])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Review
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(t("tab1_title"))
    st.caption(t("tab1_caption"))

    EXAMPLES = [
        "The delivery speed is very fast, but the packaging quality is very poor.",
        "Amazing product quality and great customer service! However, it's a bit overpriced.",
        "The food taste was excellent but the restaurant's ambiance was disappointing and noisy.",
        "Super easy to use app, but it crashes frequently. Support team was helpful though.",
    ]
    ex_choice = st.selectbox(t("tab1_example_label"),
                             [t("tab1_example_placeholder")] + EXAMPLES)
    default_text = "" if ex_choice == t("tab1_example_placeholder") else ex_choice

    review_input = st.text_area("", value=default_text, height=120,
                                placeholder=t("tab1_textarea_placeholder"))

    c_btn, c_note = st.columns([1, 4])
    with c_btn:
        analyze_btn = st.button(t("tab1_analyze_btn"), type="primary", use_container_width=True)
    with c_note:
        if not api_key:
            st.warning(t("tab1_api_warn"))

    if analyze_btn and review_input.strip():
        if not api_key:
            st.error(t("tab1_api_error"))
        else:
            with st.spinner(t("tab1_spinner")):
                try:
                    absa_result  = analyze_review(review_input, provider, api_key, model)
                    vader_result = analyze_vader(review_input)
                except Exception as e:
                    st.error(f"{t('tab1_fail')}: {e}")
                    st.stop()

            st.markdown("---")
            col_ai, col_trad = st.columns([3, 2], gap="large")

            with col_ai:
                st.markdown(f'<div class="section-header">{t("tab1_ai_header")}</div>', unsafe_allow_html=True)
                overall = absa_result.get("overall_sentiment", "neutral")
                st.markdown(
                    f'<div style="margin-bottom:1rem;font-size:.95rem;color:#94a3b8;">'
                    f'{t("tab1_overall")} {sentiment_badge(overall)}</div>',
                    unsafe_allow_html=True,
                )
                aspects = absa_result.get("aspects", [])
                if aspects:
                    for asp in aspects:
                        render_aspect_card(asp)
                    st.plotly_chart(aspect_bar_chart(aspects), use_container_width=True,
                                   config={"displayModeBar": False})
                else:
                    st.info(t("tab1_no_aspects"))

                st.markdown("---")
                st.markdown(f'<div class="section-header">{t("tab1_suggestions_header")}</div>',
                            unsafe_allow_html=True)
                sugs = absa_result.get("marketing_suggestions", [])
                if sugs:
                    render_suggestions(sugs)
                else:
                    st.caption(t("tab1_no_suggestions"))

                kws = absa_result.get("keywords", [])
                if kws:
                    badges = "  ".join(
                        f'<span style="background:rgba(109,40,217,.3);color:#c4b5fd;'
                        f'border-radius:5px;padding:2px 8px;font-size:.8rem;">{k}</span>'
                        for k in kws
                    )
                    st.markdown(f'{t("tab1_keywords")} {badges}', unsafe_allow_html=True)

            with col_trad:
                st.markdown(f'<div class="section-header">{t("tab1_trad_header")}</div>', unsafe_allow_html=True)
                label    = vader_result["label"]
                compound = vader_result["compound"]
                lcolor   = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}[label]

                # ── Single overall label (intentionally plain/muted) ──
                st.markdown(
                    f'<div class="vader-card" style="margin-bottom:1rem;">'
                    f'<div class="vader-label">{t("tab1_vader_only")}</div>'
                    f'<div class="vader-value" style="color:{lcolor};font-size:2.2rem;">{label.upper()}</div>'
                    f'<div style="color:#475569;font-size:.85rem;margin-top:.4rem;">Compound: {compound:+.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── What VADER cannot tell you (per detected aspect) ──
                st.markdown(
                    f'<div style="font-size:.78rem;font-weight:600;color:#64748b;'
                    f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem;">'
                    f'{t("tab1_vader_blind_spots")}</div>',
                    unsafe_allow_html=True,
                )
                detected_aspects = absa_result.get("aspects", [])
                if detected_aspects:
                    rows_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:.45rem .7rem;margin-bottom:.3rem;'
                        f'background:rgba(255,255,255,0.03);border-radius:6px;">'
                        f'<span style="color:#64748b;font-size:.88rem;">📌 {a["name"].replace("_"," ").title()}</span>'
                        f'<span style="color:#ef4444;font-size:.8rem;font-weight:600;">❌ Unknown</span>'
                        f'</div>'
                        for a in detected_aspects
                    )
                    st.markdown(rows_html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div style="color:#475569;font-size:.85rem;padding:.5rem;">No aspect data available.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="callout-warn" style="margin-top:1rem;">'
                    f'<b>{t("tab1_vader_warn_title")}</b><br>'
                    f'{t("tab1_vader_warn_body")}</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(t("tab2_title"))
    st.caption(t("tab2_caption"))

    _, c_dl = st.columns([3, 1])
    with c_dl:
        with open(os.path.join(os.path.dirname(__file__), "sample_data.csv"), "rb") as f:
            st.download_button(t("tab2_download_sample"), f, "sample_reviews.csv", "text/csv")

    uploaded = st.file_uploader(t("tab2_upload_label"), type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "review" not in df.columns:
            st.error(t("tab2_no_col_error"))
        else:
            st.dataframe(df.head(5), use_container_width=True)
            st.caption(f"{t('tab2_total')} **{len(df)}**")

            if not api_key:
                st.warning(t("tab2_api_warn"))
            elif st.button(t("tab2_run_btn"), type="primary"):
                reviews   = df["review"].astype(str).tolist()
                prog_bar  = st.progress(0, t("tab2_processing"))

                def upd(done, total):
                    prog_bar.progress(done / total,
                                      t("tab2_analyzing").format(done=done, total=total))

                with st.spinner(t("tab2_processing")):
                    absa_res   = analyze_batch(reviews, provider, api_key, model, upd)
                    vader_res  = analyze_batch_vader(reviews)
                prog_bar.empty()

                total_r   = len(absa_res)
                pos_count = sum(1 for r in absa_res if r.get("overall_sentiment") == "positive")
                neg_count = sum(1 for r in absa_res if r.get("overall_sentiment") == "negative")
                all_asp   = [a for r in absa_res for a in r.get("aspects", [])]
                top_asp   = "N/A"
                if all_asp:
                    from collections import Counter
                    top_asp = Counter(a["name"] for a in all_asp).most_common(1)[0][0].replace("_", " ").title()

                k1, k2, k3, k4 = st.columns(4)
                for col, val, lbl in [
                    (k1, str(total_r), t("tab2_kpi_total")),
                    (k2, f"{pos_count/total_r:.0%}", t("tab2_kpi_pos")),
                    (k3, f"{neg_count/total_r:.0%}", t("tab2_kpi_neg")),
                    (k4, top_asp, t("tab2_kpi_top")),
                ]:
                    with col:
                        st.markdown(
                            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                            f'<div class="kpi-label">{lbl}</div></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                c1b, c2b = st.columns(2)
                with c1b:
                    st.plotly_chart(sentiment_pie(absa_res), use_container_width=True,
                                   config={"displayModeBar": False})
                with c2b:
                    st.plotly_chart(aspect_grouped_bar(absa_res), use_container_width=True,
                                   config={"displayModeBar": False})

                wc_b64 = generate_wordcloud([r.get("keywords", []) for r in absa_res])
                if wc_b64:
                    st.markdown(t("tab2_wordcloud"))
                    st.markdown(
                        f'<img src="data:image/png;base64,{wc_b64}" '
                        f'style="width:100%;border-radius:10px;"/>',
                        unsafe_allow_html=True,
                    )

                # Download
                rows = []
                for r, v in zip(absa_res, vader_res):
                    base = {
                        "review": r.get("review", ""),
                        "ai_overall_sentiment": r.get("overall_sentiment", ""),
                        "vader_label": v["label"],
                        "vader_compound": v["compound"],
                        "keywords": ", ".join(r.get("keywords", [])),
                        "marketing_suggestions": " | ".join(r.get("marketing_suggestions", [])),
                    }
                    for a in r.get("aspects", []):
                        base[f"aspect_{a['name']}_sentiment"]   = a.get("sentiment", "")
                        base[f"aspect_{a['name']}_confidence"]  = a.get("confidence", "")
                    rows.append(base)
                st.download_button(t("tab2_download_results"),
                                   pd.DataFrame(rows).to_csv(index=False).encode(),
                                   "absa_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI vs Traditional
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(t("tab3_title"))
    st.caption(t("tab3_caption"))

    st.markdown(t("tab3_compare_title"))
    caps = tlist("tab3_capabilities")
    comp_df = pd.DataFrame({
        t("tab3_cap_col0"): caps,
        t("tab3_cap_col1"): ["✅", "❌", "❌", "❌", "⚠️ Limited", "❌", "✅"],
        t("tab3_cap_col2"): ["✅", "✅", "✅", "✅", "✅", "✅", "✅"],
    })
    st.dataframe(
        comp_df.style.apply(
            lambda col: [
                "color:#22c55e" if "✅" in str(v) else
                "color:#ef4444" if "❌" in str(v) else
                "color:#f59e0b"
                for v in col
            ],
            subset=[t("tab3_cap_col1"), t("tab3_cap_col2")],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown(t("tab3_examples_title"))

    DEMO_REVIEWS = [
        "The delivery speed is very fast, but the packaging quality is very poor.",
        "Amazing product quality and great customer service! However, it's a bit overpriced.",
        "The food taste was excellent but the restaurant was very noisy and dirty.",
        "Super easy to use, but it crashed twice. Support team was very responsive.",
        "Looks beautiful in photos but the actual fabric feels cheap and uncomfortable.",
    ]
    DEMO_VADER = [analyze_vader(r) for r in DEMO_REVIEWS]
    DEMO_ABSA_STATIC = [
        {"aspects": [{"name":"delivery","sentiment":"positive","confidence":.97,"quote":"delivery speed is very fast"},
                     {"name":"packaging","sentiment":"negative","confidence":.95,"quote":"packaging quality is very poor"}],
         "overall_sentiment":"neutral",
         "marketing_suggestions":["Highlight fast delivery in ad campaigns.","Urgently improve packaging; add quality guarantee."]},
        {"aspects": [{"name":"quality","sentiment":"positive","confidence":.96,"quote":"Amazing product quality"},
                     {"name":"customer_service","sentiment":"positive","confidence":.94,"quote":"great customer service"},
                     {"name":"price","sentiment":"negative","confidence":.88,"quote":"a bit overpriced"}],
         "overall_sentiment":"positive",
         "marketing_suggestions":["Use quality & service as core brand pillars.","Consider value bundles to address price perception."]},
        {"aspects": [{"name":"taste","sentiment":"positive","confidence":.98,"quote":"food taste was excellent"},
                     {"name":"ambiance","sentiment":"negative","confidence":.91,"quote":"very noisy and dirty"}],
         "overall_sentiment":"neutral",
         "marketing_suggestions":["Feature signature dishes in social content.","Address cleanliness; consider noise-reduction measures."]},
        {"aspects": [{"name":"usability","sentiment":"positive","confidence":.93,"quote":"Super easy to use"},
                     {"name":"reliability","sentiment":"negative","confidence":.92,"quote":"crashed twice"},
                     {"name":"customer_service","sentiment":"positive","confidence":.90,"quote":"Support team was very responsive"}],
         "overall_sentiment":"neutral",
         "marketing_suggestions":["Market the UX and support as differentiators.","Prioritize crash fixes in next sprint."]},
        {"aspects": [{"name":"design","sentiment":"positive","confidence":.89,"quote":"Looks beautiful in photos"},
                     {"name":"quality","sentiment":"negative","confidence":.94,"quote":"fabric feels cheap and uncomfortable"}],
         "overall_sentiment":"negative",
         "marketing_suggestions":["Audit material quality; upgrade fabric sourcing.","Update product photos to set realistic expectations."]},
    ]

    for i, (review, vader, absa) in enumerate(zip(DEMO_REVIEWS, DEMO_VADER, DEMO_ABSA_STATIC)):
        with st.expander(f'{t("tab3_review_label").format(n=i+1)} "{review[:65]}..."', expanded=(i == 0)):
            st.markdown(f'<div style="color:#cbd5e1;font-style:italic;margin-bottom:1rem;">"{review}"</div>',
                        unsafe_allow_html=True)
            ca, cv = st.columns(2)
            with ca:
                st.markdown(f'<div class="section-header">{t("tab3_ai_header")}</div>', unsafe_allow_html=True)
                for asp in absa["aspects"]:
                    render_aspect_card(asp)
                st.markdown(t("tab3_suggestions"))
                render_suggestions(absa["marketing_suggestions"])
            with cv:
                st.markdown(f'<div class="section-header">{t("tab3_vader_header")}</div>', unsafe_allow_html=True)
                label  = vader["label"]
                color  = {"positive":"#22c55e","negative":"#ef4444","neutral":"#94a3b8"}[label]
                st.markdown(
                    f'<div class="vader-card">'
                    f'<div class="vader-label">{t("tab3_vader_only_label")}</div>'
                    f'<div class="vader-value" style="color:{color};">{label.upper()}</div>'
                    f'<div style="color:#64748b;font-size:.85rem;">Compound: {vader["compound"]:+.4f}</div>'
                    f'<div class="vader-limit">{t("tab3_vader_no_action")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(f'<div class="callout">{t("tab3_conclusion")}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Testing & Validation
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(t("tab4_title"))
    st.caption(t("tab4_caption"))

    labeled_df     = load_labeled_data()
    unique_reviews = list(dict.fromkeys(r.strip().strip('"') for r in labeled_df["review"]))

    # ── Always-visible: Methodology overview ──────────────────────────────────
    st.markdown(f'<div class="section-header">{t("tab4_methodology_header")}</div>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, icon, step, desc in [
        (m1, "📋", t("tab4_step1_title"), t("tab4_step1_desc").format(n=len(labeled_df))),
        (m2, "🤖", t("tab4_step2_title"), t("tab4_step2_desc")),
        (m3, "📊", t("tab4_step3_title"), t("tab4_step3_desc")),
        (m4, "⚡", t("tab4_step4_title"), t("tab4_step4_desc")),
    ]:
        with col:
            st.markdown(
                f'<div style="background:rgba(30,27,75,.6);border:1px solid rgba(139,92,246,.2);'
                f'border-radius:10px;padding:1rem;text-align:center;height:140px;">'
                f'<div style="font-size:1.6rem;">{icon}</div>'
                f'<div style="font-size:.85rem;font-weight:600;color:#c4b5fd;margin:.4rem 0 .3rem;">{step}</div>'
                f'<div style="font-size:.78rem;color:#64748b;line-height:1.4;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Dataset info + expander ───────────────────────────────────────────────
    col_info, col_dl = st.columns([3, 1])
    with col_info:
        st.info(t("tab4_info").format(pairs=len(labeled_df), reviews=len(unique_reviews)))
    with col_dl:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        with open(os.path.join(os.path.dirname(__file__), "labeled_test_data.csv"), "rb") as f:
            st.download_button(t("tab4_download_test"), f, "labeled_test_data.csv", "text/csv",
                               use_container_width=True)

    with st.expander(t("tab4_view_data"), expanded=False):
        st.dataframe(labeled_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────────────────
    if not api_key:
        st.warning(t("tab4_api_warn"))
    run_test = st.button(t("tab4_run_btn"), type="primary", disabled=not api_key,
                         use_container_width=False)

    if run_test and api_key:
        prog = st.progress(0, t("tab4_spinner"))

        def upd_t(done, total):
            prog.progress(done / total, t("tab4_analyzing").format(done=done, total=total))

        with st.spinner(t("tab4_spinner")):
            absa_test  = analyze_batch(unique_reviews, provider, api_key, model, upd_t)
            for r, rev in zip(absa_test, unique_reviews):
                r["review"] = rev
            vader_test = analyze_batch_vader(unique_reviews)

        prog.empty()
        st.session_state["eval_metrics"]  = evaluate_results(labeled_df, absa_test)
        st.session_state["vader_metrics"] = vader_baseline_metrics(labeled_df, vader_test)

    metrics = st.session_state.get("eval_metrics")
    vader_m = st.session_state.get("vader_metrics")

    if metrics and vader_m:
        # ── KPI cards ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(t("tab4_results_title"))
        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl, color in [
            (k1, f"{metrics['aspect_detection_rate']:.1%}", t("tab4_kpi_detect"), "#a78bfa"),
            (k2, f"{metrics['sentiment_accuracy']:.1%}",   t("tab4_kpi_acc"),    "#22c55e"),
            (k3, f"{metrics['overall_f1']:.2f}",           t("tab4_kpi_absa_f1"),"#38bdf8"),
            (k4, f"{vader_m['overall_f1']:.2f}",           t("tab4_kpi_vader_f1"),"#64748b"),
        ]:
            with col:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">{val}</div>'
                    f'<div class="kpi-label">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── F1 comparison chart ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown(t("tab4_metrics_title"))
        classes  = ["positive", "negative", "neutral"]
        absa_f1  = [metrics["per_sentiment_metrics"][c]["f1"] for c in classes]
        vader_f1 = [vader_m["per_sentiment_metrics"][c]["f1"] for c in classes]
        fig = go.Figure(data=[
            go.Bar(name="SentiSight ABSA", x=classes, y=absa_f1,  marker_color="#a78bfa",
                   text=[f"{v:.2f}" for v in absa_f1],  textposition="outside"),
            go.Bar(name="VADER",           x=classes, y=vader_f1, marker_color="#334155",
                   text=[f"{v:.2f}" for v in vader_f1], textposition="outside"),
        ])
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#cbd5e1"),
            yaxis=dict(range=[0,1.15], title="F1 Score", gridcolor="rgba(255,255,255,0.07)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=320, margin=dict(t=30,b=30),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Per-class metrics table ───────────────────────────────────────────
        rows = []
        for cls in classes:
            am = metrics["per_sentiment_metrics"][cls]
            vm = vader_m["per_sentiment_metrics"][cls]
            rows.append({
                t("tab4_col_sentiment"): cls.capitalize(),
                t("tab4_col_absa_p"):   f"{am['precision']:.2f}",
                t("tab4_col_absa_r"):   f"{am['recall']:.2f}",
                t("tab4_col_absa_f"):   f"{am['f1']:.2f}",
                t("tab4_col_vader_p"):  f"{vm['precision']:.2f}",
                t("tab4_col_vader_r"):  f"{vm['recall']:.2f}",
                t("tab4_col_vader_f"):  f"{vm['f1']:.2f}",
                t("tab4_col_support"):  am["support"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Per-aspect detail ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(t("tab4_detail_title"))
        st.dataframe(pd.DataFrame(metrics["detail_rows"]), use_container_width=True, hide_index=True)

        # ── Summary callout ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            f'<div class="callout">'
            + t("tab4_summary").format(
                detected=metrics["detected_count"],
                total=metrics["total_aspects"],
                detect_rate=metrics["aspect_detection_rate"],
                acc=metrics["sentiment_accuracy"],
                absa_f1=metrics["overall_f1"],
                vader_f1=vader_m["overall_f1"],
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        # ── Empty state: prominent CTA ────────────────────────────────────────
        st.markdown(
            f'<div style="background:rgba(109,40,217,.08);border:1px dashed rgba(139,92,246,.4);'
            f'border-radius:14px;padding:2.5rem;text-align:center;margin-top:.5rem;">'
            f'<div style="font-size:2.8rem;margin-bottom:.6rem;">🧪</div>'
            f'<div style="font-size:1.05rem;font-weight:600;color:#c4b5fd;margin-bottom:.4rem;">'
            f'{t("tab4_placeholder")}</div>'
            f'<div style="font-size:.85rem;color:#64748b;max-width:500px;margin:0 auto;">'
            f'{t("tab4_placeholder_sub")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
