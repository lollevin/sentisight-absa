import base64
import io

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#94a3b8",
    "error": "#f59e0b",
}

CHART_TEMPLATE = "plotly_dark"
PAPER_BG = "rgba(15,17,26,0)"
PLOT_BG = "rgba(255,255,255,0.04)"


def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0"), x=0.5),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        margin=dict(t=50, b=30, l=20, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )


def aspect_bar_chart(aspects: list[dict]) -> go.Figure:
    if not aspects:
        return go.Figure()

    names = [a["name"].replace("_", " ").title() for a in aspects]
    confidences = [a["confidence"] for a in aspects]
    sentiments = [a["sentiment"] for a in aspects]
    colors = [SENTIMENT_COLORS.get(s, "#94a3b8") for s in sentiments]

    fig = go.Figure(
        go.Bar(
            x=confidences,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{s.capitalize()}  {c:.0%}" for s, c in zip(sentiments, confidences)],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout("Aspect Sentiment Breakdown"),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=max(200, len(aspects) * 52 + 60),
        bargap=0.25,
    )
    return fig


def sentiment_pie(results: list[dict]) -> go.Figure:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        label = r.get("overall_sentiment", "neutral")
        if label in counts:
            counts[label] += 1
        else:
            counts["neutral"] += 1

    fig = go.Figure(
        go.Pie(
            labels=[k.capitalize() for k in counts],
            values=list(counts.values()),
            marker=dict(colors=[SENTIMENT_COLORS[k] for k in counts]),
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout("Overall Sentiment Distribution"),
        height=320,
        showlegend=False,
    )
    return fig


def aspect_grouped_bar(results: list[dict]) -> go.Figure:
    aspect_counts: dict[str, dict] = {}
    for r in results:
        for a in r.get("aspects", []):
            name = a["name"].replace("_", " ").title()
            s = a.get("sentiment", "neutral")
            aspect_counts.setdefault(name, {"positive": 0, "negative": 0, "neutral": 0})
            aspect_counts[name][s] = aspect_counts[name].get(s, 0) + 1

    if not aspect_counts:
        return go.Figure()

    aspects_sorted = sorted(aspect_counts, key=lambda k: sum(aspect_counts[k].values()), reverse=True)[:10]
    traces = []
    for sentiment, color in [("positive", "#22c55e"), ("negative", "#ef4444"), ("neutral", "#94a3b8")]:
        traces.append(
            go.Bar(
                name=sentiment.capitalize(),
                x=aspects_sorted,
                y=[aspect_counts[a].get(sentiment, 0) for a in aspects_sorted],
                marker_color=color,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_base_layout("Aspect Sentiment Distribution (Top 10)"),
        barmode="group",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.08)"),
        height=360,
    )
    return fig


def vader_gauge(compound: float) -> go.Figure:
    color = "#22c55e" if compound >= 0.05 else "#ef4444" if compound <= -0.05 else "#94a3b8"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=compound,
            number=dict(font=dict(size=28, color=color)),
            gauge=dict(
                axis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["-1", "-0.5", "0", "0.5", "1"]),
                bar=dict(color=color, thickness=0.25),
                bgcolor="rgba(255,255,255,0.05)",
                steps=[
                    dict(range=[-1, -0.05], color="rgba(239,68,68,0.15)"),
                    dict(range=[-0.05, 0.05], color="rgba(148,163,184,0.1)"),
                    dict(range=[0.05, 1], color="rgba(34,197,94,0.15)"),
                ],
                threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=compound),
            ),
        )
    )
    fig.update_layout(
        **_base_layout("VADER Compound Score"),
        height=220,
    )
    return fig


def generate_wordcloud(keywords_list: list[list[str]]) -> str:
    from wordcloud import WordCloud

    freq: dict[str, int] = {}
    for kws in keywords_list:
        for kw in kws:
            freq[kw.lower()] = freq.get(kw.lower(), 0) + 1

    if not freq:
        return ""

    wc = WordCloud(
        width=700,
        height=300,
        background_color=None,
        mode="RGBA",
        colormap="cool",
        max_words=60,
        prefer_horizontal=0.85,
    ).generate_from_frequencies(freq)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
