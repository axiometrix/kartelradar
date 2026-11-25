"""KartelRadar dashboard: streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kartelradar.config import DATA_DIR, MODEL_DIR  # noqa: E402
from kartelradar.model import RadarModel  # noqa: E402
from kartelradar.network import cobid_graph, suspicious_groups  # noqa: E402
from kartelradar.report import inspector_note, risk_level  # noqa: E402

BLUE, ORANGE = "#257de0", "#f47230"
STATUS = {"YUQORI": "#dc3c3d", "O'RTA": "#fbb315", "PAST": "#0aac23"}
ICON = {"YUQORI": "▲", "O'RTA": "●", "PAST": "▼"}
LAYOUT = dict(margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
              font=dict(size=13), xaxis=dict(gridcolor="rgba(128,128,128,.15)"), yaxis=dict(gridcolor="rgba(128,128,128,.15)"))

st.set_page_config(page_title="KartelRadar", layout="wide")
dataset = st.sidebar.radio("Ma'lumot", ["synthetic", "uzex"], format_func={"synthetic": "Sintetik (belgili)", "uzex": "etender.uzex.uz (real)"}.get)
path = DATA_DIR / f"scored_{dataset}.csv"
if not path.exists():
    st.error(f"{path.name} topilmadi. Avval `python scripts/train.py` (uzex uchun `--score-uzex`) ni ishga tushiring.")
    st.stop()


@st.cache_data
def load(p):
    return pd.read_csv(p, dtype={"tender_id": str, "customer_inn": str, "provider_inn": str}).sort_values("risk", ascending=False)


@st.cache_resource
def get_model(ds):
    return RadarModel.load(MODEL_DIR / ("xgb_full.joblib" if ds == "synthetic" else "uzex_lot.joblib"))


df = load(path)
df["level"] = df["risk"].map(risk_level)
model = get_model(dataset)

st.title("KartelRadar")
st.caption("Davlat xaridlarida til biriktirish belgilarini AI bilan skrining qilish. Natija tekshiruv uchun signal, isbot emas.")

# --- filtrlar bitta qatorda
c1, c2, c3 = st.columns(3)
regions = ["Hammasi"] + sorted(df["region"].astype(str).unique().tolist()) if dataset == "synthetic" else ["Hammasi"]
region = c1.selectbox("Hudud", regions)
cats = ["Hammasi"] + df["category_name"].astype(str).value_counts().head(40).index.tolist()
cat = c2.selectbox("Toifa", cats)
min_risk = c3.slider("Minimal risk", 0, 100, 0)
f = df[df["risk"] >= min_risk]
if region != "Hammasi":
    f = f[f["region"] == region]
if cat != "Hammasi":
    f = f[f["category_name"] == cat]

# --- KPI
k = st.columns(4)
k[0].metric("Tenderlar", f"{len(f):,}")
k[1].metric(f"{ICON['YUQORI']} Yuqori risk (70+)", f"{(f['level'] == 'YUQORI').sum():,}")
k[2].metric("Yuqori riskli summa, mlrd so'm", f"{f.loc[f['level'] == 'YUQORI', 'deal_cost'].sum() / 1e9:,.1f}")
mpath = MODEL_DIR / "metrics.json"
if mpath.exists():
    mt = json.loads(mpath.read_text())["full"]
    k[3].metric("Test PR-AUC / Recall", f"{mt['pr_auc']:.2f} / {mt['recall']:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["Risk reytingi", "Tender kartasi", "Firmalar grafi", "Model benchmark"])

with tab1:
    show = f[["tender_id", "level", "risk", "customer_name", "provider_name", "category_name", "region",
              "start_cost", "deal_cost", "participants_count", "rebate"]].head(500).copy()
    show["level"] = show["level"].map(lambda v: f"{ICON[v]} {v}")
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={"risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                                "rebate": st.column_config.NumberColumn("Chegirma", format="percent"),
                                "start_cost": st.column_config.NumberColumn("Boshlang'ich narx", format="localized"),
                                "deal_cost": st.column_config.NumberColumn("G'olib narxi", format="localized")})
    hi = f[f["level"] == "YUQORI"]["rebate"].clip(-0.05, 0.6)
    lo = f[f["level"] != "YUQORI"]["rebate"].clip(-0.05, 0.6)
    fig = go.Figure()
    fig.add_histogram(x=lo, name="Past/o'rta risk", marker_color=BLUE, opacity=0.85, histnorm="percent", xbins=dict(size=0.01))
    fig.add_histogram(x=hi, name="Yuqori risk", marker_color=ORANGE, opacity=0.85, histnorm="percent", xbins=dict(size=0.01))
    fig.update_layout(**LAYOUT, barmode="overlay", title="Chegirma taqsimoti: yuqori riskli tenderlarda narx deyarli tushmaydi",
                      xaxis_title="Chegirma", yaxis_title="Tenderlar ulushi, %", xaxis_tickformat=".0%", height=320,
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    tid = st.selectbox("Tender", f["tender_id"].head(200).tolist())
    if tid:
        row = f[f["tender_id"] == tid]
        reasons = model.explain(row, top_k=8)[0]
        r = row.iloc[0]
        a, b = st.columns([1, 1])
        a.markdown(f"### {ICON[r['level']]} Risk {r['risk']:.0f} / 100 ({r['level']})")
        a.markdown(f"**Buyurtmachi:** {r['customer_name']}  \n**G'olib:** {r['provider_name']}  \n"
                   f"**Toifa:** {r['category_name']}  \n**Ishtirokchilar:** {r['participants_count']} · "
                   f"**Chegirma:** {r['rebate']:.1%}")
        if reasons:
            rs = pd.DataFrame(reasons)[::-1]
            fig = go.Figure(go.Bar(x=rs["shap"], y=rs["label"], orientation="h", marker_color=BLUE,
                                   hovertemplate="%{y}<br>SHAP hissasi: %{x:.3f}<extra></extra>"))
            fig.update_layout(**LAYOUT, title="Riskni oshirgan omillar (SHAP)", height=60 + 34 * len(rs))
            b.plotly_chart(fig, use_container_width=True)
        st.text_area("Inspektor uchun ma'lumotnoma", inspector_note({**r.to_dict()}, reasons), height=260)

with tab3:
    bpath = DATA_DIR / dataset / "bids.csv"
    if not bpath.exists():
        st.info("Bu datasetda individual takliflar yo'q. Graf sintetik yoki ishtirokchilar ro'yxati bilan ishlaydi.")
    else:
        bids = pd.read_csv(bpath, dtype={"firm_inn": str, "tender_id": str})
        minw = st.slider("Kamida necha marta birga qatnashgan", 2, 15, 5)
        g = cobid_graph(bids, min_weight=minw)
        groups = suspicious_groups(g)
        firm_risk = df.groupby("provider_inn")["risk"].mean()
        st.write(f"{g.number_of_nodes()} firma, {g.number_of_edges()} bog'lanish, {len(groups)} ta zich guruh")
        if g.number_of_nodes():
            pos = nx.spring_layout(g, seed=1, weight="weight")
            ex, ey = [], []
            for u, v in g.edges():
                ex += [pos[u][0], pos[v][0], None]
                ey += [pos[u][1], pos[v][1], None]
            nodes = list(g.nodes())
            risk = [float(firm_risk.get(n, 0)) for n in nodes]
            fig = go.Figure()
            fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="rgba(128,128,128,.35)"), hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
                            marker=dict(size=11, color=risk, cmin=0, cmax=100,
                                        colorscale=[[0, "#cce3fd"], [1, "#1b64be"]],
                                        colorbar=dict(title="O'rtacha risk"), line=dict(width=2, color="white")),
                            text=[f"STIR {n}<br>o'rtacha risk {r:.0f}<br>bog'lanishlar {g.degree(n)}" for n, r in zip(nodes, risk)],
                            hoverinfo="text", showlegend=False)
            fig.update_layout(**LAYOUT, height=520, title="Birga qatnashish grafi")
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(groups.head(20), use_container_width=True, hide_index=True)

with tab4:
    bp = MODEL_DIR / "benchmark.csv"
    if bp.exists():
        bt = pd.read_csv(bp)
        fig = go.Figure(go.Bar(x=bt["pr_auc"], y=bt["model"], orientation="h", marker_color=BLUE,
                               text=bt["pr_auc"].map("{:.2f}".format), textposition="outside",
                               hovertemplate="%{y}<br>PR-AUC %{x:.3f}<extra></extra>"))
        fig.update_layout(**LAYOUT, title="PR-AUC (test, bozor bo'yicha ajratilgan)", xaxis_range=[0, 1.1], height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bt, use_container_width=True, hide_index=True)
        st.caption("Sintetik data natijalari optimistik. Real baholash uchun belgilangan ochiq datasetlar "
                   "(masalan Shveytsariya/Yaponiya bid-rigging datasetlari) yoki tashkilotchi beradigan data kerak.")
