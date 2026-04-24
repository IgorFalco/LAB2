from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import ModelConfig, solve_airport_stand_allocation

# ── Configuração da página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Alocação de Posições – Confins",
    page_icon="✈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

# ── Paletas ─────────────────────────────────────────────────────────────────

TYPE_COLOR = {
    "turnaround": "#4C9BE8",
    "arrival":    "#2ECC71",
    "parking":    "#95A5A6",
    "departure":  "#E74C3C",
}

STAND_COLOR = {
    "contato":        "#27AE60",
    "remoto":         "#F39C12",
    "estacionamento": "#7F8C8D",
}

OBJECTIVE_LABELS = {
    "walking_distance": "Minimizar distância de caminhada",
    "contact_share":    "Maximizar uso de posições de contato",
    "tow_count":        "Minimizar número de reboques",
    "revenue":          "Maximizar receita comercial",
}

# ── Tabs principais ─────────────────────────────────────────────────────────

st.title("✈ Alocação de Posições – Aeroporto Internacional de Confins")
st.caption("ELE634 – Laboratório de Sistemas II · UFMG")

tab_config, tab_results = st.tabs(["⚙️ Configuração e Execução", "📊 Resultados"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

with tab_config:

    st.subheader("Arquivos de entrada")
    st.markdown(
        "Faça upload dos arquivos CSV ou marque a opção abaixo para usar os arquivos já presentes na pasta."
    )

    use_default = st.checkbox("Usar arquivos padrão da pasta (voos.csv, posicoes.csv, ...)", value=True)

    if not use_default:
        c1, c2 = st.columns(2)
        up_voos     = c1.file_uploader("voos.csv",                    type="csv", key="voos")
        up_posicoes = c1.file_uploader("posicoes.csv",                 type="csv", key="pos")
        up_cats     = c2.file_uploader("categoriaaeronaves.csv",       type="csv", key="cats")
        up_specs    = c2.file_uploader("especificacoesaeronaves.csv (opcional)", type="csv", key="specs")
    else:
        up_voos = up_posicoes = up_cats = up_specs = None

    st.divider()
    st.subheader("Parâmetros de otimização")

    col_p1, col_p2, col_p3 = st.columns(3)

    objective = col_p1.selectbox(
        "Objetivo",
        options=list(OBJECTIVE_LABELS.keys()),
        format_func=lambda k: OBJECTIVE_LABELS[k],
    )

    time_limit = col_p2.slider(
        "Tempo máximo (segundos)",
        min_value=30, max_value=600, value=300, step=30,
    )

    mip_gap_pct = col_p3.slider(
        "Gap MIP (%)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    )

    with st.expander("Parâmetros avançados"):
        col_a1, col_a2, col_a3 = st.columns(3)
        min_turn = col_a1.number_input("Turnaround mínimo (min)", min_value=5,  max_value=120, value=30)
        max_turn = col_a2.number_input("Turnaround máximo (min)", min_value=60, max_value=720, value=480)
        tow_thr  = col_a3.number_input("Limiar longa permanência (min)", min_value=60, max_value=360, value=180)

    st.divider()

    # ── Botão de execução ───────────────────────────────────────────────────

    ready = use_default or (up_voos and up_posicoes and up_cats)

    if not ready:
        st.warning("Faça upload de voos.csv, posicoes.csv e categoriaaeronaves.csv para continuar.")

    if st.button("▶ Executar Otimização", disabled=not ready, type="primary", use_container_width=True):

        config = ModelConfig(
            objective=objective,
            time_limit_seconds=time_limit,
            mip_gap=mip_gap_pct / 100.0,
            verbose=False,
            min_turnaround_minutes=int(min_turn),
            max_turnaround_minutes=int(max_turn),
            tow_threshold_minutes=int(tow_thr),
        )

        # Resolve caminhos (default ou upload)
        if use_default:
            paths = {
                "flights":    BASE_DIR / "voos.csv",
                "positions":  BASE_DIR / "posicoes.csv",
                "categories": BASE_DIR / "categoriaaeronaves.csv",
                "specs":      BASE_DIR / "especificacoesaeronaves.csv",
            }
            tmp_dir = None
        else:
            tmp_dir = tempfile.mkdtemp()

            def save_upload(uploaded, name):
                p = Path(tmp_dir) / name
                p.write_bytes(uploaded.getvalue())
                return p

            paths = {
                "flights":    save_upload(up_voos,     "voos.csv"),
                "positions":  save_upload(up_posicoes, "posicoes.csv"),
                "categories": save_upload(up_cats,     "categoriaaeronaves.csv"),
                "specs":      save_upload(up_specs, "especificacoesaeronaves.csv") if up_specs else None,
            }

        output_dir = BASE_DIR / "outputs"

        log_placeholder = st.empty()
        progress_bar    = st.progress(0, text="Preparando dados...")

        try:
            progress_bar.progress(10, text="Carregando e processando dados...")

            # Captura stdout do Gurobi
            old_stdout = sys.stdout
            sys.stdout = gurobi_log = io.StringIO()

            with st.spinner("Otimizando... isso pode levar alguns minutos."):
                progress_bar.progress(25, text="Construindo modelo...")
                result = solve_airport_stand_allocation(
                    flights_path=paths["flights"],
                    positions_path=paths["positions"],
                    aircraft_categories_path=paths["categories"],
                    aircraft_specs_path=paths["specs"],
                    output_dir=output_dir,
                    config=config,
                )

            sys.stdout = old_stdout
            log_text = gurobi_log.getvalue()

            progress_bar.progress(100, text="Concluído!")

            if result.allocation.empty:
                st.error(f"Otimização encerrada sem solução. Status: {result.status}")
            else:
                st.session_state["alloc"]     = result.allocation
                st.session_state["tows"]      = result.tows
                st.session_state["status"]    = result.status
                st.session_state["obj_value"] = result.objective_value
                st.session_state["objective"] = objective
                st.session_state["log"]       = log_text

                st.success(
                    f"✅ Otimização concluída! Status: **{result.status}** · "
                    f"Objetivo: **{result.objective_value:,.0f}** · "
                    f"Vá para a aba **Resultados**."
                )

                with st.expander("Log do Gurobi"):
                    st.code(log_text or "(verbose=False, sem log)")

        except Exception as e:
            sys.stdout = old_stdout
            progress_bar.empty()
            st.error(f"Erro durante a otimização: {e}")
            raise

    # Mostra último resultado se já existir
    if "status" in st.session_state and "alloc" in st.session_state:
        st.info(
            f"Último resultado carregado: status **{st.session_state['status']}**, "
            f"{len(st.session_state['alloc'])} operações alocadas, "
            f"{len(st.session_state['tows'])} reboques."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:

    if "alloc" not in st.session_state:
        # Tenta carregar arquivos salvos em disco
        alloc_path = BASE_DIR / "outputs" / "alocacao_resultado.csv"
        tows_path  = BASE_DIR / "outputs" / "reboques_resultado.csv"

        if alloc_path.exists():
            alloc = pd.read_csv(alloc_path)
            tows  = pd.read_csv(tows_path) if tows_path.exists() else pd.DataFrame()
            alloc["start_time"] = pd.to_datetime(alloc["start_time"])
            alloc["end_time"]   = pd.to_datetime(alloc["end_time"])
            st.info("Exibindo resultados do último arquivo salvo em disco.")
        else:
            st.warning("Nenhum resultado disponível. Execute a otimização na aba **Configuração**.")
            st.stop()
    else:
        alloc = st.session_state["alloc"].copy()
        tows  = st.session_state["tows"].copy()
        alloc["start_time"] = pd.to_datetime(alloc["start_time"])
        alloc["end_time"]   = pd.to_datetime(alloc["end_time"])

    # ── KPIs ────────────────────────────────────────────────────────────────

    pax_contato = alloc[alloc["is_contact"] == 1]["pax"].sum()
    pax_total   = alloc["pax"].sum()
    pct_contato = pax_contato / pax_total * 100 if pax_total > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Operações alocadas",  f"{len(alloc):,}")
    k2.metric("Posições utilizadas", alloc["stand_id"].nunique())
    k3.metric("Reboques",            f"{len(tows):,}")
    k4.metric("PAX em contato",      f"{pct_contato:.1f}%")
    k5.metric("Longa permanência",   alloc[alloc["operation_type"] == "parking"]["visit_id"].nunique())

    if "obj_value" in st.session_state:
        obj_label = OBJECTIVE_LABELS.get(st.session_state.get("objective", ""), "Objetivo")
        st.caption(f"**{obj_label}:** {st.session_state['obj_value']:,.0f}")

    st.divider()

    # ── Gantt ────────────────────────────────────────────────────────────────

    st.subheader("Gráfico de Gantt – Alocação por Posição")

    c1, c2, c3 = st.columns([2, 2, 2])

    all_dates       = sorted(alloc["start_time"].dt.date.unique())
    selected_date   = c1.selectbox("Data", all_dates)
    stand_types     = ["Todos"] + sorted(alloc["stand_type"].unique())
    sel_stand_type  = c2.selectbox("Tipo de posição", stand_types)
    op_types        = ["Todos"] + sorted(alloc["operation_type"].unique())
    sel_op_type     = c3.selectbox("Tipo de operação", op_types)

    df = alloc[alloc["start_time"].dt.date == selected_date].copy()
    if sel_stand_type != "Todos":
        df = df[df["stand_type"] == sel_stand_type]
    if sel_op_type != "Todos":
        df = df[df["operation_type"] == sel_op_type]

    min_h = int(df["start_time"].dt.hour.min()) if not df.empty else 0
    max_h = int(df["end_time"].dt.hour.max())   if not df.empty else 23
    h_start, h_end = st.slider(
        "Janela de horário",
        min_value=0, max_value=23,
        value=(min_h, min(max_h, min_h + 4)),
        step=1,
    )
    df = df[
        (df["start_time"].dt.hour <  h_end + 1) &
        (df["end_time"].dt.hour   >= h_start)
    ]

    if df.empty:
        st.info("Nenhuma operação para os filtros selecionados.")
    else:
        fig_gantt = px.timeline(
            df.sort_values("stand_id"),
            x_start="start_time", x_end="end_time",
            y="stand_id",
            color="operation_type",
            color_discrete_map=TYPE_COLOR,
            hover_data={
                "visit_id": True,
                "operation_type": True,
                "aircraft_category": True,
                "pax": True,
                "stand_type": True,
                "start_time": "|%H:%M",
                "end_time":   "|%H:%M",
                "stand_id":   False,
            },
            labels={"operation_type": "Tipo", "stand_id": "Posição"},
            height=max(420, len(df["stand_id"].unique()) * 24),
        )
        fig_gantt.update_layout(
            xaxis_title="Horário",
            yaxis_title="Posição",
            legend_title="Tipo de operação",
            yaxis={"categoryorder": "category ascending"},
            margin=dict(l=60, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

    st.divider()

    # ── Distribuições ────────────────────────────────────────────────────────

    st.subheader("Distribuições")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Operações por tipo de posição**")
        fig = px.pie(
            alloc["stand_type"].value_counts().reset_index().rename(columns={"stand_type": "Tipo", "count": "Qtd"}),
            values="Qtd", names="Tipo",
            color="Tipo", color_discrete_map=STAND_COLOR, hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Operações por categoria de aeronave**")
        fig = px.bar(
            alloc["aircraft_category"].value_counts().reset_index().rename(columns={"aircraft_category": "Categoria", "count": "Qtd"}),
            x="Categoria", y="Qtd",
            color="Categoria",
            color_discrete_sequence=px.colors.qualitative.Safe,
            text="Qtd",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        st.markdown("**Estacionamento de longa permanência por tipo**")
        park = alloc[alloc["operation_type"] == "parking"]
        fig = px.bar(
            park["stand_type"].value_counts().reset_index().rename(columns={"stand_type": "Tipo", "count": "Qtd"}),
            x="Tipo", y="Qtd",
            color="Tipo", color_discrete_map=STAND_COLOR, text="Qtd",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Utilização dos stands ────────────────────────────────────────────────

    st.subheader("Utilização das posições")

    util = (
        alloc.groupby(["stand_id", "stand_type"])
        .agg(operacoes=("operation_id", "count"), pax_total=("pax", "sum"))
        .reset_index()
        .sort_values("operacoes", ascending=False)
    )
    fig = px.bar(
        util, x="stand_id", y="operacoes",
        color="stand_type", color_discrete_map=STAND_COLOR,
        hover_data={"pax_total": True, "stand_type": True},
        labels={"stand_id": "Posição", "operacoes": "Nº de operações", "stand_type": "Tipo"},
        height=350,
    )
    fig.update_layout(
        xaxis={"categoryorder": "total descending"},
        legend_title="Tipo de posição",
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Reboques ─────────────────────────────────────────────────────────────

    st.subheader(f"Reboques identificados ({len(tows)})")

    if tows.empty:
        st.info("Nenhum reboque nesta solução.")
    else:
        tows_detail = (
            tows
            .merge(
                alloc[["operation_id", "stand_id", "start_time", "aircraft_category"]],
                on="operation_id", how="left",
            )
            .merge(
                alloc[["operation_id", "stand_id"]].rename(
                    columns={"operation_id": "successor_operation_id", "stand_id": "stand_destino"}
                ),
                on="successor_operation_id", how="left",
            )
            .rename(columns={"stand_id": "stand_origem", "start_time": "horario"})
        )
        tows_detail["horario"] = pd.to_datetime(tows_detail["horario"]).dt.strftime("%H:%M")

        st.dataframe(
            tows_detail[[
                "visit_id", "operation_id", "successor_operation_id",
                "stand_origem", "stand_destino", "aircraft_category", "horario",
            ]].reset_index(drop=True),
            use_container_width=True,
            height=300,
        )

    # ── Download ─────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("Download dos resultados")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇ alocacao_resultado.csv",
        data=alloc.to_csv(index=False).encode("utf-8"),
        file_name="alocacao_resultado.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if not tows.empty:
        d2.download_button(
            "⬇ reboques_resultado.csv",
            data=tows.to_csv(index=False).encode("utf-8"),
            file_name="reboques_resultado.csv",
            mime="text/csv",
            use_container_width=True,
        )
