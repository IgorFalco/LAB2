from __future__ import annotations
from main import ModelConfig, solve_airport_stand_allocation

import io
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    "departure":  "#E5FF00",
    "Azul":       "#1E00FF",
    "Latam":      "#FF0000",
    "Gol":        "#FFA600",

}

OPERATION_LABELS = {
    "turnaround": "Chegada → Partida",
    "arrival": "Chegada",
    "parking": "Estacionado",
    "departure": "Partida",
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


def format_duration_minutes(minutes: float) -> str:
    if minutes is None or pd.isna(minutes):
        return ""

    total_minutes = int(round(float(minutes)))
    hours, mins = divmod(total_minutes, 60)

    if hours <= 0:
        return f"{mins} min"
    if mins == 0:
        return f"{hours} h"
    return f"{hours} h {mins} min"


_STAND_ID_INT_RE = re.compile(r"^(\d+)(?:\.0+)?$")


def normalize_stand_id(value) -> str:
    """Normaliza IDs de posição para evitar rótulos como '107.0'.

    - Se o valor parecer inteiro (ex.: 107, 107.0, '107.0'), retorna só os dígitos ('107').
    - Caso contrário, retorna o texto original (strip).
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    match = _STAND_ID_INT_RE.match(text)
    if match:
        return match.group(1)
    return text


def stand_sort_key(value: str):
    try:
        return int(str(value))
    except Exception:
        return str(value)

# ── Tabs principais ─────────────────────────────────────────────────────────


st.title("✈ Alocação de Posições – Aeroporto Internacional de Confins")
st.caption("ELE634 – Laboratório de Sistemas II · UFMG")

tab_config, tab_results = st.tabs(
    ["⚙️ Configuração e Execução", "📊 Resultados"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

with tab_config:

    st.subheader("Arquivos de entrada")
    st.markdown(
        "Faça upload dos arquivos CSV ou marque a opção abaixo para usar os arquivos já presentes na pasta."
    )

    use_default = st.checkbox(
        "Usar arquivos padrão da pasta (voos.csv, posicoes.csv, ...)", value=True)

    if not use_default:
        c1, c2 = st.columns(2)
        up_voos = c1.file_uploader(
            "voos.csv",                    type="csv", key="voos")
        up_posicoes = c1.file_uploader(
            "posicoes.csv",                 type="csv", key="pos")
        up_cats = c2.file_uploader(
            "categoriaaeronaves.csv",       type="csv", key="cats")
        up_specs = c2.file_uploader(
            "especificacoesaeronaves.csv (opcional)", type="csv", key="specs")
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
        min_turn = col_a1.number_input(
            "Turnaround mínimo (min)", min_value=5,  max_value=120, value=30)
        max_turn = col_a2.number_input(
            "Turnaround máximo (min)", min_value=60, max_value=720, value=480)
        tow_thr = col_a3.number_input(
            "Limiar longa permanência (min)", min_value=60, max_value=360, value=180)

    st.divider()

    # ── Botão de execução ───────────────────────────────────────────────────

    ready = use_default or (up_voos and up_posicoes and up_cats)

    if not ready:
        st.warning(
            "Faça upload de voos.csv, posicoes.csv e categoriaaeronaves.csv para continuar.")

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
        progress_bar = st.progress(0, text="Preparando dados...")

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
                st.error(
                    f"Otimização encerrada sem solução. Status: {result.status}")
            else:
                st.session_state["alloc"] = result.allocation
                st.session_state["tows"] = result.tows
                st.session_state["status"] = result.status
                st.session_state["obj_value"] = result.objective_value
                st.session_state["objective"] = objective
                st.session_state["log"] = log_text

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
        tows_path = BASE_DIR / "outputs" / "reboques_resultado.csv"

        if alloc_path.exists():
            alloc = pd.read_csv(alloc_path)
            tows = pd.read_csv(
                tows_path) if tows_path.exists() else pd.DataFrame()
            alloc["start_time"] = pd.to_datetime(alloc["start_time"])
            alloc["end_time"] = pd.to_datetime(alloc["end_time"])
            st.info("Exibindo resultados do último arquivo salvo em disco.")
        else:
            st.warning(
                "Nenhum resultado disponível. Execute a otimização na aba **Configuração**.")
            st.stop()
    else:
        alloc = st.session_state["alloc"].copy()
        tows = st.session_state["tows"].copy()
        alloc["start_time"] = pd.to_datetime(alloc["start_time"])
        alloc["end_time"] = pd.to_datetime(alloc["end_time"])

    # Evita números como 107.0, 272.0 (pandas pode inferir float ao ler CSV)
    if "stand_id" in alloc.columns:
        alloc["stand_id"] = alloc["stand_id"].apply(normalize_stand_id)
    if "pax" in alloc.columns:
        alloc["pax"] = pd.to_numeric(
            alloc["pax"], errors="coerce").fillna(0).astype(int)
    if "aircraft_category" in alloc.columns:
        alloc["aircraft_category"] = alloc["aircraft_category"].astype(str)
    if "visit_id" in alloc.columns:
        alloc["visit_id"] = alloc["visit_id"].astype(str)
    if not tows.empty and "operation_id" in tows.columns:
        tows["operation_id"] = tows["operation_id"].astype(str)

    # ── KPIs ────────────────────────────────────────────────────────────────

    pax_contato = alloc[alloc["is_contact"] == 1]["pax"].sum()
    pax_total = alloc["pax"].sum()
    pct_contato = pax_contato / pax_total * 100 if pax_total > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Operações alocadas",  f"{len(alloc):,}")
    k2.metric("Posições utilizadas", alloc["stand_id"].nunique())
    k3.metric("Reboques",            f"{len(tows):,}")
    k4.metric("PAX em contato",      f"{pct_contato:.1f}%")
    k5.metric("Longa permanência",
              alloc[alloc["operation_type"] == "parking"]["visit_id"].nunique())

    if "obj_value" in st.session_state:
        obj_label = OBJECTIVE_LABELS.get(
            st.session_state.get("objective", ""), "Objetivo")
        st.caption(f"**{obj_label}:** {st.session_state['obj_value']:,.0f}")

    st.divider()

    # ── Gantt ────────────────────────────────────────────────────────────────

    st.subheader("Gráfico de Gantt – Alocação por Posição")

    c1, c2, c3 = st.columns([2, 2, 2])

    all_dates = sorted(alloc["start_time"].dt.date.unique())
    selected_date = c1.selectbox("Data", all_dates)
    stand_types = ["Todos"] + sorted(alloc["stand_type"].unique())
    sel_stand_type = c2.selectbox("Tipo de posição", stand_types)
    op_types = ["Todos"] + sorted(alloc["operation_type"].unique())
    sel_op_type = c3.selectbox("Tipo de operação", op_types)

    df = alloc[alloc["start_time"].dt.date == selected_date].copy()
    if sel_stand_type != "Todos":
        df = df[df["stand_type"] == sel_stand_type]
    if sel_op_type != "Todos":
        df = df[df["operation_type"] == sel_op_type]

    min_h = int(df["start_time"].dt.hour.min()) if not df.empty else 0
    max_h = int(df["end_time"].dt.hour.max()) if not df.empty else 23
    h_start, h_end = st.slider(
        "Janela de horário",
        min_value=0, max_value=23,
        value=(min_h, min(max_h, min_h + 4)),
        step=1,
    )
    df = df[
        (df["start_time"].dt.hour < h_end + 1) &
        (df["end_time"].dt.hour >= h_start)
    ]

    if df.empty:
        st.info("Nenhuma operação para os filtros selecionados.")
    else:
        visit_bounds = (
            alloc.groupby("visit_id")
            .agg(visit_start=("start_time", "min"), visit_end=("end_time", "max"))
            .reset_index()
        )
        df = df.merge(visit_bounds, on="visit_id", how="left")

        df["operation_label"] = df["operation_type"].map(
            OPERATION_LABELS).fillna(df["operation_type"])
        df["op_duration_min"] = (
            df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
        df["ground_duration_min"] = (
            df["visit_end"] - df["visit_start"]).dt.total_seconds() / 60.0
        df["op_duration"] = df["op_duration_min"].apply(
            format_duration_minutes)
        df["ground_duration"] = df["ground_duration_min"].apply(
            format_duration_minutes)

        type_color_labels = {OPERATION_LABELS.get(
            k, k): v for k, v in TYPE_COLOR.items()}

        stand_order = sorted(
            df["stand_id"].unique().tolist(), key=stand_sort_key)

        fig_gantt = px.timeline(
            df,
            x_start="start_time", x_end="end_time",
            y="stand_id",
            color="operation_label",
            color_discrete_map=type_color_labels,
            category_orders={"stand_id": stand_order},
            hover_data={
                "visit_id": True,
                "operation_label": True,
                "aircraft_category": True,
                "pax": True,
                "stand_type": True,
                "start_time": "|%d/%m %H:%M",
                "end_time":   "|%d/%m %H:%M",
                "op_duration": True,
                "visit_start": "|%d/%m %H:%M",
                "visit_end":   "|%d/%m %H:%M",
                "ground_duration": True,
                "stand_id": False,
                "operation_type": False,
                "op_duration_min": False,
                "ground_duration_min": False,
            },
            labels={
                "operation_label": "Etapa",
                "stand_id": "Posição",
                "stand_type": "Tipo de posição",
                "aircraft_category": "Categoria",
                "pax": "PAX",
                "start_time": "Início da etapa",
                "end_time": "Fim da etapa",
                "op_duration": "Duração da etapa",
                "visit_start": "Chegada",
                "visit_end": "Partida",
                "ground_duration": "Tempo em solo",
            },
            height=max(420, len(df["stand_id"].unique()) * 24),
        )
        fig_gantt.update_layout(
            xaxis_title="Horário",
            yaxis_title="Posição",
            legend_title="Etapa",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        fig_gantt.update_xaxes(tickformat="%H:%M")
        fig_gantt.update_yaxes(type="category")
        fig_gantt.update_traces(marker_line_width=0.5)
        st.plotly_chart(fig_gantt, use_container_width=True)

    # ── Lista de voos por portão (dia) ─────────────────────────────────────

    st.subheader("Lista de voos por portão (dia selecionado)")

    day_df = alloc[alloc["start_time"].dt.date == selected_date].copy()
    if day_df.empty:
        st.info("Nenhuma operação alocada para este dia.")
    else:
        day_df["operation_label"] = day_df["operation_type"].map(
            OPERATION_LABELS).fillna(day_df["operation_type"])
        day_df["op_duration_min"] = (
            day_df["end_time"] - day_df["start_time"]).dt.total_seconds() / 60.0
        day_df["op_duration"] = day_df["op_duration_min"].apply(
            format_duration_minutes)
        day_df["inicio"] = day_df["start_time"].dt.strftime("%H:%M")
        day_df["fim"] = day_df["end_time"].dt.strftime("%H:%M")

        stands_day = sorted(
            day_df["stand_id"].unique().tolist(), key=stand_sort_key)
        selected_stand = st.selectbox(
            "Portão", options=["Todos"] + stands_day, key="list_portao")

        if selected_stand != "Todos":
            day_df = day_df[day_df["stand_id"] == selected_stand]

        list_cols = [
            "stand_id",
            "operation_label",
            "inicio",
            "fim",
            "op_duration",
            "aircraft_category",
            "pax",
            "stand_type",
            "visit_id",
        ]
        list_cols = [c for c in list_cols if c in day_df.columns]

        table = (
            day_df.sort_values(["stand_id", "start_time"]).loc[:, list_cols]
            .rename(
                columns={
                    "stand_id": "Portão",
                    "operation_label": "Etapa",
                    "inicio": "Início",
                    "fim": "Fim",
                    "op_duration": "Duração",
                    "aircraft_category": "Categoria",
                    "pax": "PAX",
                    "stand_type": "Tipo de posição",
                    "visit_id": "Visita",
                }
            )
            .reset_index(drop=True)
        )

        st.dataframe(table, use_container_width=True, height=320)

    st.divider()

    # ── Distribuições ────────────────────────────────────────────────────────

    st.subheader("Distribuições")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Operações por tipo de posição**")
        fig = px.pie(
            alloc["stand_type"].value_counts().reset_index().rename(
                columns={"stand_type": "Tipo", "count": "Qtd"}),
            values="Qtd", names="Tipo",
            color="Tipo", color_discrete_map=STAND_COLOR, hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Operações por categoria de aeronave**")
        fig = px.bar(
            alloc["aircraft_category"].value_counts().reset_index().rename(
                columns={"aircraft_category": "Categoria", "count": "Qtd"}),
            x="Categoria", y="Qtd",
            color="Categoria",
            color_discrete_sequence=px.colors.qualitative.Safe,
            text="Qtd",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(
            t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        st.markdown("**Estacionamento de longa permanência por tipo**")
        park = alloc[alloc["operation_type"] == "parking"]
        fig = px.bar(
            park["stand_type"].value_counts().reset_index().rename(
                columns={"stand_type": "Tipo", "count": "Qtd"}),
            x="Tipo", y="Qtd",
            color="Tipo", color_discrete_map=STAND_COLOR, text="Qtd",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(
            t=10, b=10), xaxis_title="", yaxis_title="")
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
        labels={"stand_id": "Posição",
                "operacoes": "Nº de operações", "stand_type": "Tipo"},
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
                alloc[["operation_id", "stand_id",
                       "start_time", "aircraft_category"]],
                on="operation_id", how="left",
            )
            .merge(
                alloc[["operation_id", "stand_id"]].rename(
                    columns={"operation_id": "successor_operation_id",
                             "stand_id": "stand_destino"}
                ),
                on="successor_operation_id", how="left",
            )
            .rename(columns={"stand_id": "stand_origem", "start_time": "horario"})
        )
        tows_detail["horario"] = pd.to_datetime(
            tows_detail["horario"]).dt.strftime("%H:%M")

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
