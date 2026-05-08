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

OPERATION_COLOR = {
    "turnaround": "#0B3D91",  # azul escuro (alto contraste)
    "arrival": "#0B6E4F",     # verde escuro
    "parking": "#374151",     # cinza escuro
    "departure": "#B45309",   # âmbar escuro
}

COMPANY_COLOR = {
    # Cores mais leves (pastel) para o preenchimento.
    "Azul": "#BFD3FF",
    "Azul Conecta": "#BFD3FF",
    "Gol": "#FFD7A8",
    "Latam": "#FFB3B3",
}

OPERATION_LABELS = {
    "turnaround": "Turnaround",
    "arrival": "Arrival",
    "parking": "Parking",
    "departure": "Departure",
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


def format_int_pt(value) -> str:
    """Formata inteiro com separador de milhar em PT-BR (ponto)."""
    if value is None or pd.isna(value):
        return ""
    try:
        number = int(round(float(value)))
    except Exception:
        return str(value)
    return f"{number:,}".replace(",", ".")


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


def normalize_company(value: str) -> str:
    text = str(value).strip()
    key = text.casefold()
    if key == "azul":
        return "Azul"
    if key == "azul conecta":
        return "Azul Conecta"
    if key == "gol":
        return "Gol"
    if key == "latam":
        return "Latam"
    return text

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
            "Turnaround mínimo (min)", min_value=40, max_value=120, value=40)
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
                    f"Objetivo: **{format_int_pt(result.objective_value)}** · "
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
    if "company" in alloc.columns:
        alloc["company"] = alloc["company"].astype(str).map(normalize_company)
    else:
        alloc["company"] = ""
    if "aircraft_category" in alloc.columns:
        alloc["aircraft_category"] = alloc["aircraft_category"].astype(str)
    if "visit_id" in alloc.columns:
        alloc["visit_id"] = alloc["visit_id"].astype(str)
    if not tows.empty and "operation_id" in tows.columns:
        tows["operation_id"] = tows["operation_id"].astype(str)

    # ── Seleção de dia (afeta KPIs, Gantt e lista do dia) ──────────────────

    all_dates = sorted(alloc["start_time"].dt.date.unique())
    if not all_dates:
        st.warning("Nenhuma data encontrada nos resultados.")
        st.stop()

    selected_date = st.selectbox(
        "Data",
        all_dates,
        key="selected_date",
    )

    alloc_day = alloc[alloc["start_time"].dt.date == selected_date].copy()

    if tows.empty:
        tows_day = tows.copy()
    else:
        tows_with_time = tows.merge(
            alloc[["operation_id", "start_time"]],
            on="operation_id",
            how="left",
        )
        tows_day = (
            tows_with_time[tows_with_time["start_time"].dt.date == selected_date]
            .drop(columns=["start_time"], errors="ignore")
            .reset_index(drop=True)
        )

    # ── KPIs + métricas (dia selecionado) ─────────────────────────────────

    if "revenue_factor" in alloc.columns:
        alloc["revenue_factor"] = pd.to_numeric(
            alloc["revenue_factor"], errors="coerce"
        ).fillna(0.0)
    else:
        # Compatibilidade com resultados antigos: reconstroi o fator de receita via tipo da posição.
        base_cfg = ModelConfig()
        stand_type_to_factor = {
            "contato": float(base_cfg.contact_revenue_factor),
            "remoto": float(base_cfg.remote_revenue_factor),
            "estacionamento": float(base_cfg.parking_revenue_factor),
        }
        alloc["revenue_factor"] = (
            alloc["stand_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(stand_type_to_factor)
            .fillna(0.0)
        )

    # Mantém o mesmo revenue_factor também na fatia do dia.
    if "revenue_factor" not in alloc_day.columns:
        alloc_day = alloc_day.merge(
            alloc[["operation_id", "revenue_factor"]],
            on="operation_id",
            how="left",
        )

    def compute_scope_metrics(scope_alloc: pd.DataFrame, scope_tows: pd.DataFrame) -> dict:
        if scope_alloc.empty:
            return {
                "ops": 0,
                "stands": 0,
                "tows": int(len(scope_tows)),
                "pax_contato": 0,
                "pax_total": 0,
                "pct_contato": 0.0,
                "walking_total": 0.0,
                "revenue_total": 0.0,
                "parking_minutes_total": 0.0,
            }

        pax_contato = int(scope_alloc[scope_alloc["is_contact"] == 1]["pax"].sum())
        pax_total = int(scope_alloc["pax"].sum())
        pct_contato = pax_contato / pax_total * 100 if pax_total > 0 else 0.0
        walking_total = float((scope_alloc["pax"] * scope_alloc["walking_distance"]).sum())
        revenue_total = float((scope_alloc["pax"] * scope_alloc["revenue_factor"]).sum())

        parking_ops = scope_alloc[scope_alloc["operation_type"] == "parking"]
        parking_minutes_total = float(
            ((parking_ops["end_time"] - parking_ops["start_time"]).dt.total_seconds() / 60.0).sum()
        )

        return {
            "ops": int(len(scope_alloc)),
            "stands": int(scope_alloc["stand_id"].nunique()),
            "tows": int(len(scope_tows)),
            "pax_contato": pax_contato,
            "pax_total": pax_total,
            "pct_contato": float(pct_contato),
            "walking_total": walking_total,
            "revenue_total": revenue_total,
            "parking_minutes_total": parking_minutes_total,
        }

    daily = compute_scope_metrics(alloc_day, tows_day)

    st.subheader("KPIs")
    st.caption(f"Data: {selected_date.strftime('%d/%m/%Y')}")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Operações alocadas", format_int_pt(daily["ops"]))
    d2.metric("Posições utilizadas", format_int_pt(daily["stands"]))
    d3.metric("Reboques", format_int_pt(daily["tows"]))
    d4.metric("PAX em contato", f"{daily['pct_contato']:.1f}%")
    d5.metric("Longa permanência (min)", format_int_pt(daily["parking_minutes_total"]))

    objective_key = st.session_state.get("objective")
    if objective_key:
        obj_label = OBJECTIVE_LABELS.get(objective_key, "Objetivo")
        obj_value_day = {
            "walking_distance": daily["walking_total"],
            "contact_share": daily["pax_contato"],
            "tow_count": daily["tows"],
            "revenue": daily["revenue_total"],
        }.get(objective_key)
        if obj_value_day is not None:
            st.caption(f"**{obj_label} (dia):** {format_int_pt(obj_value_day)}")

    def build_objective_table(scope: dict) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Função objetivo": OBJECTIVE_LABELS["walking_distance"],
                    "Sentido": "min",
                    "Valor": int(round(scope["walking_total"])),
                },
                {
                    "Função objetivo": OBJECTIVE_LABELS["contact_share"],
                    "Sentido": "max",
                    "Valor": int(scope["pax_contato"]),
                },
                {
                    "Função objetivo": OBJECTIVE_LABELS["tow_count"],
                    "Sentido": "min",
                    "Valor": int(scope["tows"]),
                },
                {
                    "Função objetivo": OBJECTIVE_LABELS["revenue"],
                    "Sentido": "max",
                    "Valor": int(round(scope["revenue_total"])),
                },
            ]
        )

    st.markdown("**Funções objetivo**")
    objective_table = build_objective_table(daily)
    objective_table["Valor"] = objective_table["Valor"].map(format_int_pt)
    st.dataframe(objective_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── Gantt ────────────────────────────────────────────────────────────────

    st.subheader("Gráfico de Gantt – Alocação por Posição")

    c1, c2 = st.columns([2, 2])

    stand_types = ["Todos"] + sorted(alloc["stand_type"].unique())
    sel_stand_type = c1.selectbox("Tipo de posição", stand_types)
    op_types = ["Todos"] + sorted(alloc["operation_type"].unique())
    sel_op_type = c2.selectbox("Tipo de operação", op_types)

    df = alloc_day.copy()
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
        key=f"hour_window_{selected_date.isoformat()}",
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

        stand_order = sorted(
            df["stand_id"].unique().tolist(), key=stand_sort_key)

        # Se o resultado não traz empresa (ex.: CSV antigo), evita gráfico "monocromático".
        has_company = df["company"].astype(str).str.strip().ne("").any()
        if has_company:
            color_col = "company"
            color_map = COMPANY_COLOR
            legend_title = "Empresa"
        else:
            st.warning(
                "Este resultado não contém a coluna de empresa (provavelmente um CSV antigo). "
                "Execute a otimização novamente para ver as cores por empresa."
            )
            color_col = "operation_label"
            color_map = {OPERATION_LABELS.get(k, k): v for k, v in OPERATION_COLOR.items()}
            legend_title = "Etapa"

        # Cor de preenchimento = empresa; cor da borda = tipo de operação.
        # (O tipo de operação também aparece no tooltip.)
        fig_gantt = px.timeline(
            df,
            x_start="start_time", x_end="end_time",
            y="stand_id",
            color=color_col,
            color_discrete_map=color_map,
            category_orders={"stand_id": stand_order},
            hover_data={
                "visit_id": True,
                "company": True,
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
                "company": "Empresa",
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
            legend_title=legend_title,
            margin=dict(l=60, r=20, t=20, b=40),
        )
        fig_gantt.update_xaxes(tickformat="%H:%M")
        fig_gantt.update_yaxes(type="category")

        # Aplica a cor de borda por tipo de operação.
        df["_op_border_color"] = (
            df["operation_type"].astype(str).str.strip().str.lower().map(OPERATION_COLOR).fillna("#000000")
        )
        for trace in fig_gantt.data:
            trace_name = str(getattr(trace, "name", ""))
            mask = df[color_col].astype(str) == trace_name
            border_colors = df.loc[mask, "_op_border_color"].tolist()
            if border_colors:
                trace.marker.line.color = border_colors
                trace.marker.line.width = 3
        st.plotly_chart(fig_gantt, use_container_width=True)

    # ── Lista de voos por portão (dia) ─────────────────────────────────────

    st.subheader("Lista de voos por portão")

    day_df = alloc_day.copy()
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
            "Portão",
            options=["Todos"] + stands_day,
            key=f"list_portao_{selected_date.isoformat()}",
        )

        if selected_stand != "Todos":
            day_df = day_df[day_df["stand_id"] == selected_stand]

        list_cols = [
            "stand_id",
            "operation_label",
            "inicio",
            "fim",
            "op_duration",
            "company",
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
                    "company": "Empresa",
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
            alloc_day["stand_type"].value_counts().reset_index().rename(
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
            alloc_day["aircraft_category"].value_counts().reset_index().rename(
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
        park = alloc_day[alloc_day["operation_type"] == "parking"]
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
        alloc_day.groupby(["stand_id", "stand_type"])
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

    st.subheader(f"Reboques ({len(tows_day)})")

    if tows_day.empty:
        st.info("Nenhum reboque neste dia.")
    else:
        tows_detail = (
            tows_day
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
