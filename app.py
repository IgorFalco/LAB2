from __future__ import annotations
from main import (
    ModelConfig,
    build_adjacent_stands,
    intervals_overlap,
    is_category_compatible,
    normalize_flights,
    normalize_positions,
    solve_airport_stand_allocation,
)

import datetime as dt
import io
import re
import sys
import tempfile
import time
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
    "holding": "#6B7280",     # sobrevoo/espera fora do portão
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
    "holding": "Sobrevoo",
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


# ── Simulação de atrasos (sem reotimizar) ──────────────────────────────────


_VISIT_ID_RE = re.compile(r"^V_(M\d{5})_(M\d{5})$")


def parse_visit_movement_ids(visit_id: str) -> tuple[str | None, str | None]:
    """Extrai (arrival_id, departure_id) do visit_id no formato 'V_M00001_M00002'."""
    text = str(visit_id).strip()
    match = _VISIT_ID_RE.match(text)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def company_key(value: str) -> str:
    return " ".join(str(value).split()).casefold()


@st.cache_data(show_spinner=False)
def load_positions_context(positions_path: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    raw = pd.read_csv(positions_path)
    base_cfg = ModelConfig()
    positions = normalize_positions(raw, base_cfg)
    adjacency = build_adjacent_stands(positions)
    positions = positions.set_index("stand_id", drop=False)
    return positions, adjacency


@st.cache_data(show_spinner=False)
def load_flights_by_movement_id(flights_path: str) -> pd.DataFrame:
    raw = pd.read_csv(flights_path)
    flights = normalize_flights(raw)
    return flights.set_index("movement_id", drop=False)


@st.cache_data(show_spinner=False)
def load_flight_dates_for_block_config(flights_path: str) -> list[dt.date]:
    """Lê as datas disponíveis no planejamento para evitar bloqueio em data errada."""
    raw = pd.read_csv(flights_path)
    flights = normalize_flights(raw)
    dates = sorted(flights["datetime"].dt.date.dropna().unique().tolist())
    return dates

def get_uploaded_flight_dates(uploaded_file) -> list[dt.date]:
    """Lê datas do voos.csv enviado via upload sem alterar o arquivo original."""
    if uploaded_file is None:
        return []
    raw = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    flights = normalize_flights(raw)
    return sorted(flights["datetime"].dt.date.dropna().unique().tolist())


def build_visit_flight_lookup(
    visit_ids: list[str],
    flights_by_id: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Retorna metadados de voo (nº e horários) por visit_id, se disponível."""
    lookup: dict[str, dict[str, str]] = {}
    if flights_by_id is None or flights_by_id.empty:
        return lookup

    for visit_id in visit_ids:
        arr_id, dep_id = parse_visit_movement_ids(visit_id)
        info: dict[str, str] = {}

        if arr_id and arr_id in flights_by_id.index:
            info["arrival_flight_number"] = str(flights_by_id.loc[arr_id, "flight_number"])
            info["arrival_time"] = str(flights_by_id.loc[arr_id, "datetime"])
            info["arrival_od"] = str(flights_by_id.loc[arr_id, "origin_destination"])

        if dep_id and dep_id in flights_by_id.index:
            info["departure_flight_number"] = str(flights_by_id.loc[dep_id, "flight_number"])
            info["departure_time"] = str(flights_by_id.loc[dep_id, "datetime"])
            info["departure_od"] = str(flights_by_id.loc[dep_id, "origin_destination"])

        if info:
            lookup[str(visit_id)] = info

    return lookup


def compute_tows_from_allocation(alloc: pd.DataFrame) -> pd.DataFrame:
    required = {"operation_id", "visit_id", "operation_type", "stand_id"}
    if alloc is None or alloc.empty or not required.issubset(set(alloc.columns)):
        return pd.DataFrame(columns=["operation_id", "visit_id", "successor_operation_id", "tow"])

    op_to_stand = (
        alloc[["operation_id", "stand_id"]]
        .astype(str)
        .drop_duplicates(subset=["operation_id"])
        .set_index("operation_id")["stand_id"]
        .to_dict()
    )

    rows: list[dict] = []
    for _, row in alloc.iterrows():
        op_type = str(row.get("operation_type", "")).strip().lower()
        visit_id = str(row.get("visit_id", "")).strip()
        op_id = str(row.get("operation_id", "")).strip()

        if op_type == "arrival":
            succ = f"{visit_id}_PARK"
        elif op_type == "parking":
            succ = f"{visit_id}_DEP"
        else:
            continue

        if succ in op_to_stand and op_to_stand.get(op_id) != op_to_stand.get(succ):
            rows.append(
                {
                    "operation_id": op_id,
                    "visit_id": visit_id,
                    "successor_operation_id": succ,
                    "tow": 1,
                }
            )

    return pd.DataFrame(rows)


def compatible_stands_for_operation(
    op_row: pd.Series,
    positions: pd.DataFrame,
    config: ModelConfig,
) -> list[str]:
    op_cat = str(op_row.get("aircraft_category", "")).strip().upper()
    allow_parking_only = str(op_row.get("operation_type", "")).strip().lower() == "parking"
    company = str(op_row.get("company", "")).strip()
    company_k = company_key(company)

    de_allowed = {str(s).strip() for s in (config.de_allowed_stands or ())}
    azul_only_company_keys = {
        " ".join(str(c).split()).casefold() for c in (config.azul_only_companies or ())
    }

    candidates: list[str] = []
    for stand_id, stand in positions.iterrows():
        stand_id = str(stand_id)

        if stand_id in (config.azul_only_stands or ()) and company_k not in azul_only_company_keys:
            continue

        if op_cat in {"D", "E"} and de_allowed and stand_id not in de_allowed:
            continue

        allowed_categories = stand.get("allowed_categories")
        if not is_category_compatible(op_cat, allowed_categories):
            continue

        stand_is_parking_only = bool(stand.get("is_parking_only", 0))
        if not allow_parking_only and stand_is_parking_only:
            continue

        candidates.append(stand_id)

    return candidates


def is_feasible_assignment(
    alloc: pd.DataFrame,
    op_id: str,
    op_start: pd.Timestamp,
    op_end: pd.Timestamp,
    op_cat: str,
    stand_id: str,
    adjacency: dict[str, list[str]],
    buffer_minutes: int,
) -> bool:
    if alloc is None or alloc.empty:
        return True

    buffer = pd.Timedelta(minutes=int(buffer_minutes))
    op_id = str(op_id)
    stand_id = str(stand_id)
    op_cat = str(op_cat).strip().upper()

    others = alloc[alloc["operation_id"].astype(str) != op_id]

    # Conflito temporal na mesma posição.
    same_stand = others[others["stand_id"].astype(str) == stand_id]
    if not same_stand.empty:
        time_conflict = same_stand[
            (op_start < same_stand["end_time"] + buffer)
            & (same_stand["start_time"] < op_end + buffer)
        ]
        if not time_conflict.empty:
            return False

    # Conflito por adjacência envolvendo D/E.
    overlapping = others[
        (op_start < others["end_time"] + buffer)
        & (others["start_time"] < op_end + buffer)
    ]
    if overlapping.empty:
        return True

    if op_cat in {"D", "E"}:
        adj_list = adjacency.get(stand_id, [])
        if adj_list and overlapping["stand_id"].astype(str).isin(adj_list).any():
            return False

    blocking = overlapping[
        overlapping["aircraft_category"].astype(str).str.strip().str.upper().isin(["D", "E"])
    ]
    if not blocking.empty:
        for blocking_stand in blocking["stand_id"].astype(str).unique().tolist():
            if stand_id in adjacency.get(str(blocking_stand), []):
                return False

    return True


def choose_feasible_stand(
    op_row: pd.Series,
    alloc: pd.DataFrame,
    positions: pd.DataFrame,
    adjacency: dict[str, list[str]],
    config: ModelConfig,
    buffer_minutes: int,
    blocked_stands: list[str] | None = None,
) -> str | None:
    candidates = compatible_stands_for_operation(op_row, positions, config)
    if not candidates:
        return None

    # Exclui portões bloqueados por manutenção
    if blocked_stands:
        candidates = [s for s in candidates if str(s) not in blocked_stands]
    if not candidates:
        return None

    op_type = str(op_row.get("operation_type", "")).strip().lower()

    def sort_key(stand_id: str):
        stand_id = str(stand_id)
        walking = float(positions.loc[stand_id, "walking_distance"]) if stand_id in positions.index else 1e9
        if op_type == "parking":
            is_parking_only = int(positions.loc[stand_id, "is_parking_only"]) if stand_id in positions.index else 0
            return (0 if is_parking_only == 1 else 1, walking)
        return (walking,)

    for stand_id in sorted(candidates, key=sort_key):
        if is_feasible_assignment(
            alloc=alloc,
            op_id=str(op_row.get("operation_id")),
            op_start=op_row.get("start_time"),
            op_end=op_row.get("end_time"),
            op_cat=str(op_row.get("aircraft_category")),
            stand_id=stand_id,
            adjacency=adjacency,
            buffer_minutes=buffer_minutes,
        ):
            return stand_id

    return None


def update_operation_stand_inplace(
    alloc: pd.DataFrame,
    op_id: str,
    new_stand: str,
    positions: pd.DataFrame,
) -> None:
    mask = alloc["operation_id"].astype(str) == str(op_id)
    alloc.loc[mask, "stand_id"] = str(new_stand)

    if str(new_stand) not in positions.index:
        return

    for col in ["stand_type", "walking_distance", "revenue_factor", "is_contact"]:
        if col in alloc.columns and col in positions.columns:
            alloc.loc[mask, col] = positions.loc[str(new_stand), col]


def apply_departure_delay_and_reassign(
    alloc: pd.DataFrame,
    visit_id: str,
    delay_minutes: int,
    positions: pd.DataFrame,
    adjacency: dict[str, list[str]],
    config: ModelConfig,
    blocked_stands: list[str] | None = None,
) -> dict:
    """Aplica atraso na partida (em minutos) e realoca operações que entrarem em conflito.
    Portões listados em `blocked_stands` são tratados como indisponíveis."""
    delay_minutes = int(delay_minutes)
    if delay_minutes <= 0:
        return {"changed_ops": [], "moved_ops": []}

    visit_id = str(visit_id)
    delta = pd.Timedelta(minutes=delay_minutes)

    mask_visit = alloc["visit_id"].astype(str) == visit_id
    if not mask_visit.any():
        return {"changed_ops": [], "moved_ops": []}

    # Atraso na PARTIDA:
    # - turnaround: estende o fim
    # - parking: estende o fim (mais tempo estacionado)
    # - departure: desloca início e fim (mantém duração da etapa)
    mask_turn = mask_visit & (alloc["operation_type"].astype(str).str.strip().str.lower() == "turnaround")
    alloc.loc[mask_turn, "end_time"] = alloc.loc[mask_turn, "end_time"] + delta

    mask_park = mask_visit & (alloc["operation_type"].astype(str).str.strip().str.lower() == "parking")
    alloc.loc[mask_park, "end_time"] = alloc.loc[mask_park, "end_time"] + delta

    mask_dep = mask_visit & (alloc["operation_type"].astype(str).str.strip().str.lower() == "departure")
    alloc.loc[mask_dep, "start_time"] = alloc.loc[mask_dep, "start_time"] + delta
    alloc.loc[mask_dep, "end_time"] = alloc.loc[mask_dep, "end_time"] + delta

    changed_ops = alloc.loc[
        mask_visit
        & alloc["operation_type"].astype(str).str.strip().str.lower().isin(["turnaround", "parking", "departure"]),
        "operation_id",
    ].astype(str).tolist()

    moved_ops: list[dict] = []
    buffer_minutes = int(config.turnaround_buffer_minutes)

    # Repara conflitos movendo só as operações afetadas, sem reotimizar.
    for op_id in changed_ops:
        op_row = alloc.loc[alloc["operation_id"].astype(str) == str(op_id)].iloc[0]
        current_stand = str(op_row.get("stand_id", ""))
        op_cat = str(op_row.get("aircraft_category", "")).strip().upper()

        if current_stand and str(current_stand) not in (blocked_stands or []) and is_feasible_assignment(
            alloc=alloc,
            op_id=str(op_id),
            op_start=op_row.get("start_time"),
            op_end=op_row.get("end_time"),
            op_cat=op_cat,
            stand_id=current_stand,
            adjacency=adjacency,
            buffer_minutes=buffer_minutes,
        ):
            continue

        new_stand = choose_feasible_stand(
            op_row=op_row,
            alloc=alloc,
            positions=positions,
            adjacency=adjacency,
            config=config,
            buffer_minutes=buffer_minutes,
            blocked_stands=blocked_stands,
        )
        if new_stand is None:
            raise ValueError(
                f"Não há portão compatível disponível para realocar {op_id} (visita {visit_id})."
            )

        if new_stand != current_stand:
            update_operation_stand_inplace(alloc, op_id=op_id, new_stand=new_stand, positions=positions)
            moved_ops.append(
                {
                    "operation_id": str(op_id),
                    "operation_type": str(op_row.get("operation_type", "")),
                    "stand_from": current_stand,
                    "stand_to": str(new_stand),
                }
            )

    return {"changed_ops": changed_ops, "moved_ops": moved_ops}


def apply_gate_block(
    alloc: pd.DataFrame,
    blocked_stand: str,
    block_date,
    positions: pd.DataFrame,
    adjacency: dict[str, list[str]],
    config: ModelConfig,
) -> dict:
    """Bloqueia um portão para manutenção num dado dia.

    - Marca o portão como bloqueado no session_state (chamador é responsável).
    - Realoca todas as operações já alocadas naquele portão naquele dia para o
      portão compatível mais próximo (menor walking_distance), sem reotimizar.
    """
    blocked_stand = str(blocked_stand)
    buffer_minutes = int(config.turnaround_buffer_minutes)

    # Operações alocadas no portão bloqueado, no dia em questão
    mask = (
        (alloc["stand_id"].astype(str) == blocked_stand)
        & (alloc["start_time"].dt.date == block_date)
    )
    ops_to_move = alloc.loc[mask, "operation_id"].astype(str).tolist()

    moved_ops: list[dict] = []
    for op_id in ops_to_move:
        op_row = alloc.loc[alloc["operation_id"].astype(str) == op_id].iloc[0]

        # Encontra portão compatível excluindo o bloqueado
        candidates = compatible_stands_for_operation(op_row, positions, config)
        candidates = [s for s in candidates if str(s) != blocked_stand]

        op_type = str(op_row.get("operation_type", "")).strip().lower()

        def sort_key(stand_id: str):
            stand_id = str(stand_id)
            walking = float(positions.loc[stand_id, "walking_distance"]) if stand_id in positions.index else 1e9
            if op_type == "parking":
                is_parking_only = int(positions.loc[stand_id, "is_parking_only"]) if stand_id in positions.index else 0
                return (0 if is_parking_only == 1 else 1, walking)
            return (walking,)

        new_stand = None
        for s in sorted(candidates, key=sort_key):
            if is_feasible_assignment(
                alloc=alloc,
                op_id=op_id,
                op_start=op_row.get("start_time"),
                op_end=op_row.get("end_time"),
                op_cat=str(op_row.get("aircraft_category", "")).strip().upper(),
                stand_id=s,
                adjacency=adjacency,
                buffer_minutes=buffer_minutes,
            ):
                new_stand = s
                break

        if new_stand is None:
            raise ValueError(
                f"Não há portão compatível disponível para realocar a operação {op_id} "
                f"que estava no portão {blocked_stand}."
            )

        update_operation_stand_inplace(alloc, op_id=op_id, new_stand=new_stand, positions=positions)
        moved_ops.append(
            {
                "operation_id": op_id,
                "operation_type": str(op_row.get("operation_type", "")),
                "stand_from": blocked_stand,
                "stand_to": str(new_stand),
            }
        )

    return {"moved_ops": moved_ops, "total": len(moved_ops)}


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

        st.markdown("---")
        operational_block_spacing_minutes = st.number_input(
            "Espaçamento entre movimentos ajustados após restrição de pista (min)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            key="operational_block_spacing_minutes",
            help=(
                "Quando múltiplos voos são empurrados para depois do fim do bloqueio, "
                "este intervalo separa os horários ajustados para evitar conflitos."
            ),
        )

    st.divider()
    st.subheader("Restrição de janela de tempo – Pista")
    st.markdown(
        """
        Defina intervalos em que o aeroporto **não permitirá novos pousos nem decolagens**.

        - **Avião já em solo:** permanece normalmente e só pode partir após o fim do bloqueio.
          _Exemplo: avião chegou às 9h com partida às 11h e bloqueio das 10h às 12h → permanece até as 12h._
        - **Avião que chegaria durante o bloqueio:** fica em sobrevoo e pousa ao final do intervalo.
          _Exemplo: avião que chegaria às 11h com bloqueio das 10h às 12h → pousa às 12h._

        Deixe sem bloqueios para rodar a otimização sem restrição de horário na pista.
        """
    )

    if "operational_blocks_ui" not in st.session_state:
        st.session_state["operational_blocks_ui"] = []

    # Datas reais existentes no planejamento. O erro mais comum era adicionar o
    # bloqueio na data de hoje, enquanto o voo exibido era de outra data
    # (ex.: 06/03/2026). Assim o pré-processamento não encontrava nada para ajustar.
    try:
        if use_default:
            _flight_dates_for_blocks = load_flight_dates_for_block_config(str(BASE_DIR / "voos.csv"))
        else:
            _flight_dates_for_blocks = get_uploaded_flight_dates(up_voos)
    except Exception:
        _flight_dates_for_blocks = []

    if _flight_dates_for_blocks:
        st.caption(
            "Datas disponíveis no planejamento: "
            + ", ".join(d.strftime("%d/%m/%Y") for d in _flight_dates_for_blocks[:10])
            + ("..." if len(_flight_dates_for_blocks) > 10 else "")
        )
    else:
        st.warning(
            "Não consegui ler as datas do voos.csv. Confira se o bloqueio está na mesma data dos voos."
        )

    with st.form("operational_block_form", clear_on_submit=True):
        c_b1, c_b2, c_b3, c_b4 = st.columns([1.2, 1, 1, 2])

        if _flight_dates_for_blocks:
            block_date = c_b1.selectbox(
                "Data do bloqueio",
                options=_flight_dates_for_blocks,
                format_func=lambda d: d.strftime("%d/%m/%Y"),
                key="block_date_select",
            )
        else:
            block_date = c_b1.date_input("Data do bloqueio", value=dt.date.today(), key="block_date")

        block_start = c_b2.time_input("Início", value=dt.time(9, 0), key="block_start")
        block_end = c_b3.time_input("Fim", value=dt.time(10, 45), key="block_end")
        block_reason = c_b4.text_input("Motivo", value="", key="block_reason")

        if st.form_submit_button("Adicionar restrição"):
            if block_end <= block_start:
                st.error("O horário de fim deve ser posterior ao horário de início.")
            else:
                block_item = {
                    "date": block_date.isoformat(),
                    "start_time": block_start.strftime("%H:%M"),
                    "end_time": block_end.strftime("%H:%M"),
                    "reason": str(block_reason).strip(),
                }
                st.session_state["operational_blocks_ui"].append(block_item)
                st.success(
                    f"Restrição adicionada: {block_date.strftime('%d/%m/%Y')} "
                    f"das {block_start.strftime('%H:%M')} às {block_end.strftime('%H:%M')}."
                )

    blocks_now = list(st.session_state.get("operational_blocks_ui") or [])
    if blocks_now:
        st.dataframe(
            pd.DataFrame(blocks_now).rename(
                columns={
                    "date": "Data",
                    "start_time": "Início",
                    "end_time": "Fim",
                    "reason": "Motivo",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Remover todas as restrições"):
            st.session_state["operational_blocks_ui"] = []
            st.rerun()
    else:
        st.info("Nenhuma restrição de pista configurada. A otimização será executada sem restrição de horário.")

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
            operational_blocks=list(st.session_state.get("operational_blocks_ui") or []),
            operational_block_spacing_minutes=int(operational_block_spacing_minutes),
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
                st.session_state["input_paths"] = {
                    k: (str(v) if v is not None else None) for k, v in paths.items()
                }
                st.session_state["solution_id"] = f"run:{time.time_ns()}"
                st.session_state["operational_blocks"] = list(config.operational_blocks or [])
                st.session_state["operational_block_spacing_minutes_used"] = int(config.operational_block_spacing_minutes)

                if config.operational_blocks:
                    st.info(
                        "Restrições de pista aplicadas: "
                        + "; ".join(
                            f"{b.get('date')} {b.get('start_time')}–{b.get('end_time')}"
                            + (f" ({b.get('reason')})" if b.get('reason') else "")
                            for b in config.operational_blocks
                        )
                    )
                else:
                    st.info("Otimização executada sem restrição de janela de tempo na pista.")

                # Se já havia simulação de atraso na aba Resultados, reseta para o novo resultado.
                for key in ("alloc_base", "alloc_current", "delay_events", "_active_solution_id"):
                    st.session_state.pop(key, None)

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

    # Carrega resultado (memória ou disco) e mantém um estado "corrente" para
    # simular atrasos sem rodar o otimizador novamente.
    if "alloc" in st.session_state:
        alloc_loaded = st.session_state["alloc"].copy()
        current_solution_id = st.session_state.get("solution_id") or "run:unknown"
    else:
        alloc_path = BASE_DIR / "outputs" / "alocacao_resultado.csv"
        if alloc_path.exists():
            alloc_loaded = pd.read_csv(alloc_path)
            current_solution_id = f"disk:{alloc_path.stat().st_mtime_ns}:{alloc_path.stat().st_size}"
            st.info("Exibindo resultados do último arquivo salvo em disco.")
        else:
            st.warning(
                "Nenhum resultado disponível. Execute a otimização na aba **Configuração**.")
            st.stop()

    alloc_loaded["start_time"] = pd.to_datetime(alloc_loaded["start_time"])
    alloc_loaded["end_time"] = pd.to_datetime(alloc_loaded["end_time"])

    if st.session_state.get("_active_solution_id") != current_solution_id:
        st.session_state["_active_solution_id"] = current_solution_id
        st.session_state["alloc_base"] = alloc_loaded.copy()
        st.session_state["alloc_current"] = alloc_loaded.copy()
        st.session_state["delay_events"] = []

    alloc = st.session_state["alloc_current"]

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
    # ── Seleção de dia (afeta KPIs, Gantt e lista do dia) ──────────────────

    all_dates = sorted(alloc["start_time"].dt.date.unique())
    if not all_dates:
        st.warning("Nenhuma data encontrada nos resultados.")
        st.stop()

    if "selected_date" in st.session_state and st.session_state["selected_date"] not in all_dates:
        st.session_state.pop("selected_date", None)

    selected_date = st.selectbox(
        "Data",
        all_dates,
        key="selected_date",
    )

    alloc_day = alloc[alloc["start_time"].dt.date == selected_date].copy()

    # ── Bloqueio de portão por manutenção ─────────────────────────────────

    st.subheader("🔒 Bloqueio de portão por manutenção")
    st.caption(
        "Selecione um portão e marque-o como bloqueado para manutenção no dia selecionado. "
        "Todos os voos já alocados nele serão realocados para o portão compatível mais próximo. "
        "Voos atrasados também não poderão ser direcionados ao portão bloqueado."
    )

    # Inicializa conjunto de portões bloqueados no session_state
    if "blocked_gates" not in st.session_state:
        st.session_state["blocked_gates"] = {}  # {"YYYY-MM-DD": ["stand_id", ...]}

    _date_key = selected_date.isoformat()
    _blocked_today = st.session_state["blocked_gates"].get(_date_key, [])

    _all_stands_day = sorted(
        alloc_day["stand_id"].astype(str).unique().tolist(), key=stand_sort_key
    ) if not alloc_day.empty else []

    _available_to_block = [s for s in _all_stands_day if s not in _blocked_today]

    _col_b1, _col_b2 = st.columns([3, 1])
    _stand_to_block = _col_b1.selectbox(
        "Portão para bloquear",
        options=_available_to_block if _available_to_block else [""],
        key=f"block_stand_{_date_key}",
        disabled=not _available_to_block,
    )
    _do_block = _col_b2.button(
        "🔒 Bloquear portão",
        type="primary",
        use_container_width=True,
        key=f"do_block_{_date_key}",
        disabled=not _available_to_block or not _stand_to_block,
    )

    # Carrega posições/adjacência para uso no bloqueio e na simulação de atraso
    _input_paths_ui = st.session_state.get("input_paths") or {}
    _pos_path_ui = _input_paths_ui.get("positions") or str(BASE_DIR / "posicoes.csv")
    if Path(_pos_path_ui).exists():
        _positions_norm_ui, _adjacency_ui = load_positions_context(str(_pos_path_ui))
    else:
        _positions_norm_ui, _adjacency_ui = None, {}

    if _do_block and _stand_to_block:
        try:
            if _positions_norm_ui is None:
                st.error("Arquivo posicoes.csv não encontrado. Não é possível realocar.")
            else:
                _bsummary = apply_gate_block(
                    alloc=alloc,
                    blocked_stand=_stand_to_block,
                    block_date=selected_date,
                    positions=_positions_norm_ui,
                    adjacency=_adjacency_ui,
                    config=ModelConfig(),
                )
                # Registra portão bloqueado
                _blocked_today_new = list(_blocked_today) + [str(_stand_to_block)]
                st.session_state["blocked_gates"][_date_key] = _blocked_today_new

                _moved = _bsummary.get("moved_ops", [])
                if _moved:
                    _parts = []
                    for _item in _moved:
                        _ot = OPERATION_LABELS.get(str(_item.get("operation_type", "")), str(_item.get("operation_type", "")))
                        _parts.append(f"{_ot}: {_item.get('stand_from')}→{_item.get('stand_to')}")
                    st.success(
                        f"Portão **{_stand_to_block}** bloqueado para manutenção. "
                        f"{_bsummary['total']} operação(ões) realocada(s): {'; '.join(_parts)}."
                    )
                else:
                    st.success(
                        f"Portão **{_stand_to_block}** bloqueado. Nenhuma operação precisou ser realocada."
                    )
        except Exception as _be:
            st.error(f"Erro ao bloquear portão: {_be}")

    # Exibe portões já bloqueados no dia
    _blocked_now = st.session_state["blocked_gates"].get(_date_key, [])
    if _blocked_now:
        st.markdown(
            f"**Portões bloqueados hoje ({selected_date.strftime('%d/%m/%Y')}):** "
            + ", ".join([f"🔒 {s}" for s in _blocked_now])
        )
        _col_ub1, _ = st.columns([2, 4])
        _unblock_stand = _col_ub1.selectbox(
            "Desbloquear portão",
            options=[""] + _blocked_now,
            key=f"unblock_stand_{_date_key}",
        )
        if _unblock_stand:
            if st.button(
                f"🔓 Desbloquear {_unblock_stand}",
                key=f"do_unblock_{_date_key}_{_unblock_stand}",
            ):
                _new_list = [s for s in _blocked_now if s != _unblock_stand]
                st.session_state["blocked_gates"][_date_key] = _new_list
                st.info(f"Portão {_unblock_stand} desbloqueado. As realocações anteriores permanecem ativas; reotimize se necessário.")
                st.rerun()

    st.divider()

    # ── Simulação de atraso (somente lista do dia) ─────────────────────────

    st.subheader("Simular atraso")
    st.caption(
        "Atraso aplicado na partida (estende a permanência em solo). "
        "Se houver conflito de portão, o voo é realocado automaticamente para um portão compatível disponível."
    )

    input_paths = st.session_state.get("input_paths") or {}
    positions_path = input_paths.get("positions") or str(BASE_DIR / "posicoes.csv")
    flights_path = input_paths.get("flights") or str(BASE_DIR / "voos.csv")

    positions_ok = Path(positions_path).exists()
    if not positions_ok:
        st.warning(
            "Não foi possível carregar o arquivo de posições (posicoes.csv). "
            "A simulação de atraso/realocação depende dele."
        )
    elif alloc_day.empty:
        st.info("Nenhuma operação no dia selecionado.")
    else:
        base_cfg = ModelConfig()
        positions_norm, adjacency = load_positions_context(str(positions_path))

        flights_by_id = pd.DataFrame()
        if Path(flights_path).exists():
            try:
                flights_by_id = load_flights_by_movement_id(str(flights_path))
            except Exception:
                flights_by_id = pd.DataFrame()

        visit_ids_day = sorted(alloc_day["visit_id"].astype(str).unique().tolist())
        visit_flights = build_visit_flight_lookup(visit_ids_day, flights_by_id)

        gate_rows = alloc_day[
            alloc_day["operation_type"].astype(str).str.strip().str.lower().isin(["turnaround", "departure"])
        ]
        gate_by_visit = (
            gate_rows.sort_values("start_time")
            .groupby("visit_id")["stand_id"]
            .first()
            .astype(str)
            .to_dict()
        )

        visits_day = (
            alloc_day.groupby("visit_id")
            .agg(
                company=("company", "first"),
                aircraft_category=("aircraft_category", "first"),
                visit_start=("start_time", "min"),
                visit_end=("end_time", "max"),
            )
            .reset_index()
            .sort_values("visit_start")
            .reset_index(drop=True)
        )
        visits_day["gate_stand"] = visits_day["visit_id"].map(gate_by_visit).fillna("")
        visits_day["dep_flight"] = visits_day["visit_id"].map(
            lambda vid: visit_flights.get(str(vid), {}).get("departure_flight_number", "")
        )
        visits_day["arr_flight"] = visits_day["visit_id"].map(
            lambda vid: visit_flights.get(str(vid), {}).get("arrival_flight_number", "")
        )

        visit_labels: dict[str, str] = {}
        for _, r in visits_day.iterrows():
            vid = str(r["visit_id"])
            dep = str(r.get("dep_flight") or "").strip()
            arr = str(r.get("arr_flight") or "").strip()
            comp = str(r.get("company") or "").strip()
            stand = str(r.get("gate_stand") or "").strip()
            start = r.get("visit_start")
            end = r.get("visit_end")

            flight_part = vid
            if arr and dep:
                flight_part = f"{comp} · {arr}→{dep}"
            elif dep:
                flight_part = f"{comp} · DEP {dep}"
            elif comp:
                flight_part = f"{comp} · {vid}"

            if pd.notna(start) and pd.notna(end):
                time_part = f"{pd.to_datetime(start).strftime('%H:%M')}–{pd.to_datetime(end).strftime('%H:%M')}"
            else:
                time_part = ""

            stand_part = f"Portão {stand}" if stand else ""
            extra = " · ".join([p for p in [time_part, stand_part, vid] if p])
            visit_labels[vid] = f"{flight_part} · {extra}" if extra else flight_part

        col_d1, col_d2, col_d3 = st.columns([3, 1, 1])
        selected_visit_id = col_d1.selectbox(
            "Voo/visita (somente do dia)",
            options=visits_day["visit_id"].astype(str).tolist(),
            format_func=lambda vid: visit_labels.get(str(vid), str(vid)),
            key=f"delay_visit_{selected_date.isoformat()}",
        )
        delay_minutes = col_d2.number_input(
            "Atraso (min)",
            min_value=0,
            max_value=600,
            value=15,
            step=5,
            key=f"delay_minutes_{selected_date.isoformat()}",
        )
        apply_delay = col_d3.button(
            "Aplicar atraso",
            type="primary",
            use_container_width=True,
            key=f"apply_delay_{selected_date.isoformat()}",
        )

        if apply_delay:
            try:
                _blocked_for_delay = st.session_state.get("blocked_gates", {}).get(selected_date.isoformat(), [])
                summary = apply_departure_delay_and_reassign(
                    alloc=alloc,
                    visit_id=str(selected_visit_id),
                    delay_minutes=int(delay_minutes),
                    positions=positions_norm,
                    adjacency=adjacency,
                    config=base_cfg,
                    blocked_stands=_blocked_for_delay if _blocked_for_delay else None,
                )

                moved = summary.get("moved_ops", []) or []
                moved_summary = ""
                if moved:
                    parts = []
                    for item in moved:
                        op_t = str(item.get("operation_type", "")).strip()
                        op_t = OPERATION_LABELS.get(op_t, op_t)
                        parts.append(f"{op_t}: {item.get('stand_from')}→{item.get('stand_to')}")
                    moved_summary = "; ".join(parts)

                dep_flight = visit_flights.get(str(selected_visit_id), {}).get("departure_flight_number", "")
                company = (
                    alloc.loc[alloc["visit_id"].astype(str) == str(selected_visit_id), "company"].astype(str).head(1).tolist()
                    or [""]
                )[0]

                st.session_state.setdefault("delay_events", []).append(
                    {
                        "date": selected_date.isoformat(),
                        "visit_id": str(selected_visit_id),
                        "company": str(company),
                        "departure_flight": str(dep_flight),
                        "delay_minutes": int(delay_minutes),
                        "moved": moved_summary,
                    }
                )

                if moved_summary:
                    st.success(f"Atraso aplicado. Realocação: {moved_summary}.")
                else:
                    st.success("Atraso aplicado. Nenhuma realocação foi necessária.")

            except Exception as e:
                st.error(f"Não foi possível aplicar o atraso: {e}")

        # Recarrega a fatia do dia após possíveis mudanças.
        alloc_day = alloc[alloc["start_time"].dt.date == selected_date].copy()

        events_day = [
            e for e in (st.session_state.get("delay_events") or [])
            if str(e.get("date")) == selected_date.isoformat()
        ]
        if events_day:
            st.markdown("**Voos com atraso (dia selecionado)**")
            events_table = pd.DataFrame(
                [
                    {
                        "Voo (partida)": e.get("departure_flight", ""),
                        "Empresa": e.get("company", ""),
                        "Visita": e.get("visit_id", ""),
                        "Atraso (min)": e.get("delay_minutes", 0),
                        "Realocação": e.get("moved", ""),
                    }
                    for e in events_day
                ]
            )
            st.dataframe(events_table, use_container_width=True, hide_index=True)

    # ── Reboques (recalculado a partir da alocação atual) ─────────────────

    tows = compute_tows_from_allocation(alloc)
    if not tows.empty:
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
    else:
        tows_day = tows.copy()

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
