from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# Configuração
# ============================================================


@dataclass
class ModelConfig:
    """Parâmetros principais do modelo e do pré-processamento."""

    # Emparelhamento chegada-partida para reconstruir visitas
    min_turnaround_minutes: int = 30
    max_turnaround_minutes: int = 8 * 60

    # Regra para permitir reboque / divisão em 3 operações
    tow_threshold_minutes: int = 180
    disembark_minutes: int = 45
    embark_minutes: int = 45
    turnaround_buffer_minutes: int = 15

    # Valores heurísticos usados quando o dado não existe no CSV
    contact_base_distance: float = 120.0
    remote_base_distance: float = 350.0
    parking_base_distance: float = 500.0
    patio_distance_step: float = 20.0
    stand_distance_step: float = 5.0
    contact_revenue_factor: float = 12.0
    remote_revenue_factor: float = 8.0
    parking_revenue_factor: float = 0.0

    # Objetivo
    objective: str = "walking_distance"  # walking_distance | contact_share | tow_count | revenue
    time_limit_seconds: Optional[int] = None
    mip_gap: Optional[float] = None
    verbose: bool = True

    # Ajustes manuais para códigos ausentes no CSV de categorias
    aircraft_category_overrides: Dict[str, str] = field(
        default_factory=lambda: {
            "7M8": "C",
            "735": "C",
            "E92": "C",
            "789": "E",
        }
    )

    # Mantido para compatibilidade; agora também geramos adjacência automática
    overlapping_stands: Dict[str, List[str]] = field(default_factory=dict)


# ============================================================
# Leitura e preparação dos dados
# ============================================================


@dataclass
class ProblemData:
    flights_raw: pd.DataFrame
    positions: pd.DataFrame
    visits: pd.DataFrame
    operations: pd.DataFrame
    compatible_stands: Dict[str, List[str]]
    overlapping_ops: Dict[str, List[str]]
    overlapping_stands: Dict[str, List[str]]
    adjacent_stands: Dict[str, List[str]]


def read_input_data(
    flights_path: str | Path,
    positions_path: str | Path,
    aircraft_categories_path: str | Path,
    aircraft_specs_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    flights = pd.read_csv(flights_path)
    positions = pd.read_csv(positions_path)
    aircraft_categories = pd.read_csv(aircraft_categories_path)

    aircraft_specs = None
    if aircraft_specs_path is not None and Path(aircraft_specs_path).exists():
        aircraft_specs = pd.read_csv(aircraft_specs_path, sep=";")

    return flights, positions, aircraft_categories, aircraft_specs


def normalize_flights(flights: pd.DataFrame) -> pd.DataFrame:
    df = flights.copy()
    df.columns = [c.strip() for c in df.columns]

    df["movement_type"] = df["Chegada Partida"].astype(str).str.strip().str.lower()
    df["date"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df["datetime"] = pd.to_datetime(
        df["Data"].astype(str) + " " + df["Horário (Hora Local)"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df["company"] = df["Empresa"].astype(str).str.strip()
    df["flight_number"] = df["Voo"].astype(str).str.strip()
    df["aircraft_code"] = df["Aeronave"].astype(str).str.strip()
    df["seats"] = pd.to_numeric(df["Assentos"], errors="coerce").fillna(0).astype(int)
    df["origin_destination"] = df["Origem Destino"].astype(str).str.strip()

    df = df.dropna(subset=["datetime", "date"]).sort_values("datetime").reset_index(drop=True)
    df["movement_id"] = [f"M{idx:05d}" for idx in range(len(df))]
    return df


def build_aircraft_category_map(
    aircraft_categories: pd.DataFrame,
    config: ModelConfig,
) -> Dict[str, str]:
    mapping = {
        str(row["Aeronave"]).strip(): str(row["Categoria"]).strip()
        for _, row in aircraft_categories.iterrows()
    }
    mapping.update(config.aircraft_category_overrides)
    return mapping


def normalize_stand_type(raw_type: str) -> str:
    value = str(raw_type).strip().lower()

    mapping = {
        "contato": "contato",
        "rcmoto": "remoto",
        "cstacionamcnto": "estacionamento",
    }

    if value in mapping:
        return mapping[value]

    return value


def parse_allowed_categories(raw_value: str) -> set[str]:
    """
    Regras do CSV de posições:
    - A pode parar em qualquer posição
    - C  -> posição aceita A e C
    - DE -> posição aceita somente D e E
    """
    value = str(raw_value).strip().upper()

    if value == "DE":
        return {"D", "E"}

    if value == "C":
        return {"A", "C"}

    if value == "D":
        return {"A", "D"}

    if value == "E":
        return {"A", "E"}

    if value == "A":
        return {"A"}

    # fallback genérico
    return set(value) if value else set()


def normalize_positions(positions: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    df = positions.copy()
    df.columns = [c.strip() for c in df.columns]

    df["stand_id"] = df["Posicao"].astype(str).str.strip()
    df["stand_number"] = pd.to_numeric(df["Posicao"], errors="coerce")
    df["stand_type"] = df["Tipo"].apply(normalize_stand_type)
    df["patio"] = pd.to_numeric(df["Patio"], errors="coerce").fillna(0).astype(int)
    df["aircraft_category"] = df["Aeronave"].astype(str).str.strip().str.upper()
    df["allowed_categories"] = df["aircraft_category"].apply(parse_allowed_categories)

    df = df.sort_values(["patio", "stand_number"]).reset_index(drop=True)

    walking_distances: List[float] = []
    revenues: List[float] = []

    for idx, row in df.iterrows():
        stand_type = row["stand_type"]
        patio_adjustment = (row["patio"] - 1) * config.patio_distance_step
        order_adjustment = idx * config.stand_distance_step

        if stand_type == "contato":
            walking_distance = config.contact_base_distance + patio_adjustment + order_adjustment
            revenue_factor = config.contact_revenue_factor
        elif stand_type == "remoto":
            walking_distance = config.remote_base_distance + patio_adjustment + order_adjustment
            revenue_factor = config.remote_revenue_factor
        else:
            walking_distance = config.parking_base_distance + patio_adjustment + order_adjustment
            revenue_factor = config.parking_revenue_factor

        walking_distances.append(walking_distance)
        revenues.append(revenue_factor)

    df["walking_distance"] = walking_distances
    df["revenue_factor"] = revenues
    df["is_contact"] = (df["stand_type"] == "contato").astype(int)
    df["is_parking_only"] = (df["stand_type"] == "estacionamento").astype(int)
    return df


def build_adjacent_stands(positions: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Considera adjacentes posições consecutivas dentro do mesmo pátio.
    Exemplo: 123 adjacente a 122 e 124.
    """
    adjacency: Dict[str, set[str]] = {sid: set() for sid in positions["stand_id"]}

    for patio, group in positions.groupby("patio"):
        group = group.sort_values("stand_number")
        rows = group.to_dict("records")

        for i in range(len(rows) - 1):
            current_row = rows[i]
            next_row = rows[i + 1]

            if pd.isna(current_row["stand_number"]) or pd.isna(next_row["stand_number"]):
                continue

            if int(next_row["stand_number"]) - int(current_row["stand_number"]) == 1:
                a = current_row["stand_id"]
                b = next_row["stand_id"]
                adjacency[a].add(b)
                adjacency[b].add(a)

    return {k: sorted(v) for k, v in adjacency.items()}


def reconstruct_visits(
    flights: pd.DataFrame,
    aircraft_category_map: Dict[str, str],
    config: ModelConfig,
) -> pd.DataFrame:
    arrivals = flights[flights["movement_type"] == "chegada"].copy()
    departures = flights[flights["movement_type"] == "partida"].copy()

    departures_by_key: Dict[Tuple[pd.Timestamp, str, str], List[dict]] = {}
    for _, row in departures.iterrows():
        key = (row["date"], row["company"], row["aircraft_code"])
        departures_by_key.setdefault(key, []).append(row.to_dict())

    for key in departures_by_key:
        departures_by_key[key].sort(key=lambda x: x["datetime"])

    visits: List[dict] = []
    used_departures: set[str] = set()

    for _, arr in arrivals.sort_values("datetime").iterrows():
        key = (arr["date"], arr["company"], arr["aircraft_code"])
        candidates = departures_by_key.get(key, [])

        chosen_dep = None
        for dep in candidates:
            if dep["movement_id"] in used_departures:
                continue
            delta_minutes = (dep["datetime"] - arr["datetime"]).total_seconds() / 60.0
            if config.min_turnaround_minutes <= delta_minutes <= config.max_turnaround_minutes:
                chosen_dep = dep
                break

        if chosen_dep is None:
            continue

        used_departures.add(chosen_dep["movement_id"])
        turnaround_minutes = int((chosen_dep["datetime"] - arr["datetime"]).total_seconds() / 60.0)
        aircraft_code = arr["aircraft_code"]
        category = aircraft_category_map.get(aircraft_code)

        if category is None:
            continue

        visits.append(
            {
                "visit_id": f"V_{arr['movement_id']}_{chosen_dep['movement_id']}",
                "arrival_id": arr["movement_id"],
                "departure_id": chosen_dep["movement_id"],
                "company": arr["company"],
                "aircraft_code": aircraft_code,
                "aircraft_category": category,
                "arrival_time": arr["datetime"],
                "departure_time": chosen_dep["datetime"],
                "arrival_pax": int(arr["seats"]),
                "departure_pax": int(chosen_dep["seats"]),
                "arrival_flight_number": arr["flight_number"],
                "departure_flight_number": chosen_dep["flight_number"],
                "turnaround_minutes": turnaround_minutes,
                "is_long_stay": int(turnaround_minutes > config.tow_threshold_minutes),
            }
        )

    visits_df = pd.DataFrame(visits)
    if visits_df.empty:
        raise ValueError(
            "Nenhuma visita chegada-partida foi reconstruída. "
            "Revise os parâmetros de emparelhamento ou os dados de entrada."
        )

    return visits_df.sort_values("arrival_time").reset_index(drop=True)


def build_operations(visits: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    operations: List[dict] = []

    for _, visit in visits.iterrows():
        visit_id = visit["visit_id"]
        arr_t = visit["arrival_time"]
        dep_t = visit["departure_time"]
        category = visit["aircraft_category"]
        company = visit["company"]
        long_stay = bool(visit["is_long_stay"])

        if long_stay:
            arr_end = min(arr_t + pd.Timedelta(minutes=config.disembark_minutes), dep_t)
            dep_start = max(dep_t - pd.Timedelta(minutes=config.embark_minutes), arr_end)

            operations.extend(
                [
                    {
                        "operation_id": f"{visit_id}_ARR",
                        "visit_id": visit_id,
                        "operation_type": "arrival",
                        "start_time": arr_t,
                        "end_time": arr_end,
                        "pax": int(visit["arrival_pax"]),
                        "company": company,
                        "aircraft_category": category,
                        "successor_operation_id": f"{visit_id}_PARK",
                        "allow_parking_only": False,
                    },
                    {
                        "operation_id": f"{visit_id}_PARK",
                        "visit_id": visit_id,
                        "operation_type": "parking",
                        "start_time": arr_end,
                        "end_time": dep_start,
                        "pax": 0,
                        "company": company,
                        "aircraft_category": category,
                        "successor_operation_id": f"{visit_id}_DEP",
                        "allow_parking_only": True,
                    },
                    {
                        "operation_id": f"{visit_id}_DEP",
                        "visit_id": visit_id,
                        "operation_type": "departure",
                        "start_time": dep_start,
                        "end_time": dep_t,
                        "pax": int(visit["departure_pax"]),
                        "company": company,
                        "aircraft_category": category,
                        "successor_operation_id": None,
                        "allow_parking_only": False,
                    },
                ]
            )
        else:
            operations.append(
                {
                    "operation_id": f"{visit_id}_TURN",
                    "visit_id": visit_id,
                    "operation_type": "turnaround",
                    "start_time": arr_t,
                    "end_time": dep_t,
                    "pax": int(visit["arrival_pax"] + visit["departure_pax"]),
                    "company": company,
                    "aircraft_category": category,
                    "successor_operation_id": None,
                    "allow_parking_only": False,
                }
            )

    ops_df = pd.DataFrame(operations)
    ops_df = ops_df[ops_df["end_time"] > ops_df["start_time"]].copy()
    ops_df = ops_df.sort_values("start_time").reset_index(drop=True)
    return ops_df


def is_category_compatible(operation_category: str, allowed_categories: set[str]) -> bool:
    op_cat = str(operation_category).strip().upper()
    return op_cat in allowed_categories


def build_compatible_stands(
    operations: pd.DataFrame,
    positions: pd.DataFrame,
) -> Dict[str, List[str]]:
    compatible: Dict[str, List[str]] = {}

    for _, op in operations.iterrows():
        op_id = op["operation_id"]
        op_cat = str(op["aircraft_category"]).strip().upper()
        allow_parking_only = bool(op["allow_parking_only"])

        candidates: List[str] = []
        for _, stand in positions.iterrows():
            stand_id = stand["stand_id"]
            allowed_categories = stand["allowed_categories"]
            stand_is_parking_only = bool(stand["is_parking_only"])

            if not is_category_compatible(op_cat, allowed_categories):
                continue
            if not allow_parking_only and stand_is_parking_only:
                continue

            candidates.append(stand_id)

        if not candidates:
            raise ValueError(f"A operação {op_id} não possui nenhuma posição compatível.")

        compatible[op_id] = candidates

    return compatible


def intervals_overlap(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
    buffer_minutes: int,
) -> bool:
    end_a_buffered = end_a + pd.Timedelta(minutes=buffer_minutes)
    end_b_buffered = end_b + pd.Timedelta(minutes=buffer_minutes)
    return start_a < end_b_buffered and start_b < end_a_buffered


def build_overlapping_operations(
    operations: pd.DataFrame,
    buffer_minutes: int,
) -> Dict[str, List[str]]:
    overlap_map: Dict[str, List[str]] = {op_id: [] for op_id in operations["operation_id"]}
    ops = operations.to_dict("records")

    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            op_i = ops[i]
            op_j = ops[j]
            if intervals_overlap(
                op_i["start_time"],
                op_i["end_time"],
                op_j["start_time"],
                op_j["end_time"],
                buffer_minutes,
            ):
                overlap_map[op_i["operation_id"]].append(op_j["operation_id"])
                overlap_map[op_j["operation_id"]].append(op_i["operation_id"])

    return overlap_map


def merge_overlap_maps(
    manual_overlap: Dict[str, List[str]],
    adjacency_overlap: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged: Dict[str, set[str]] = {}

    all_keys = set(manual_overlap.keys()).union(adjacency_overlap.keys())
    for key in all_keys:
        merged[key] = set(manual_overlap.get(key, [])) | set(adjacency_overlap.get(key, []))

    return {k: sorted(v) for k, v in merged.items()}


def prepare_problem_data(
    flights_path: str | Path,
    positions_path: str | Path,
    aircraft_categories_path: str | Path,
    aircraft_specs_path: Optional[str | Path],
    config: ModelConfig,
) -> ProblemData:
    flights_raw, positions_raw, aircraft_categories_raw, _ = read_input_data(
        flights_path=flights_path,
        positions_path=positions_path,
        aircraft_categories_path=aircraft_categories_path,
        aircraft_specs_path=aircraft_specs_path,
    )

    flights = normalize_flights(flights_raw)
    positions = normalize_positions(positions_raw, config)
    aircraft_category_map = build_aircraft_category_map(aircraft_categories_raw, config)
    visits = reconstruct_visits(flights, aircraft_category_map, config)
    operations = build_operations(visits, config)
    compatible_stands = build_compatible_stands(operations, positions)
    overlapping_ops = build_overlapping_operations(operations, config.turnaround_buffer_minutes)

    auto_adjacent_stands = build_adjacent_stands(positions)
    overlapping_stands = merge_overlap_maps(config.overlapping_stands, auto_adjacent_stands)

    return ProblemData(
        flights_raw=flights,
        positions=positions,
        visits=visits,
        operations=operations,
        compatible_stands=compatible_stands,
        overlapping_ops=overlapping_ops,
        overlapping_stands=overlapping_stands,
        adjacent_stands=auto_adjacent_stands,
    )


# ============================================================
# Construção do modelo Gurobi
# ============================================================


@dataclass
class OptimizationResult:
    status: str
    objective_value: Optional[float]
    allocation: pd.DataFrame
    tows: pd.DataFrame


def build_model(problem: ProblemData, config: ModelConfig) -> Tuple[gp.Model, gp.tupledict, gp.tupledict]:
    model = gp.Model("airport_stand_allocation")
    model.Params.OutputFlag = 1 if config.verbose else 0
    if config.time_limit_seconds is not None:
        model.Params.TimeLimit = config.time_limit_seconds
    if config.mip_gap is not None:
        model.Params.MIPGap = config.mip_gap

    operations = problem.operations.set_index("operation_id")
    positions = problem.positions.set_index("stand_id")

    x_index = [
        (op_id, stand_id)
        for op_id, stands in problem.compatible_stands.items()
        for stand_id in stands
    ]
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x")

    tow_candidates = operations[operations["successor_operation_id"].notna()].index.tolist()
    y = model.addVars(tow_candidates, vtype=GRB.BINARY, name="y")

    # 1) Alocação única
    for op_id, stands in problem.compatible_stands.items():
        model.addConstr(gp.quicksum(x[op_id, s] for s in stands) == 1, name=f"assign_{op_id}")

    # 2) Conflitos temporais na mesma posição
    added_same_stand_conflicts = set()
    for op_i, overlapping_list in problem.overlapping_ops.items():
        compatible_i = set(problem.compatible_stands[op_i])
        for op_k in overlapping_list:
            pair_key = tuple(sorted((op_i, op_k)))
            if pair_key in added_same_stand_conflicts:
                continue
            added_same_stand_conflicts.add(pair_key)

            common_stands = compatible_i.intersection(problem.compatible_stands[op_k])
            for stand_id in common_stands:
                model.addConstr(
                    x[op_i, stand_id] + x[op_k, stand_id] <= 1,
                    name=f"time_conflict_{op_i}_{op_k}_{stand_id}",
                )

    # 3) Bloqueio de posições adjacentes quando D/E ocupa uma posição
    added_adjacency_conflicts = set()
    for op_i, overlapping_list in problem.overlapping_ops.items():
        op_i_cat = str(operations.loc[op_i, "aircraft_category"]).upper()

        # Só D ou E bloqueiam adjacentes
        if op_i_cat not in {"D", "E"}:
            continue

        for stand_a in problem.compatible_stands[op_i]:
            adjacent_list = problem.overlapping_stands.get(stand_a, [])
            if not adjacent_list:
                continue

            for op_k in overlapping_list:
                for stand_b in adjacent_list:
                    if (op_k, stand_b) not in x:
                        continue
                    if (op_i, stand_a) not in x:
                        continue

                    key = (op_i, stand_a, op_k, stand_b)
                    if key in added_adjacency_conflicts:
                        continue
                    added_adjacency_conflicts.add(key)

                    model.addConstr(
                        x[op_i, stand_a] + x[op_k, stand_b] <= 1,
                        name=f"adj_block_{op_i}_{stand_a}_{op_k}_{stand_b}",
                    )

    # 4) Definição do reboque
    for op_id in tow_candidates:
        successor_id = operations.loc[op_id, "successor_operation_id"]
        common_stands = set(problem.compatible_stands[op_id]).intersection(problem.compatible_stands[successor_id])
        for stand_id in common_stands:
            model.addConstr(
                x[op_id, stand_id] - x[successor_id, stand_id] <= y[op_id],
                name=f"tow_def_{op_id}_{stand_id}",
            )

    set_objective(model, x, y, problem, config)
    return model, x, y


def set_objective(
    model: gp.Model,
    x: gp.tupledict,
    y: gp.tupledict,
    problem: ProblemData,
    config: ModelConfig,
) -> None:
    operations = problem.operations.set_index("operation_id")
    positions = problem.positions.set_index("stand_id")

    if config.objective == "walking_distance":
        expr = gp.quicksum(
            operations.loc[op_id, "pax"] * positions.loc[stand_id, "walking_distance"] * x[op_id, stand_id]
            for op_id, stand_id in x.keys()
        )
        model.setObjective(expr, GRB.MINIMIZE)

    elif config.objective == "contact_share":
        expr = gp.quicksum(
            operations.loc[op_id, "pax"] * positions.loc[stand_id, "is_contact"] * x[op_id, stand_id]
            for op_id, stand_id in x.keys()
        )
        model.setObjective(expr, GRB.MAXIMIZE)

    elif config.objective == "tow_count":
        expr = gp.quicksum(y[op_id] for op_id in y.keys())
        model.setObjective(expr, GRB.MINIMIZE)

    elif config.objective == "revenue":
        expr = gp.quicksum(
            operations.loc[op_id, "pax"] * positions.loc[stand_id, "revenue_factor"] * x[op_id, stand_id]
            for op_id, stand_id in x.keys()
        )
        model.setObjective(expr, GRB.MAXIMIZE)

    else:
        raise ValueError(
            "Objetivo inválido. Use: walking_distance, contact_share, tow_count ou revenue."
        )


# ============================================================
# Pós-processamento
# ============================================================


def extract_solution(
    model: gp.Model,
    x: gp.tupledict,
    y: gp.tupledict,
    problem: ProblemData,
) -> OptimizationResult:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    status = status_map.get(model.Status, f"STATUS_{model.Status}")

    if model.SolCount == 0:
        return OptimizationResult(
            status=status,
            objective_value=None,
            allocation=pd.DataFrame(),
            tows=pd.DataFrame(),
        )

    operations = problem.operations.set_index("operation_id")
    positions = problem.positions.set_index("stand_id")

    allocation_rows: List[dict] = []
    for (op_id, stand_id), var in x.items():
        if var.X > 0.5:
            allocation_rows.append(
                {
                    "operation_id": op_id,
                    "visit_id": operations.loc[op_id, "visit_id"],
                    "operation_type": operations.loc[op_id, "operation_type"],
                    "aircraft_category": operations.loc[op_id, "aircraft_category"],
                    "start_time": operations.loc[op_id, "start_time"],
                    "end_time": operations.loc[op_id, "end_time"],
                    "pax": operations.loc[op_id, "pax"],
                    "stand_id": stand_id,
                    "stand_type": positions.loc[stand_id, "stand_type"],
                    "walking_distance": positions.loc[stand_id, "walking_distance"],
                    "is_contact": positions.loc[stand_id, "is_contact"],
                }
            )

    tow_rows: List[dict] = []
    for op_id, var in y.items():
        if var.X > 0.5:
            tow_rows.append(
                {
                    "operation_id": op_id,
                    "visit_id": operations.loc[op_id, "visit_id"],
                    "successor_operation_id": operations.loc[op_id, "successor_operation_id"],
                    "tow": 1,
                }
            )

    allocation_df = pd.DataFrame(allocation_rows).sort_values(["start_time", "stand_id"]).reset_index(drop=True)
    tows_df = pd.DataFrame(tow_rows).reset_index(drop=True)

    return OptimizationResult(
        status=status,
        objective_value=float(model.ObjVal),
        allocation=allocation_df,
        tows=tows_df,
    )


def save_outputs(result: OptimizationResult, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not result.allocation.empty:
        result.allocation.to_csv(output_path / "alocacao_resultado.csv", index=False)
    if not result.tows.empty:
        result.tows.to_csv(output_path / "reboques_resultado.csv", index=False)


# ============================================================
# Execução principal
# ============================================================


def solve_airport_stand_allocation(
    flights_path: str | Path,
    positions_path: str | Path,
    aircraft_categories_path: str | Path,
    aircraft_specs_path: Optional[str | Path] = None,
    output_dir: str | Path = "outputs",
    config: Optional[ModelConfig] = None,
) -> OptimizationResult:
    if config is None:
        config = ModelConfig()

    problem = prepare_problem_data(
        flights_path=flights_path,
        positions_path=positions_path,
        aircraft_categories_path=aircraft_categories_path,
        aircraft_specs_path=aircraft_specs_path,
        config=config,
    )

    model, x, y = build_model(problem, config)
    model.optimize()
    result = extract_solution(model, x, y, problem)
    save_outputs(result, output_dir)
    return result


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    config = ModelConfig(
        objective="walking_distance",
        time_limit_seconds=300,
        mip_gap=0.01,
        verbose=True,
    )

    result = solve_airport_stand_allocation(
        flights_path=BASE_DIR / "voos.csv",
        positions_path=BASE_DIR / "posicoes.csv",
        aircraft_categories_path=BASE_DIR / "categoriaaeronaves.csv",
        aircraft_specs_path=BASE_DIR / "especificacoesaeronaves.csv",
        output_dir=BASE_DIR / "outputs",
        config=config,
    )

    print("\n================ RESULTADO ================")
    print(f"Status: {result.status}")
    print(f"Valor da função objetivo: {result.objective_value}")
    print(f"Operações alocadas: {len(result.allocation)}")
    print(f"Reboques identificados: {len(result.tows)}")
    if not result.allocation.empty:
        print(result.allocation.head(20).to_string(index=False))