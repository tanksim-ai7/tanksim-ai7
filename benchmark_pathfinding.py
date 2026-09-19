"""
benchmark_pathfinding.py

A* vs D* Lite 속도 비교 벤치마크
--------------------------------
맵: 300 x 300 고정 (cell_size, obstacle_margin 등은 상단 CONFIG에서 조정)

3개 시나리오를 각각 CSV로 로깅한다:
  1) cold_start   : 장애물 세팅 직후 첫 find_path() 1회 (장애물 개수별 반복)
  2) tick_replan  : goal 고정, start가 매 틱 이동할 때 find_path() (실시간 추적 시뮬레이션)
  3) obstacle_evt : 장애물 추가/이동 직후 find_path() (동적 장애물 대응)

실행:
    python benchmark_pathfinding.py

결과:
    ./bench_results/cold_start.csv
    ./bench_results/tick_replan.csv
    ./bench_results/obstacle_event.csv
콘솔에는 시나리오별 요약 통계(mean/median/p95/max)도 함께 출력됨.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import List, Tuple, Type

# ----------------------------------------------------------------------
# 프로젝트 실제 경로 구조에 맞춰 하나만 살아있으면 됨
# ----------------------------------------------------------------------
try:
    from pathfinding.astar_planner import AStarPlanner, ObstacleRect
    from pathfinding.dstar_lite_planner import DStarLitePlanner
except ImportError:
    from astar_planner import AStarPlanner, ObstacleRect
    from dstar_lite_planner import DStarLitePlanner


# ========================================================================
# CONFIG
# ========================================================================
GRID_KW = dict(
    grid_min_x=0.0, grid_max_x=300.0,
    grid_min_z=0.0, grid_max_z=300.0,
    cell_size=1.0,
    obstacle_margin=2.0,
    allow_diagonal=True,
)

START = (10.0, 150.0)
GOAL = (290.0, 150.0)

SEED = 42                      # 모든 알고리즘이 동일 조건을 보도록 시드 고정
OUT_DIR = "bench_results"

# 시나리오 1: 콜드스타트
COLD_START_OBSTACLE_COUNTS = [5, 15, 30]
COLD_START_TRIALS_PER_COUNT = 20

# 시나리오 2: 틱 재계산
TICK_EPISODES = 10
TICKS_PER_EPISODE = 200
TICK_STEP = 1.0                 # 틱당 이동 거리(월드 단위)
TICK_OBSTACLE_COUNT = 20        # 이동 중엔 장애물 고정

# 시나리오 3: 장애물 변화 재계산
OBSTACLE_EVT_EPISODES = 10
OBSTACLE_EVT_PER_EPISODE = 10
OBSTACLE_EVT_BASE_COUNT = 15    # 시작 장애물 개수
OBSTACLE_EVT_ADD_SIZE = (10.0, 25.0)  # 매 이벤트마다 추가되는 장애물 크기 범위


# ========================================================================
# 유틸
# ========================================================================
def path_length(path: List[Tuple[float, float]]) -> float:
    """경로(웨이포인트 리스트)의 총 유클리드 거리. 경로 품질(동일성) 비교용."""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for (x1, z1), (x2, z2) in zip(path, path[1:]):
        total += math.hypot(x2 - x1, z2 - z1)
    return total


def gen_obstacles(rng: random.Random, count: int) -> List[ObstacleRect]:
    """START/GOAL 근처는 피해서 랜덤 사각 장애물 생성 (시드 고정 rng 사용 = 재현 가능)"""
    obstacles = []
    gmin_x, gmax_x = GRID_KW["grid_min_x"], GRID_KW["grid_max_x"]
    gmin_z, gmax_z = GRID_KW["grid_min_z"], GRID_KW["grid_max_z"]
    for _ in range(count):
        cx = rng.uniform(gmin_x + 40, gmax_x - 40)
        cz = rng.uniform(gmin_z + 40, gmax_z - 40)
        sx = rng.uniform(10.0, 35.0)
        sz = rng.uniform(10.0, 35.0)
        obstacles.append(ObstacleRect(center_x=cx, center_z=cz, size_x=sx, size_z=sz))
    return obstacles


def timed_find_path(planner, start, goal) -> Tuple[List[Tuple[float, float]], float]:
    """A*/D* Lite 공통: find_path() 를 감싸서 wall-clock 기준 ms를 잰다.
    D* Lite는 자체 last_compute_time_ms가 있지만, A*와 '같은 기준'으로 비교하려면
    이 외부 타이머 값을 기준 지표로 써야 한다 (reconstruct_path 포함 여부 등 조건 통일)."""
    t0 = time.perf_counter()
    path = planner.find_path(start, goal)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return path, elapsed_ms


def compute_type_of(planner) -> str:
    """D* Lite면 full_init/incremental 반환, A*는 해당 개념이 없으므로 'n/a'."""
    return getattr(planner, "last_compute_type", "n/a")


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def summarize(values: List[float]) -> str:
    if not values:
        return "n/a"
    vs = sorted(values)
    p95 = vs[int(len(vs) * 0.95) - 1] if len(vs) >= 20 else vs[-1]
    return (f"mean={mean(values):.3f}ms  median={median(values):.3f}ms  "
            f"p95={p95:.3f}ms  max={max(values):.3f}ms  n={len(values)}")


# ========================================================================
# 시나리오 1: Cold Start
# ========================================================================
def run_cold_start(planner_cls: Type, algo_name: str, writer: csv.writer):
    print(f"\n=== [{algo_name}] 시나리오 1: Cold Start ===")
    for obstacle_count in COLD_START_OBSTACLE_COUNTS:
        times = []
        for trial in range(COLD_START_TRIALS_PER_COUNT):
            rng = random.Random((SEED, obstacle_count, trial))
            obstacles = gen_obstacles(rng, obstacle_count)

            planner = planner_cls(**GRID_KW)
            planner.set_obstacles(obstacles)

            path, elapsed_ms = timed_find_path(planner, START, GOAL)
            times.append(elapsed_ms)

            writer.writerow([algo_name, obstacle_count, trial,
                              elapsed_ms, len(path), path_length(path)])

        print(f"  obstacle_count={obstacle_count:>3}: {summarize(times)}")


# ========================================================================
# 시나리오 2: 틱 단위 재계산 (실시간 위치 추적)
# ========================================================================
def run_tick_replan(planner_cls: Type, algo_name: str, writer: csv.writer):
    print(f"\n=== [{algo_name}] 시나리오 2: 틱 재계산 (start 이동, goal 고정) ===")
    all_times_after_first = []

    for episode in range(TICK_EPISODES):
        rng = random.Random((SEED, "tick", episode))
        obstacles = gen_obstacles(rng, TICK_OBSTACLE_COUNT)

        planner = planner_cls(**GRID_KW)
        planner.set_obstacles(obstacles)
        if hasattr(planner, "reset"):
            planner.reset()

        current_pos = START
        for tick in range(TICKS_PER_EPISODE):
            path, elapsed_ms = timed_find_path(planner, current_pos, GOAL)
            ctype = compute_type_of(planner)

            writer.writerow([algo_name, episode, tick, ctype,
                              elapsed_ms, len(path), path_length(path)])

            if tick > 0:  # 첫 틱(=사실상 콜드스타트)은 재계산 통계에서 제외
                all_times_after_first.append(elapsed_ms)

            # 다음 웨이포인트 방향으로 한 스텝 이동 (경로 없으면 제자리)
            if path and len(path) > 1:
                nx, nz = path[1]
                dx, dz = nx - current_pos[0], nz - current_pos[1]
                dist = math.hypot(dx, dz)
                if dist > 1e-6:
                    step = min(TICK_STEP, dist)
                    current_pos = (current_pos[0] + dx / dist * step,
                                   current_pos[1] + dz / dist * step)

            if math.hypot(current_pos[0] - GOAL[0], current_pos[1] - GOAL[1]) < 2.0:
                break

    print(f"  틱 재계산(2번째 틱부터): {summarize(all_times_after_first)}")


# ========================================================================
# 시나리오 3: 장애물 변화 재계산
# ========================================================================
def run_obstacle_events(planner_cls: Type, algo_name: str, writer: csv.writer):
    print(f"\n=== [{algo_name}] 시나리오 3: 장애물 변화 재계산 ===")
    all_times = []

    for episode in range(OBSTACLE_EVT_EPISODES):
        rng = random.Random((SEED, "obs_evt", episode))
        obstacles = gen_obstacles(rng, OBSTACLE_EVT_BASE_COUNT)

        planner = planner_cls(**GRID_KW)
        planner.set_obstacles(obstacles)
        if hasattr(planner, "reset"):
            planner.reset()

        # 최초 경로 1회 확보 (측정 대상 아님, 다음 재계산의 전제 조건일 뿐)
        planner.find_path(START, GOAL)

        current_pos = START
        for evt_idx in range(OBSTACLE_EVT_PER_EPISODE):
            gmin_x, gmax_x = GRID_KW["grid_min_x"], GRID_KW["grid_max_x"]
            gmin_z, gmax_z = GRID_KW["grid_min_z"], GRID_KW["grid_max_z"]
            new_obs = ObstacleRect(
                center_x=rng.uniform(gmin_x + 40, gmax_x - 40),
                center_z=rng.uniform(gmin_z + 40, gmax_z - 40),
                size_x=rng.uniform(*OBSTACLE_EVT_ADD_SIZE),
                size_z=rng.uniform(*OBSTACLE_EVT_ADD_SIZE),
            )
            obstacles = obstacles + [new_obs]

            t0 = time.perf_counter()
            planner.set_obstacles(obstacles)          # 장애물 갱신 자체 비용도 포함
            path, find_ms = timed_find_path(planner, current_pos, GOAL)
            total_ms = (time.perf_counter() - t0) * 1000.0

            all_times.append(total_ms)
            writer.writerow([algo_name, episode, evt_idx,
                              total_ms, find_ms, len(path), path_length(path)])

    print(f"  장애물 변화 이벤트: {summarize(all_times)}")


# ========================================================================
# MAIN
# ========================================================================
def main():
    ensure_out_dir()

    algos = [
        (AStarPlanner, "AStar"),
        (DStarLitePlanner, "DStarLite"),
    ]

    with open(os.path.join(OUT_DIR, "cold_start.csv"), "w", newline="") as f1, \
         open(os.path.join(OUT_DIR, "tick_replan.csv"), "w", newline="") as f2, \
         open(os.path.join(OUT_DIR, "obstacle_event.csv"), "w", newline="") as f3:

        w1 = csv.writer(f1)
        w1.writerow(["algo", "obstacle_count", "trial", "elapsed_ms", "path_points", "path_length"])

        w2 = csv.writer(f2)
        w2.writerow(["algo", "episode", "tick", "compute_type", "elapsed_ms", "path_points", "path_length"])

        w3 = csv.writer(f3)
        w3.writerow(["algo", "episode", "event_idx", "total_ms", "find_path_ms", "path_points", "path_length"])

        for planner_cls, name in algos:
            run_cold_start(planner_cls, name, w1)
            run_tick_replan(planner_cls, name, w2)
            run_obstacle_events(planner_cls, name, w3)

    print(f"\n완료. CSV는 ./{OUT_DIR}/ 안에 저장됨 (엑셀/PPT 차트용으로 바로 사용 가능).")


if __name__ == "__main__":
    main()
