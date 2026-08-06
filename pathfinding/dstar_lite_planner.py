"""
dstar_lite_planner.py

D* Lite 기반 2D(XZ) 경로 탐색 + 시각화 클래스
--------------------------------------------
- astar_planner.py 의 AStarPlanner 와 "완전히 동일한 외부 인터페이스"를 갖도록 설계함
  (grid_min_x/max_x/min_z/max_z, cell_size, obstacle_margin, allow_diagonal,
   set_obstacles(), find_path(), plot()).
  -> server_sample_v2_3_10.py 에서는 import 문과 planner 생성 라인만 바꾸면
     나머지 코드는 그대로 재사용 가능.

- D* Lite 는 "목적지(goal)를 뿌리로 삼아 역방향으로 탐색"하는 알고리즘이라,
  목적지는 고정된 채 (1) 탱크(start)가 계속 움직이거나 (2) 장애물 일부만 바뀌는
  상황에서는 매번 전체 그리드를 다시 계산하지 않고, "바뀐 부분 주변"만
  다시 계산(증분 재계산)해서 훨씬 빠르게 새 경로를 뽑아낼 수 있음.
  -> 실험 설계서에서 말한 "재계산 시 baseline(A*) 대비 얼마나 빠른가"를
     확인하기 위한 핵심 포인트가 바로 이 부분.

- 계산 시간 측정:
    planner.last_compute_time_ms  : 가장 최근 계산(재계산 포함)에 걸린 시간(ms)
    planner.last_compute_type     : "full_init" / "incremental" 중 어떤 방식이었는지
    planner.compute_log           : 지금까지의 모든 계산 기록 [(type, ms), ...]
  find_path() 를 호출할 때마다 자동으로 걸린 시간을 콘솔에 출력함.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - matplotlib 미설치 환경 대비
    _HAS_MPL = False

@dataclass
class ObstacleRect:
    """
    XZ 평면에서의 축 정렬 사각형 장애물
    (Unity 에서 전달받는 /update_obstacle payload 형식과 쉽게 매핑하기 위한 구조)

    astar_planner.py 에 있던 것과 동일한 정의를 이 파일 안으로 옮겨온 것.
    """
    center_x: float
    center_z: float
    size_x: float
    size_z: float

    @property
    def x_min(self) -> float:
        return self.center_x - self.size_x * 0.5

    @property
    def x_max(self) -> float:
        return self.center_x + self.size_x * 0.5

    @property
    def z_min(self) -> float:
        return self.center_z - self.size_z * 0.5

    @property
    def z_max(self) -> float:
        return self.center_z + self.size_z * 0.5

    @classmethod
    def from_min_max(cls, x_min: float, x_max: float, z_min: float, z_max: float) -> "ObstacleRect":
        """/update_obstacle 의 x_min, x_max, z_min, z_max 형식에서 바로 만들기 편하도록 제공"""
        cx = (x_min + x_max) * 0.5
        cz = (z_min + z_max) * 0.5
        sx = (x_max - x_min)
        sz = (z_max - z_min)
        return cls(center_x=cx, center_z=cz, size_x=sx, size_z=sz)


INF = math.inf


class _DNode:
    """D* Lite 내부용 노드. g/rhs 두 개의 비용 값을 갖는 것이 A* 노드와의 차이점."""

    __slots__ = ("ix", "iz", "walkable", "g", "rhs", "key")

    def __init__(self, ix: int, iz: int, walkable: bool):
        self.ix = ix
        self.iz = iz
        self.walkable = walkable
        self.g: float = INF
        self.rhs: float = INF
        # key: 이 노드가 현재 open queue 안에 유효한 상태로 들어있다면 그 key(k1, k2) 튜플.
        # 큐에 들어있지 않으면 None. (heapq는 특정 원소를 직접 지울 수 없으므로
        # "지금 큐에 있는 이 노드의 최신 key가 뭔지"를 여기 저장해두고,
        # pop 했을 때 이 값과 다르면 오래된(stale) 항목이라고 보고 버림)
        self.key: Optional[Tuple[float, float]] = None


class DStarLitePlanner:
    """
    D* Lite 경로 탐색 + 시각화 클래스 (AStarPlanner 와 동일한 외부 인터페이스)

    사용 흐름 (server 쪽 관점에서는 AStarPlanner 와 완전히 동일):
        planner = DStarLitePlanner(grid_min_x=..., ...)
        planner.set_obstacles([...])
        path = planner.find_path(start, goal)
        ...(장애물 갱신)...
        planner.set_obstacles([...])          # 내부적으로 바뀐 셀만 찾아서 반영
        path = planner.find_path(new_start, goal)   # 내부적으로 증분 재계산

    내부적으로는:
        - goal 이 바뀌면(target 자체가 달라지면) -> 전체 재초기화(full re-init) 후 계산
        - goal 이 그대로고 start 만 바뀌었으면 -> km(heuristic 보정항) 갱신 후 증분 재계산
        - 장애물만 바뀌었으면 -> 바뀐 셀 + 그 이웃만 UpdateVertex 후 증분 재계산
    """

    def __init__(
        self,
        grid_min_x: float,
        grid_max_x: float,
        grid_min_z: float,
        grid_max_z: float,
        cell_size: float = 1.0,
        obstacle_margin: float = 2.0,
        allow_diagonal: bool = True,
    ) -> None:
        assert cell_size > 0.0, "cell_size must be > 0"

        self.grid_min_x = float(grid_min_x)
        self.grid_max_x = float(grid_max_x)
        self.grid_min_z = float(grid_min_z)
        self.grid_max_z = float(grid_max_z)
        self.cell_size = float(cell_size)
        self.obstacle_margin = float(obstacle_margin)
        self.allow_diagonal = bool(allow_diagonal)

        self.grid_size_x = max(1, int(math.ceil((self.grid_max_x - self.grid_min_x) / self.cell_size)))
        self.grid_size_z = max(1, int(math.ceil((self.grid_max_z - self.grid_min_z) / self.cell_size)))

        self._obstacles: List[ObstacleRect] = []
        self._grid: List[List[_DNode]] = []
        self._grid_built: bool = False

        # D* Lite 탐색 상태
        self._open_heap: List[Tuple[float, float, int, _DNode]] = []
        self._counter = itertools.count()
        self._km: float = 0.0
        self._s_start: Optional[_DNode] = None
        self._s_goal: Optional[_DNode] = None
        self._goal_world: Optional[Tuple[float, float]] = None  # goal 이 바뀌었는지 판단용
        self._last_start_node: Optional[_DNode] = None          # km 갱신용(직전 start)
        self._initialized: bool = False

        # 계산 시간 로깅
        self.last_compute_time_ms: float = 0.0
        self.last_compute_type: Optional[str] = None
        self.compute_log: List[Tuple[str, float]] = []

    # ------------------------------------------------------------------
    # 그리드 & 장애물
    # ------------------------------------------------------------------
    def _build_grid(self) -> None:
        """최초 1회 grid_size_x * grid_size_z 크기의 노드 배열을 walkable=True 로 생성"""
        self._grid = [
            [_DNode(ix, iz, True) for iz in range(self.grid_size_z)]
            for ix in range(self.grid_size_x)
        ]
        self._grid_built = True

    def _obstacle_cells(self, obstacles: Iterable[ObstacleRect]) -> "set[Tuple[int, int]]":
        """장애물(+margin) 목록이 덮는 그리드 셀 (ix, iz) 집합을 계산"""
        cells: "set[Tuple[int, int]]" = set()
        for obs in obstacles:
            x_min = obs.x_min - self.obstacle_margin
            x_max = obs.x_max + self.obstacle_margin
            z_min = obs.z_min - self.obstacle_margin
            z_max = obs.z_max + self.obstacle_margin

            ix_min = max(0, int(math.floor((x_min - self.grid_min_x) / self.cell_size)))
            ix_max = min(self.grid_size_x - 1, int(math.floor((x_max - self.grid_min_x) / self.cell_size)))
            iz_min = max(0, int(math.floor((z_min - self.grid_min_z) / self.cell_size)))
            iz_max = min(self.grid_size_z - 1, int(math.floor((z_max - self.grid_min_z) / self.cell_size)))

            if ix_min > ix_max or iz_min > iz_max:
                continue
            for ix in range(ix_min, ix_max + 1):
                for iz in range(iz_min, iz_max + 1):
                    cells.add((ix, iz))
        return cells

    def set_obstacles(self, obstacles: Iterable[ObstacleRect]) -> None:
        """
        장애물 목록을 갱신.

        [증분 갱신의 핵심]
        AStarPlanner.set_obstacles() 는 그냥 리스트만 저장해두고 다음 find_path() 때
        그리드 전체를 다시 그림(walkable 전체 재계산). D* Lite 에서는 대신:
          1) 새 장애물 목록으로 "블록되어야 할 셀 집합"을 계산
          2) 기존 walkable 상태와 비교해서 "실제로 상태가 바뀐 셀"만 추려냄
          3) 바뀐 셀만 walkable 플래그를 뒤집고, 그 셀 + 8방향 이웃에 대해서만
             UpdateVertex() 를 호출해서 rhs 값을 갱신 (= 영향받는 영역만 큐에 재등록)
        -> 다음 find_path() 에서 ComputeShortestPath() 가 바뀐 부분 주변만 훑고
           끝나므로, 맵 전체 크기와 무관하게 "바뀐 장애물 수"에 비례해서 빨라짐.
        """
        if not self._grid_built:
            self._build_grid()

        new_cells = self._obstacle_cells(obstacles)
        changed: List[_DNode] = []

        # 새로 블록된 셀
        for (ix, iz) in new_cells:
            node = self._grid[ix][iz]
            if node.walkable:
                node.walkable = False
                changed.append(node)

        # 기존엔 블록이었는데 이번엔 빠진 셀 (장애물이 사라짐/이동)
        # -> 이전 장애물 집합을 기억해뒀다가 비교
        old_cells = self._obstacle_cells(self._obstacles)
        removed_cells = old_cells - new_cells
        for (ix, iz) in removed_cells:
            node = self._grid[ix][iz]
            if not node.walkable:
                node.walkable = True
                changed.append(node)

        self._obstacles = list(obstacles)

        if not self._initialized:
            # 아직 탐색을 시작한 적이 없으면(첫 find_path 이전) 그냥 상태만 반영해두고 끝.
            return

        if changed:
            # 바뀐 셀 자신 + 그 이웃들만 UpdateVertex.
            # (이웃까지 갱신하는 이유: 이 셀을 거쳐가던 이웃 노드들의 rhs 가
            #  이 셀의 walkable 상태 변화로 인해 같이 바뀌어야 하기 때문)
            to_update: "set[_DNode]" = set()
            for node in changed:
                to_update.add(node)
                to_update.update(self._neighbors(node))
            for node in to_update:
                self._update_vertex(node)

    # ------------------------------------------------------------------
    # 좌표 변환 (AStarPlanner 와 동일)
    # ------------------------------------------------------------------
    def world_to_grid(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        x = round(float(x), 2)
        z = round(float(z), 2)
        if not (self.grid_min_x <= x <= self.grid_max_x and self.grid_min_z <= z <= self.grid_max_z):
            return None
        fx = (x - self.grid_min_x) / self.cell_size
        fz = (z - self.grid_min_z) / self.cell_size
        ix = int(math.floor(fx))
        iz = int(math.floor(fz))
        if ix < 0 or ix >= self.grid_size_x or iz < 0 or iz >= self.grid_size_z:
            return None
        return ix, iz

    def grid_index_to_world(self, ix: int, iz: int) -> Tuple[float, float]:
        x = self.grid_min_x + (ix + 0.5) * self.cell_size
        z = self.grid_min_z + (iz + 0.5) * self.cell_size
        return round(x, 2), round(z, 2)

    def _nearest_walkable(self, ix: int, iz: int, max_radius: int = 25) -> Optional[_DNode]:
        """AStarPlanner 와 동일한 목적: 시작/목적지가 장애물 위에 걸려있을 때 대체 셀 탐색"""
        if self._grid[ix][iz].walkable:
            return self._grid[ix][iz]
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if max(abs(dx), abs(dz)) != radius:
                        continue
                    nx, nz = ix + dx, iz + dz
                    if 0 <= nx < self.grid_size_x and 0 <= nz < self.grid_size_z:
                        node = self._grid[nx][nz]
                        if node.walkable:
                            return node
        return None

    # ------------------------------------------------------------------
    # D* Lite 핵심 로직
    # ------------------------------------------------------------------
    def _neighbors(self, node: _DNode) -> Iterable[_DNode]:
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                if not self.allow_diagonal and abs(dx) + abs(dz) > 1:
                    continue
                ix = node.ix + dx
                iz = node.iz + dz
                if 0 <= ix < self.grid_size_x and 0 <= iz < self.grid_size_z:
                    yield self._grid[ix][iz]

    @staticmethod
    def _octile(a_ix: int, a_iz: int, b_ix: int, b_iz: int) -> float:
        """A* 쪽 _distance_cost 와 동일한 스케일(직선=10, 대각선=14)의 휴리스틱/이동비용"""
        dx = abs(a_ix - b_ix)
        dz = abs(a_iz - b_iz)
        diag = min(dx, dz)
        straight = abs(dx - dz)
        return 14 * diag + 10 * straight

    def _heuristic(self, a: _DNode, b: _DNode) -> float:
        return self._octile(a.ix, a.iz, b.ix, b.iz)

    def _edge_cost(self, u: _DNode, v: _DNode) -> float:
        """u, v 가 인접 셀일 때 이동 비용. 둘 중 하나라도 막혀있으면 무한대."""
        if not u.walkable or not v.walkable:
            return INF
        return self._octile(u.ix, u.iz, v.ix, v.iz)

    def _calc_key(self, node: _DNode) -> Tuple[float, float]:
        g_min = min(node.g, node.rhs)
        return (g_min + self._heuristic(self._s_start, node) + self._km, g_min)

    def _push(self, node: _DNode) -> None:
        node.key = self._calc_key(node)
        heapq.heappush(
            self._open_heap,
            (node.key[0], node.key[1], next(self._counter), node),
        )

    def _update_vertex(self, node: _DNode) -> None:
        """
        rhs(node) 를 이웃 기준으로 다시 계산하고, g != rhs 이면 (재)등록,
        g == rhs 이면 큐에서 사실상 제거(= key 를 None 처리, pop 될 때 stale 로 걸러짐)
        """
        if node is not self._s_goal:
            best = INF
            for nb in self._neighbors(node):
                cost = self._edge_cost(node, nb)
                if cost < INF:
                    val = cost + nb.g
                    if val < best:
                        best = val
            node.rhs = best

        if node.g != node.rhs:
            self._push(node)
        else:
            node.key = None  # 큐에 남아있던 예전 항목은 pop 시 stale 로 판정되어 버려짐

    def _compute_shortest_path(self) -> None:
        s_start = self._s_start
        while self._open_heap:
            k1, k2, _, u = self._open_heap[0]

            # stale(오래된) 항목이면 그냥 버리고 다음 것 확인
            if u.key is None or (k1, k2) != u.key:
                heapq.heappop(self._open_heap)
                continue

            start_key = self._calc_key(s_start)
            if (k1, k2) >= start_key and s_start.rhs == s_start.g:
                break

            heapq.heappop(self._open_heap)
            k_new = self._calc_key(u)

            if (k1, k2) < k_new:
                # 그 사이에 km 등이 바뀌어 key 가 더 커짐 -> 갱신해서 다시 넣기
                self._push(u)
            elif u.g > u.rhs:
                # 더 짧은 경로를 찾음 -> 확정
                u.g = u.rhs
                u.key = None
                for nb in self._neighbors(u):
                    self._update_vertex(nb)
            else:
                # u 를 거쳐가는 경로가 무효화됨(장애물 등) -> g 를 무한대로 낮추고 재전파
                u.g = INF
                self._update_vertex(u)
                for nb in self._neighbors(u):
                    self._update_vertex(nb)

    # ------------------------------------------------------------------
    # 외부 공개 API: find_path (AStarPlanner.find_path 와 동일한 시그니처)
    # ------------------------------------------------------------------
    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """
        D* Lite 로 start -> goal 경로를 계산해서 월드 좌표 리스트로 반환.
        (내부적으로 상황에 따라 full_init 또는 incremental 방식을 자동 선택)
        """
        if not self._grid_built:
            self._build_grid()

        start_idx = self.world_to_grid(*start)
        goal_idx = self.world_to_grid(*goal)
        if start_idx is None or goal_idx is None:
            return []

        sx, sz = start_idx
        gx, gz = goal_idx

        start_node = self._grid[sx][sz]
        goal_node = self._grid[gx][gz]

        if not start_node.walkable:
            snapped = self._nearest_walkable(sx, sz)
            if snapped is None:
                return []
            start_node = snapped
        if not goal_node.walkable:
            snapped = self._nearest_walkable(gx, gz)
            if snapped is None:
                return []
            goal_node = snapped

        goal_changed = (self._goal_world != (goal_node.ix, goal_node.iz))

        t0 = time.perf_counter()

        if not self._initialized or goal_changed:
            # ---- 전체 초기화 (baseline 과 동등 조건: 이번에도 그리드 전체를 훑음) ----
            self._full_init(start_node, goal_node)
            self._compute_shortest_path()
            compute_type = "full_init"
        else:
            # ---- 증분 재계산: goal 은 그대로, start 만 이동했거나 장애물만 바뀐 상태 ----
            if self._last_start_node is not None and self._last_start_node is not start_node:
                self._km += self._heuristic(self._last_start_node, start_node)
            self._s_start = start_node
            self._last_start_node = start_node
            self._compute_shortest_path()
            compute_type = "incremental"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_compute_time_ms = elapsed_ms
        self.last_compute_type = compute_type
        self.compute_log.append((compute_type, elapsed_ms))

        print(f"⏱️ [D*Lite] {compute_type} 계산 시간: {elapsed_ms:.3f} ms "
              f"(open_heap size={len(self._open_heap)})")

        if start_node.g == INF:
            return []  # 경로 없음

        return self._reconstruct_path(start_node, goal_node)

    def _full_init(self, start_node: _DNode, goal_node: _DNode) -> None:
        """Initialize() : g/rhs 전체 리셋 후 goal 만 rhs=0 으로 큐에 등록"""
        for col in self._grid:
            for node in col:
                node.g = INF
                node.rhs = INF
                node.key = None

        self._open_heap = []
        self._counter = itertools.count()
        self._km = 0.0
        self._s_start = start_node
        self._s_goal = goal_node
        self._goal_world = (goal_node.ix, goal_node.iz)
        self._last_start_node = start_node
        self._initialized = True

        goal_node.rhs = 0.0
        self._push(goal_node)

    def _reconstruct_path(self, start_node: _DNode, goal_node: _DNode) -> List[Tuple[float, float]]:
        """
        start 에서 시작해서, 매 스텝 "이동 비용 + g" 가 가장 작은 이웃을 따라
        goal 까지 그리디하게 내려가며 경로를 만듦 (g 값이 이미 goal 로부터의
        최단거리이므로, 이렇게만 따라가도 최단 경로가 됨).
        """
        path: List[Tuple[float, float]] = [self.grid_index_to_world(start_node.ix, start_node.iz)]
        current = start_node
        visited = {current}
        max_steps = self.grid_size_x * self.grid_size_z + 4

        steps = 0
        while current is not goal_node and steps < max_steps:
            best_node = None
            best_val = INF
            for nb in self._neighbors(current):
                cost = self._edge_cost(current, nb)
                if cost >= INF:
                    continue
                val = cost + nb.g
                if val < best_val:
                    best_val = val
                    best_node = nb
            if best_node is None or best_val == INF or best_node in visited:
                return []  # 경로 없음 / 루프 감지
            current = best_node
            visited.add(current)
            path.append(self.grid_index_to_world(current.ix, current.iz))
            steps += 1

        if current is not goal_node:
            return []
        return path

    def reset(self) -> None:
        """
        에피소드 재시작 등으로 탐색 상태를 완전히 초기화하고 싶을 때 호출.
        (server_sample 의 reset_planning_state() 에서 함께 호출하면 됨)
        장애물 목록/그리드 자체는 유지하고 g/rhs/큐/초기화 플래그만 리셋.
        """
        self._open_heap = []
        self._counter = itertools.count()
        self._km = 0.0
        self._s_start = None
        self._s_goal = None
        self._goal_world = None
        self._last_start_node = None
        self._initialized = False
        if self._grid_built:
            for col in self._grid:
                for node in col:
                    node.g = INF
                    node.rhs = INF
                    node.key = None

    # ------------------------------------------------------------------
    # 시각화 (AStarPlanner.plot() 과 동일한 시그니처)
    # ------------------------------------------------------------------
    def plot(
        self,
        path: Optional[List[Tuple[float, float]]] = None,
        show_grid: bool = True,
        figsize: Tuple[int, int] = (6, 6),
        title: Optional[str] = None,
    ) -> None:
        if not _HAS_MPL:  # pragma: no cover
            return
        if not self._grid_built:
            self._build_grid()

        fig, ax = plt.subplots(figsize=figsize)

        for obs in self._obstacles:
            x_min = obs.x_min - self.obstacle_margin
            x_max = obs.x_max + self.obstacle_margin
            z_min = obs.z_min - self.obstacle_margin
            z_max = obs.z_max + self.obstacle_margin
            w = x_max - x_min
            h = z_max - z_min
            rect = plt.Rectangle((x_min, z_min), w, h, alpha=0.3)
            ax.add_patch(rect)

        if show_grid:
            xs = [self.grid_min_x + i * self.cell_size for i in range(self.grid_size_x + 1)]
            zs = [self.grid_min_z + i * self.cell_size for i in range(self.grid_size_z + 1)]
            for x in xs:
                ax.axvline(x, linewidth=0.3)
            for z in zs:
                ax.axhline(z, linewidth=0.3)

        if path:
            xs = [p[0] for p in path]
            zs = [p[1] for p in path]
            ax.plot(xs, zs, marker="o")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.grid_min_x, self.grid_max_x)
        ax.set_ylim(self.grid_min_z, self.grid_max_z)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 간단한 자체 테스트 + (astar_planner.py 가 있을 경우) A* 대비 재계산 속도 비교 데모
    # astar_planner.py 가 삭제된 뒤에도 이 파일은 단독으로 실행 가능하도록
    # AStarPlanner import 는 선택적으로만 시도한다.
    try:
        from astar_planner import AStarPlanner
        _HAS_ASTAR = True
    except ImportError:
        _HAS_ASTAR = False
        print("[안내] astar_planner.py 를 찾을 수 없어 A* 비교 없이 D* Lite만 테스트합니다.\n")

    GRID_KW = dict(
        grid_min_x=0.0, grid_max_x=300.0,
        grid_min_z=0.0, grid_max_z=300.0,
        cell_size=1.0, obstacle_margin=2.0, allow_diagonal=True,
    )

    obs = ObstacleRect.from_min_max(x_min=120.0, x_max=180.0, z_min=90.0, z_max=210.0)
    start = (10.0, 150.0)
    goal = (290.0, 150.0)

    print("=== 최초 1회 계산 (둘 다 그리드 전체를 훑는 baseline 조건) ===")
    if _HAS_ASTAR:
        astar = AStarPlanner(**GRID_KW)
        astar.set_obstacles([obs])
        t0 = time.perf_counter()
        a_path = astar.find_path(start, goal)
        a_ms = (time.perf_counter() - t0) * 1000.0
        print(f"A*      : {a_ms:.3f} ms, path len={len(a_path)}")

    dstar = DStarLitePlanner(**GRID_KW)
    dstar.set_obstacles([obs])
    d_path = dstar.find_path(start, goal)
    print(f"D* Lite : path len={len(d_path)}")

    print("\n=== 탱크가 한 칸 이동한 뒤 재계산 (증분 vs A* 풀 재계산) ===")
    moved_start = (11.0, 150.0)

    if _HAS_ASTAR:
        t0 = time.perf_counter()
        a_path2 = astar.find_path(moved_start, goal)
        a_ms2 = (time.perf_counter() - t0) * 1000.0
        print(f"A* (재계산)      : {a_ms2:.3f} ms, path len={len(a_path2)}")

    d_path2 = dstar.find_path(moved_start, goal)
    print(f"D* Lite (증분)   : {dstar.last_compute_time_ms:.3f} ms, path len={len(d_path2)}")

    print("\n=== 장애물 하나 추가 후 재계산 ===")
    obs2 = ObstacleRect.from_min_max(x_min=200.0, x_max=220.0, z_min=140.0, z_max=160.0)

    if _HAS_ASTAR:
        t0 = time.perf_counter()
        astar.set_obstacles([obs, obs2])
        a_path3 = astar.find_path(moved_start, goal)
        a_ms3 = (time.perf_counter() - t0) * 1000.0
        print(f"A* (재계산)      : {a_ms3:.3f} ms, path len={len(a_path3)}")

    dstar.set_obstacles([obs, obs2])
    d_path3 = dstar.find_path(moved_start, goal)
    print(f"D* Lite (증분)   : {dstar.last_compute_time_ms:.3f} ms, path len={len(d_path3)}")