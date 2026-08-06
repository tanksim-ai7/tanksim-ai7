"""
astar_planner.py

A* 기반 2D(XZ) 경로 탐색 + 시각화 클래스
--------------------------------------
- tracking mode 에서 사용하는 Flask 서버 코드에서 import 해서 사용하기 위한 용도
- /set_destination API 로 받은 위치까지 최단 경로 계산
- /update_obstacle API 로 받은 장애물(x_min, x_max, z_min, z_max) 정보 사용
- 전차의 크기를 고려한 margin(기본 2.0) 적용
- 필요 시 matplotlib 로 장애물 + 경로 시각화

[성능 개선]
- open_set을 리스트(선형 탐색, O(n))가 아니라 heapq(우선순위 큐, O(log n))로 관리하도록 변경.
  기존 방식은 목적지가 멀거나 그리드가 클 때 min(open_set, ...)이 매 스텝마다 전체를 훑어야 해서
  느려짐(체감상 최악의 경우 수십 초까지 걸림). heapq는 항상 최소 f_cost 노드를 O(log n)에 꺼낸다.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple

import math

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


class _Node:
    """내부용 A* 노드 구조체 (grid index + 비용 정보)"""

    __slots__ = (
        "ix", "iz", "walkable",
        "g_cost", "h_cost", "parent",
        "in_open", "in_closed",
    )

    def __init__(self, ix: int, iz: int, walkable: bool):
        self.ix = ix
        self.iz = iz
        self.walkable = walkable
        self.g_cost: int = 0
        self.h_cost: int = 0
        self.parent: Optional["_Node"] = None
        # heapq 방식에서는 "이미 open_set에 들어갔던 적 있는지 / closed 되었는지"를
        # 별도 플래그로 관리한다 (heapq는 리스트 안의 특정 원소를 지우거나 갱신할 수 없기 때문)
        self.in_open: bool = False
        self.in_closed: bool = False

    @property
    def f_cost(self) -> int:
        return self.g_cost + self.h_cost


class AStarPlanner:
    """
    A* 경로 탐색 + 시각화 클래스

    - grid_min_x ~ grid_max_x, grid_min_z ~ grid_max_z 범위 안을 cell_size 로 자른 2D 그리드를 구성
    - 장애물 + obstacle_margin 을 고려해서 walkable / blocked 셀 판정
    - find_path() 로 시작점(start) ~ 목적지(goal) 사이의 최단 경로 계산
    - plot() 으로 장애물 + 경로를 matplotlib 으로 시각화 가능

    좌표계:
        - Unity 상의 X / Z 를 그대로 사용한다고 가정
        - (x, z) 튜플을 월드 좌표처럼 사용
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

        # 그리드 해상도(셀 개수)
        self.grid_size_x = max(1, int(math.ceil((self.grid_max_x - self.grid_min_x) / self.cell_size)))
        self.grid_size_z = max(1, int(math.ceil((self.grid_max_z - self.grid_min_z) / self.cell_size)))

        # 장애물 리스트
        self._obstacles: List[ObstacleRect] = []

        # 노드 그리드 (lazy build)
        self._grid: List[List[_Node]] = []
        self._grid_valid: bool = False

    # ------------------------------------------------------------------
    # 장애물 & 그리드
    # ------------------------------------------------------------------
    def set_obstacles(self, obstacles: Iterable[ObstacleRect]) -> None:
        """장애물 리스트를 설정하고, 그리드를 다시 빌드하도록 플래그 표시"""
        self._obstacles = list(obstacles)
        self._grid_valid = False

    def _build_grid(self) -> None:
        """
        장애물 + margin 을 고려하여 walkable 정보를 포함한 그리드 초기화

        [성능 개선] 기존 방식은 "모든 칸 x 모든 장애물"을 전부 비교했음
        (grid_size_x * grid_size_z * len(obstacles) 만큼의 연산, 예: 90000 * 350 ≈ 3천만 번).
        장애물 하나는 그리드 전체 중 아주 작은 영역만 차지하므로, 반대로
        "장애물 기준으로 자신이 덮는 칸 범위만 계산해서 칠하는" 래스터화(rasterize) 방식으로 바꿈.
        전체 칸을 먼저 walkable=True로 깔아두고, 장애물마다 자신이 덮는 (ix, iz) 사각 범위만
        blocked로 표시 -> 연산량이 장애물이 실제로 덮는 칸 수에 비례하게 되어 훨씬 빨라짐.
        """
        # 1) 모든 칸을 walkable=True로 초기화 (아직 아무 장애물도 반영 안 된 상태)
        self._grid = [
            [_Node(ix, iz, True) for iz in range(self.grid_size_z)]
            for ix in range(self.grid_size_x)
        ]

        # 2) 장애물마다 자신이 덮는 그리드 인덱스 범위만 계산해서 walkable=False로 칠하기
        for obs in self._obstacles:
            x_min = obs.x_min - self.obstacle_margin
            x_max = obs.x_max + self.obstacle_margin
            z_min = obs.z_min - self.obstacle_margin
            z_max = obs.z_max + self.obstacle_margin

            # 월드 좌표 범위 -> 그리드 인덱스 범위로 변환 (그리드 밖으로 나가면 클램핑)
            ix_min = max(0, int(math.floor((x_min - self.grid_min_x) / self.cell_size)))
            ix_max = min(self.grid_size_x - 1, int(math.floor((x_max - self.grid_min_x) / self.cell_size)))
            iz_min = max(0, int(math.floor((z_min - self.grid_min_z) / self.cell_size)))
            iz_max = min(self.grid_size_z - 1, int(math.floor((z_max - self.grid_min_z) / self.cell_size)))

            if ix_min > ix_max or iz_min > iz_max:
                continue  # 그리드 범위와 아예 안 겹치는 장애물

            for ix in range(ix_min, ix_max + 1):
                col = self._grid[ix]
                for iz in range(iz_min, iz_max + 1):
                    col[iz].walkable = False

        self._grid_valid = True

    def _is_blocked(self, x: float, z: float) -> bool:
        """
        (x, z) 위치가 장애물 + margin 영역 안에 있으면 True
        - 전차의 반경 + 여유를 obstacle_margin 으로 보고,
          사각형 장애물의 x/z min/max 를 margin 만큼 확장해서 충돌 판정
        """
        for obs in self._obstacles:
            x_min = obs.x_min - self.obstacle_margin
            x_max = obs.x_max + self.obstacle_margin
            z_min = obs.z_min - self.obstacle_margin
            z_max = obs.z_max + self.obstacle_margin
            if x_min <= x <= x_max and z_min <= z <= z_max:
                return True
        return False

    # ------------------------------------------------------------------
    # 좌표 변환
    # ------------------------------------------------------------------
    def world_to_grid(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        """
        월드 좌표 (x, z)를 그리드 index (ix, iz) 로 변환.
        그리드 범위 밖이면 None 반환.

        * 입력 좌표는 소수점 둘째자리까지 반올림하여 사용.
        """
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
        """
        그리드 index (ix, iz)를 셀 중앙의 월드 좌표 (x, z) 로 변환.

        * 반환 좌표는 소수점 둘째자리까지 반올림하여 반환.
        """
        x = self.grid_min_x + (ix + 0.5) * self.cell_size
        z = self.grid_min_z + (iz + 0.5) * self.cell_size
        return round(x, 2), round(z, 2)

    # ------------------------------------------------------------------
    # A* 핵심 로직 (heapq 기반)
    # ------------------------------------------------------------------
    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """
        A* 알고리즘으로 start (x, z) -> goal (x, z) 최단 경로를 계산해서
        월드 좌표 리스트 [(x1, z1), (x2, z2), ...] 형태로 반환.
        """
        if not self._grid_valid:
            self._build_grid()

        start_idx = self.world_to_grid(*start)
        goal_idx = self.world_to_grid(*goal)
        if start_idx is None or goal_idx is None:
            return []

        sx, sz = start_idx
        gx, gz = goal_idx

        start_node = self._grid[sx][sz]
        goal_node = self._grid[gx][gz]

        # 시작점/목적지가 장애물(+margin) 위에 걸려 있으면 바로 실패시키지 않고,
        # 가장 가까운 walkable 칸을 찾아서 대신 사용.
        # (실제 사례: 지뢰 폭발 등으로 탱크가 장애물 경계 쪽으로 밀려나면 시작점 자체가
        #  막힌 칸이 되어버려서, 목적지를 바꿔도 항상 "경로 없음"이 나오는 문제가 있었음)
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

        # g/h/parent/open/closed 상태 초기화
        for ix in range(self.grid_size_x):
            for iz in range(self.grid_size_z):
                node = self._grid[ix][iz]
                node.g_cost = 0
                node.h_cost = 0
                node.parent = None
                node.in_open = False
                node.in_closed = False

        # heapq는 (우선순위, tie-break, 노드) 튜플로 관리.
        # tie-break용 카운터를 두는 이유: _Node 객체끼리는 <, > 비교가 정의되어 있지 않아서
        # f_cost가 동점일 때 heapq가 두 번째 요소(_Node)를 직접 비교하려 하면 에러가 남.
        # counter를 두 번째 요소로 넣으면 항상 숫자끼리 먼저 비교되어 이 문제를 피할 수 있음.
        counter = itertools.count()
        open_heap: List[Tuple[int, int, _Node]] = []

        start_node.in_open = True
        heapq.heappush(open_heap, (start_node.f_cost, next(counter), start_node))

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current.in_closed:
                # 같은 노드가 갱신되면서 여러 번 push 됐을 수 있음(오래된 항목) -> 건너뜀
                continue

            if current is goal_node:
                return self._reconstruct_path(start_node, goal_node)

            current.in_open = False
            current.in_closed = True

            for neighbor in self._neighbors(current):
                if not neighbor.walkable or neighbor.in_closed:
                    continue

                new_g = current.g_cost + self._distance_cost(current, neighbor)
                if new_g < neighbor.g_cost or not neighbor.in_open:
                    neighbor.g_cost = new_g
                    neighbor.h_cost = self._distance_cost(neighbor, goal_node)
                    neighbor.parent = current
                    neighbor.in_open = True
                    # 갱신된 값으로 새로 push (오래된 항목은 pop될 때 in_closed 체크로 자연히 무시됨)
                    heapq.heappush(open_heap, (neighbor.f_cost, next(counter), neighbor))

        # no path
        return []

    def _nearest_walkable(self, ix: int, iz: int, max_radius: int = 25) -> Optional[_Node]:
        """
        (ix, iz)가 막힌 칸일 때, 그 주변에서 가장 가까운 walkable 칸을 찾아서 반환.
        사각형 링(ring) 모양으로 반경을 1칸씩 넓혀가며 탐색 -> 가장 가까운 것을 우선 발견.
        max_radius 안에서 못 찾으면 None.
        """
        if self._grid[ix][iz].walkable:
            return self._grid[ix][iz]

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # 링의 테두리만 검사 (이전 radius에서 이미 검사한 안쪽은 건너뜀)
                    if max(abs(dx), abs(dz)) != radius:
                        continue
                    nx, nz = ix + dx, iz + dz
                    if 0 <= nx < self.grid_size_x and 0 <= nz < self.grid_size_z:
                        node = self._grid[nx][nz]
                        if node.walkable:
                            return node
        return None

    def _neighbors(self, node: _Node) -> Iterable[_Node]:
        """상하좌우(+대각선) 이웃 노드"""
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
    def _distance_cost(a: _Node, b: _Node) -> int:
        """
        A* 휴리스틱 및 이동 비용 계산용
        - 대각선 비용을 14, 직선 비용을 10 으로 두는 그리드 A* 전통 사용
        """
        dx = abs(a.ix - b.ix)
        dz = abs(a.iz - b.iz)
        diag = min(dx, dz)
        straight = abs(dx - dz)
        return 14 * diag + 10 * straight

    def _reconstruct_path(
        self,
        start_node: _Node,
        goal_node: _Node,
    ) -> List[Tuple[float, float]]:
        """goal 에서 parent 를 따라 start 까지 거슬러 올라간 뒤 월드 좌표 리스트로 반환"""
        path_nodes: List[_Node] = []
        cur: Optional[_Node] = goal_node

        while cur is not None and cur is not start_node:
            path_nodes.append(cur)
            cur = cur.parent
        if cur is start_node:
            path_nodes.append(start_node)

        path_nodes.reverse()
        world_path: List[Tuple[float, float]] = [
            self.grid_index_to_world(n.ix, n.iz) for n in path_nodes
        ]
        return world_path

    # ------------------------------------------------------------------
    # 시각화 (교육용)
    # ------------------------------------------------------------------
    def plot(
        self,
        path: Optional[List[Tuple[float, float]]] = None,
        show_grid: bool = True,
        figsize: Tuple[int, int] = (6, 6),
        title: Optional[str] = None,
    ) -> None:
        """
        matplotlib 을 이용해 장애물 + 경로를 시각화.
        * matplotlib 이 설치되어 있지 않으면 아무 것도 하지 않음.
        """
        if not _HAS_MPL:  # pragma: no cover
            return

        if not self._grid_valid:
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
    import time

    planner = AStarPlanner(
        grid_min_x=0.0,
        grid_max_x=300.0,
        grid_min_z=0.0,
        grid_max_z=300.0,
        cell_size=1.0,
        obstacle_margin=2.0,
        allow_diagonal=True,
    )

    obs = ObstacleRect.from_min_max(x_min=120.0, x_max=180.0, z_min=90.0, z_max=210.0)
    planner.set_obstacles([obs])

    start = (10.0, 150.0)
    goal = (290.0, 150.0)

    t0 = time.time()
    path = planner.find_path(start, goal)
    print(f"Path length: {len(path)}, took {time.time() - t0:.3f}s")