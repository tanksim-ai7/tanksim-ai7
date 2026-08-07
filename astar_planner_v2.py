# -*- coding: utf-8 -*-
"""
astar_planner_v2.py

A* 기반 2D(XZ) 경로 탐색 - 성능 개선 + 비용 레이어 확장판

원본(astar_planner.py) 대비 변경점
  [성능]
    1. open_set 을 heapq 로 교체        (O(n) -> O(log n))
    2. 노드 전체 초기화 제거, dict 로 방문 노드만 관리
    3. numpy 벡터 연산으로 그리드 빌드
    4. 대각 이동 시 코너 컷팅 방지
  [기능]
    5. cost_layer 훅 추가  <- 은폐/피탐 위험 비용을 얹는 자리
    6. 경로 스무딩 (가시선 단축)
    7. start/goal 이 막혔을 때 가장 가까운 통행 가능 셀로 스냅

사용 예:
    planner = AStarPlanner(0, 300, 0, 300, cell_size=1.0, obstacle_margin=2.0)
    planner.set_obstacles(obs_list)

    # 기본(최단 경로)
    path = planner.find_path((10, 150), (290, 150))

    # 은폐 경로 - exposure[iz, ix] 가 클수록 위험한 지역
    planner.set_cost_layer(exposure, weight=8.0)
    path = planner.find_path((10, 150), (290, 150))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple

import heapq
import math

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


STRAIGHT = 10
DIAGONAL = 14


@dataclass
class ObstacleRect:
    """XZ 평면의 축 정렬 사각형 장애물"""
    center_x: float
    center_z: float
    size_x: float
    size_z: float

    @property
    def x_min(self) -> float: return self.center_x - self.size_x * 0.5
    @property
    def x_max(self) -> float: return self.center_x + self.size_x * 0.5
    @property
    def z_min(self) -> float: return self.center_z - self.size_z * 0.5
    @property
    def z_max(self) -> float: return self.center_z + self.size_z * 0.5

    @classmethod
    def from_min_max(cls, x_min, x_max, z_min, z_max) -> "ObstacleRect":
        return cls((x_min + x_max) * .5, (z_min + z_max) * .5,
                   x_max - x_min, z_max - z_min)


class AStarPlanner:
    def __init__(self, grid_min_x, grid_max_x, grid_min_z, grid_max_z,
                 cell_size=1.0, obstacle_margin=2.0, allow_diagonal=True):
        assert cell_size > 0
        self.grid_min_x = float(grid_min_x)
        self.grid_max_x = float(grid_max_x)
        self.grid_min_z = float(grid_min_z)
        self.grid_max_z = float(grid_max_z)
        self.cell_size = float(cell_size)
        self.obstacle_margin = float(obstacle_margin)
        self.allow_diagonal = bool(allow_diagonal)

        self.nx = max(1, int(math.ceil((self.grid_max_x - self.grid_min_x) / self.cell_size)))
        self.nz = max(1, int(math.ceil((self.grid_max_z - self.grid_min_z) / self.cell_size)))

        self._obstacles: List[ObstacleRect] = []
        # walkable[iz, ix] : bool
        self.walkable = np.ones((self.nz, self.nx), dtype=bool)
        self._grid_valid = True

        # 추가 비용 레이어 (은폐/피탐 위험 등)
        self._cost_layer: Optional[np.ndarray] = None
        self._cost_weight: float = 0.0

    # ------------------------------------------------------------------
    # 장애물 / 비용 레이어
    # ------------------------------------------------------------------
    def set_obstacles(self, obstacles: Iterable[ObstacleRect]) -> None:
        self._obstacles = list(obstacles)
        self._grid_valid = False

    def set_cost_layer(self, layer: Optional[np.ndarray], weight: float = 1.0) -> None:
        """
        추가 비용 레이어 등록.
          layer  : (nz, nx) 배열. 0~1 로 정규화된 '통과하기 싫은 정도'
                   예) 적 가시선에 노출되는 정도, 개활지 정도
          weight : 거리 비용 대비 가중치. 0 이면 순수 최단 경로

        비용 = STRAIGHT/DIAGONAL(거리) + weight * STRAIGHT * layer[목적셀]
        """
        if layer is not None:
            layer = np.asarray(layer, dtype=float)
            if layer.shape != (self.nz, self.nx):
                raise ValueError(f"cost_layer shape {layer.shape} != {(self.nz, self.nx)}")
        self._cost_layer = layer
        self._cost_weight = float(weight)

    def _build_grid(self) -> None:
        """numpy 벡터 연산으로 통행 가능 여부 일괄 계산"""
        self.walkable = np.ones((self.nz, self.nx), dtype=bool)
        if not self._obstacles:
            self._grid_valid = True
            return

        xs = self.grid_min_x + (np.arange(self.nx) + 0.5) * self.cell_size
        zs = self.grid_min_z + (np.arange(self.nz) + 0.5) * self.cell_size
        m = self.obstacle_margin

        for o in self._obstacles:
            ix = (xs >= o.x_min - m) & (xs <= o.x_max + m)
            iz = (zs >= o.z_min - m) & (zs <= o.z_max + m)
            if ix.any() and iz.any():
                self.walkable[np.ix_(iz, ix)] = False
        self._grid_valid = True

    # ------------------------------------------------------------------
    # 좌표 변환
    # ------------------------------------------------------------------
    def world_to_grid(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        ix = int(math.floor((x - self.grid_min_x) / self.cell_size))
        iz = int(math.floor((z - self.grid_min_z) / self.cell_size))
        if 0 <= ix < self.nx and 0 <= iz < self.nz:
            return ix, iz
        return None

    def grid_index_to_world(self, ix: int, iz: int) -> Tuple[float, float]:
        return (round(self.grid_min_x + (ix + 0.5) * self.cell_size, 2),
                round(self.grid_min_z + (iz + 0.5) * self.cell_size, 2))

    def _snap_to_walkable(self, ix: int, iz: int, max_r: int = 30) -> Optional[Tuple[int, int]]:
        """막힌 셀이면 가장 가까운 통행 가능 셀을 찾아 반환"""
        if self.walkable[iz, ix]:
            return ix, iz
        for r in range(1, max_r + 1):
            x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
            z0, z1 = max(0, iz - r), min(self.nz, iz + r + 1)
            sub = self.walkable[z0:z1, x0:x1]
            if sub.any():
                zz, xx = np.nonzero(sub)
                d = (xx + x0 - ix) ** 2 + (zz + z0 - iz) ** 2
                k = int(np.argmin(d))
                return int(xx[k] + x0), int(zz[k] + z0)
        return None

    # ------------------------------------------------------------------
    # A*
    # ------------------------------------------------------------------
    @staticmethod
    def _h(ix, iz, gx, gz) -> int:
        """옥타일 거리 휴리스틱 (admissible)"""
        dx, dz = abs(ix - gx), abs(iz - gz)
        return DIAGONAL * min(dx, dz) + STRAIGHT * (abs(dx - dz))

    def find_path(self, start, goal, smooth: bool = True) -> List[Tuple[float, float]]:
        if not self._grid_valid:
            self._build_grid()

        s = self.world_to_grid(*start)
        g = self.world_to_grid(*goal)
        if s is None or g is None:
            return []
        s = self._snap_to_walkable(*s)
        g = self._snap_to_walkable(*g)
        if s is None or g is None:
            return []

        sx, sz = s
        gx, gz = g
        if (sx, sz) == (gx, gz):
            return [self.grid_index_to_world(gx, gz)]

        layer = self._cost_layer
        w = self._cost_weight
        walk = self.walkable
        nx, nz = self.nx, self.nz

        if self.allow_diagonal:
            nbrs = ((1,0,STRAIGHT), (-1,0,STRAIGHT), (0,1,STRAIGHT), (0,-1,STRAIGHT),
                    (1,1,DIAGONAL), (1,-1,DIAGONAL), (-1,1,DIAGONAL), (-1,-1,DIAGONAL))
        else:
            nbrs = ((1,0,STRAIGHT), (-1,0,STRAIGHT), (0,1,STRAIGHT), (0,-1,STRAIGHT))

        start_key = sz * nx + sx
        goal_key = gz * nx + gx

        g_score = {start_key: 0}
        parent = {}
        closed = set()
        open_heap = [(self._h(sx, sz, gx, gz), start_key)]

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal_key:
                return self._reconstruct(parent, start_key, goal_key, smooth)
            closed.add(cur)

            cz, cx = divmod(cur, nx)
            cg = g_score[cur]

            for dx, dz, step in nbrs:
                ax, az = cx + dx, cz + dz
                if not (0 <= ax < nx and 0 <= az < nz):
                    continue
                if not walk[az, ax]:
                    continue
                # 대각 코너 컷팅 방지 - 양옆이 모두 막히면 통과 불가
                if dx and dz:
                    if not walk[cz, ax] or not walk[az, cx]:
                        continue
                nk = az * nx + ax
                if nk in closed:
                    continue

                cost = step
                if layer is not None and w:
                    cost += w * STRAIGHT * layer[az, ax]

                ng = cg + cost
                if ng < g_score.get(nk, float("inf")):
                    g_score[nk] = ng
                    parent[nk] = cur
                    heapq.heappush(open_heap, (ng + self._h(ax, az, gx, gz), nk))

        return []

    def _reconstruct(self, parent, start_key, goal_key, smooth) -> List[Tuple[float, float]]:
        nx = self.nx
        keys = [goal_key]
        while keys[-1] != start_key:
            keys.append(parent[keys[-1]])
        keys.reverse()
        cells = [(k % nx, k // nx) for k in keys]
        if smooth:
            cells = self._smooth(cells)
        return [self.grid_index_to_world(ix, iz) for ix, iz in cells]

    # ------------------------------------------------------------------
    # 경로 스무딩 - 가시선이 통하면 중간 waypoint 를 건너뛴다
    # ------------------------------------------------------------------
    def _line_clear(self, a, b) -> bool:
        """두 셀 사이가 모두 통행 가능한지 (Bresenham)"""
        x0, z0 = a
        x1, z1 = b
        dx, dz = abs(x1 - x0), abs(z1 - z0)
        sx = 1 if x0 < x1 else -1
        sz = 1 if z0 < z1 else -1
        err = dx - dz
        walk = self.walkable
        while True:
            if not walk[z0, x0]:
                return False
            if (x0, z0) == (x1, z1):
                return True
            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x0 += sx
            if e2 < dx:
                err += dx
                z0 += sz

    def _smooth(self, cells):
        """
        직선으로 갈 수 있는 구간은 중간점을 제거.
        281개 waypoint -> 수 개로 줄어 제어가 훨씬 쉬워진다.
        * 비용 레이어를 쓸 때는 우회 경로가 펴질 수 있으므로 주의
        """
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_clear(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    # ------------------------------------------------------------------
    # 시각화
    # ------------------------------------------------------------------
    def plot(self, path=None, figsize=(7, 7), title=None, show_cost=True, ax=None):
        if not _HAS_MPL:
            return
        if not self._grid_valid:
            self._build_grid()

        own = ax is None
        if own:
            _, ax = plt.subplots(figsize=figsize)

        ext = [self.grid_min_x, self.grid_max_x, self.grid_min_z, self.grid_max_z]
        if show_cost and self._cost_layer is not None:
            ax.imshow(self._cost_layer, origin="lower", extent=ext,
                      cmap="Reds", alpha=.75, vmin=0, vmax=1)
        blocked = np.ma.masked_where(self.walkable, ~self.walkable)
        ax.imshow(blocked, origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1)

        if path:
            xs = [p[0] for p in path]
            zs = [p[1] for p in path]
            ax.plot(xs, zs, "-o", color="#2f81f7", ms=3, lw=1.8)
            ax.plot(xs[0], zs[0], "o", color="lime", ms=11, label="start")
            ax.plot(xs[-1], zs[-1], "*", color="gold", ms=17, label="goal")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_aspect("equal")
        ax.set_xlim(self.grid_min_x, self.grid_max_x)
        ax.set_ylim(self.grid_min_z, self.grid_max_z)
        ax.set_xlabel("X"); ax.set_ylabel("Z")
        if title:
            ax.set_title(title)
        if own:
            plt.tight_layout()
        return ax


if __name__ == "__main__":
    import time

    planner = AStarPlanner(0, 300, 0, 300, cell_size=1.0, obstacle_margin=2.0)
    planner.set_obstacles([ObstacleRect.from_min_max(120, 180, 90, 210)])

    t = time.time()
    path = planner.find_path((10., 150.), (290., 150.))
    print(f"탐색 {time.time()-t:.4f}초, waypoint {len(path)}개")
    print(" ", path)
