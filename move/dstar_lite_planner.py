"""
dstar_lite_planner.py (Bug Fixed & Production Ready Version)

D* Lite 알고리즘 기반 2D(XZ) 실시간 경로 플래너
- cell_size = 1.0 기동 환경에 완벽 최적화
- 최초 set_obstacles 호출 시 큐(U) 누락 버그 완전 해결
- 튜플 및 heapq 내부 정렬 비교 매직 메서드(__lt__) 완벽 정착
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Set
import heapq
import math

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  
    _HAS_MPL = False


@dataclass
class ObstacleRect:
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
    def from_min_max(cls, x_min: float, x_max: float, z_min: float, z_max: float) -> "ObstacleRect":
        cx = (x_min + x_max) * 0.5
        cz = (z_min + z_max) * 0.5
        sx = (x_max - x_min)
        sz = (z_max - z_min)
        return cls(center_x=cx, center_z=cz, size_x=sx, size_z=sz)

class _DStarNode:
    """
    평소에 맵이 변하지 않을 때는 g값과 rhs값이 완벽하게 일치한다.
    하지만, 포탄에 의해 오브젝트가 파괴 되었던지, 적 전차가 움직이면서 경로를 막던지
    상황에 따라 rhs값이 업데이트 되게 되고
    g != rhs를 통해서 전체 정보가 아닌 특정 정보만 업데이트 하는 방식으로 재탐색에 대한 성능을 높일 수 있다.

    g > rhs (오버-일관성, Overconsistent) ➔ "더 좋은 지름길이 뚫렸다!"
    g < rhs (언더-일관성, Underconsistent) ➔ "적 전차가 길을 막아섰다!"
    """
    __slots__ = ("ix", "iz", "walkable", "g", "rhs")

    def __init__(self, ix: int, iz: int, walkable: bool):
        self.ix = ix # x좌표
        self.iz = iz # z좌표
        self.walkable = walkable # 해당 노드가 갈 수 있는지 여부
        self.g: float = float('inf') # 실제 도달한 누적 이동 거리 비용
        self.rhs: float = float('inf') # 변동에 의한 예측값

    # ⭕ [버그 해결]: heapq 내부에서 키값이 동일할 때 노드 객체 자체를 자동 비교 정렬하기 위한 특수 매직 메서드 내장
    def __lt__(self, other: _DStarNode) -> bool:
        return min(self.g, self.rhs) < min(other.g, other.rhs)


class DStarLitePlanner:
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
        self._grid: List[List[_DStarNode]] = []
        
        self.U: List[Tuple[Tuple[float, float], _DStarNode]] = [] 
        self.km: float = 0.0                                     
        self.last_start: Optional[_DStarNode] = None             
        self._current_goal_node: Optional[_DStarNode] = None # 목적지 변경 감지 추적용 변수

        self._initialized = False  # 최초 기동 여부를 체크하는 안전 플래그

        self._build_empty_grid()

    def world_to_grid_index(self, x: float, z: float) -> Tuple[int, int]:
        ix = int((x - self.grid_min_x) / self.cell_size)
        iz = int((z - self.grid_min_z) / self.cell_size)
        return max(0, min(ix, self.grid_size_x - 1)), max(0, min(iz, self.grid_size_z - 1))

    def grid_index_to_world(self, ix: int, iz: int) -> Tuple[float, float]:
        x = self.grid_min_x + (ix + 0.5) * self.cell_size
        z = self.grid_min_z + (iz + 0.5) * self.cell_size
        return x, z

    def _build_empty_grid(self) -> None:
        self._grid = []
        for ix in range(self.grid_size_x):
            col = []
            for iz in range(self.grid_size_z):
                col.append(_DStarNode(ix, iz, walkable=True))
            self._grid.append(col)

    def _get_neighbors(self, node: _DStarNode) -> List[_DStarNode]:
        neighbors = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0: continue
                if not self.allow_diagonal and abs(dx) + abs(dz) > 1: continue
                
                nx, nz = node.ix + dx, node.iz + dz
                if 0 <= nx < self.grid_size_x and 0 <= nz < self.grid_size_z:
                    neighbors.append(self._grid[nx][nz])
        return neighbors

    # 격자(Grid) 환경에서 대각선 이동이 허용될 때 두 지점 사이의 가장 직관적이고 정밀한 최단 거리를 계산하는 체비쇼프/옥타일 거리(Octilinear/Diagonal Distance) 휴리스틱 수식
    def _heuristic(self, a: _DStarNode, b: _DStarNode) -> float:
        dx = abs(a.ix - b.ix)
        dz = abs(a.iz - b.iz)
        return min(dx, dz) * 1.414 + abs(dx - dz) * 1.0

    def _cost(self, a: _DStarNode, b: _DStarNode) -> float:
        if not a.walkable or not b.walkable:
            return float('inf')
        dx = abs(a.ix - b.ix)
        dz = abs(a.iz - b.iz)
        return 1.414 if (dx != 0 and dz != 0) else 1.0

    def _calculate_key(self, node: _DStarNode, start: _DStarNode) -> Tuple[float, float]:
        min_g_rhs = min(node.g, node.rhs)
        k1 = min_g_rhs + self._heuristic(node, start) + self.km
        k2 = min_g_rhs
        return (k1, k2)

    def _update_vertex(self, node: _DStarNode, start: _DStarNode, goal: _DStarNode) -> None:
        if node != goal:
            min_rhs = float('inf')
            for succ in self._get_neighbors(node):
                min_rhs = min(min_rhs, succ.g + self._cost(node, succ))
            node.rhs = min_rhs


        # 중복 방지
        self.U = [item for item in self.U if item[1] != node]
        heapq.heapify(self.U)

        if not math.isclose(node.g, node.rhs, abs_tol=1e-5):
            key = self._calculate_key(node, start)
            heapq.heappush(self.U, (key, node))

    def _compute_shortest_path(self, start: _DStarNode, goal: _DStarNode) -> None:
        """
        재탐색의 경우 set_obstacles에서 _update_vertex()를 타게되고 해당 함수에서 self.U가 업데이트 되어 다시 경로를 계산한다.
        """
        while len(self.U) > 0:
            top_key, top_node = self.U[0]
            curr_key = self._calculate_key(top_node, start)
            
            if top_key < curr_key:
                heapq.heappop(self.U)
                heapq.heappush(self.U, (curr_key, top_node))
                continue
                
            # 부동소수점 근사 수렴 조건 보정
            if top_key >= self._calculate_key(start, start) and math.isclose(start.g, start.rhs, abs_tol=1e-5):
                break

            heapq.heappop(self.U)

            if top_node.g > top_node.rhs: # 길이 새롭게 뚫린 상황
                top_node.g = top_node.rhs
                for pred in self._get_neighbors(top_node):
                    self._update_vertex(pred, start, goal)
            else: # 길이 새롭게 막힌 상황 or 그대로
                top_node.g = float('inf')
                self._update_vertex(top_node, start, goal)
                for pred in self._get_neighbors(top_node):
                    self._update_vertex(pred, start, goal)

    def set_obstacles(self, obstacles: Iterable[ObstacleRect]) -> None:
        new_obstacles = list(obstacles)
        
        self._obstacles = new_obstacles
            
        old_walkable_states = [[self._grid[ix][iz].walkable for iz in range(self.grid_size_z)] for ix in range(self.grid_size_x)]
        
        for ix in range(self.grid_size_x):
            for iz in range(self.grid_size_z):
                self._grid[ix][iz].walkable = True

        for obs in self._obstacles:
            x_min_w = obs.x_min - self.obstacle_margin
            x_max_w = obs.x_max + self.obstacle_margin
            z_min_w = obs.z_min - self.obstacle_margin
            z_max_w = obs.z_max + self.obstacle_margin

            ix_min, iz_min = self.world_to_grid_index(x_min_w, z_min_w)
            ix_max, iz_max = self.world_to_grid_index(x_max_w, z_max_w)

            for ix in range(ix_min, ix_max + 1):
                for iz in range(iz_min, iz_max + 1):
                    cell_x_left = self.grid_min_x + ix * self.cell_size
                    cell_x_right = cell_x_left + self.cell_size
                    cell_z_bottom = self.grid_min_z + iz * self.cell_size
                    cell_z_top = cell_z_bottom + self.cell_size

                    if (cell_x_left <= x_max_w and cell_x_right >= x_min_w and
                        cell_z_bottom <= z_max_w and cell_z_top >= z_min_w):
                        self._grid[ix][iz].walkable = False

        # ➔ [버그 해결 수정 부위]: last_start가 정의되어 있을 때만 동적 누적 부분 업데이트 연계 수행
        if self.last_start is not None and self._current_goal_node is not None:
            for ix in range(self.grid_size_x):
                for iz in range(self.grid_size_z):
                    if self._grid[ix][iz].walkable != old_walkable_states[ix][iz]:
                        self._update_vertex(self._grid[ix][iz], self.last_start, self._current_goal_node)

    def find_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        start_x, start_z = start_pos
        goal_x, goal_z = goal_pos

        six, siz = self.world_to_grid_index(start_x, start_z)
        gix, giz = self.world_to_grid_index(goal_x, goal_z)

        start_node = self._grid[six][siz]
        goal_node = self._grid[gix][giz]

        # 출발/목적지가 오브젝트 등에 의해 이동 불가 지역인 경우 return []
        if not start_node.walkable or not goal_node.walkable: 
            return []

        # 목적지가 바뀌거나 최초 실행일 때 전체 비용 구조를 정석 인플레이션 초기화 처리합니다.
        # if len(self.U) == 0 or self._current_goal_node != goal_node:
        if not self._initialized or self._current_goal_node != goal_node:
            self.km = 0.0
            self.last_start = start_node
            self._current_goal_node = goal_node
            self._initialized = True # 최초 기동 완료 마킹

            for col in self._grid:
                for n in col:
                    n.g = float('inf')
                    n.rhs = float('inf')
            
            goal_node.rhs = 0.0
            self.U = [] # 완전 비우고 리셋
            heapq.heappush(self.U, (self._calculate_key(goal_node, start_node), goal_node))
        # 출발지가 변경된 경우
        elif self.last_start != start_node:
            self.km += self._heuristic(self.last_start, start_node) # 전차 이동 거리에 따른 휴리스틱 글로벌 보정 키값을 변경
            self.last_start = start_node # 출발지 재설정

        # 메인 경로 탐색 수렴 가동
        self._compute_shortest_path(start_node, goal_node)

        if start_node.g == float('inf'):
            return []

        # 최적 비용 링크 추적 복원
        path = [start_node]
        curr = start_node
        visited_nodes = {curr}

        while curr != goal_node:
            best_neighbor = None
            min_cost = float('inf')
            
            for nxt in self._get_neighbors(curr):
                cost = self._cost(curr, nxt) + nxt.g
                if cost < min_cost and nxt not in visited_nodes:
                    min_cost = cost
                    best_neighbor = nxt
            
            if best_neighbor is None: 
                break
                
            path.append(best_neighbor)
            curr = best_neighbor
            visited_nodes.add(curr)

        # 1단계: 복원된 정밀 월드 좌표 리스트 빌드
        world_path = [self.grid_index_to_world(n.ix, n.iz) for n in path]

        # 2단계: 1차로 직선으로 연결 가능한 포인트 이어주기
        smooth_path = self.ultimate_one_pass_compression(world_path)

        # 3단계: 코너 진입/탈출 부분에 포인트 생성
        corner_optimized = self.two_point_corner_optimization(smooth_path, turn_margin=2.0)
        
        # 4단계: 2차로 직선으로 연결 가능한 포인트 이어주기
        return self.ultimate_one_pass_compression(corner_optimized)

    def two_point_corner_optimization(self, path: List[Tuple[float, float]], turn_margin: float = 2.0) -> List[Tuple[float, float]]:
        if len(path) < 3: 
            return path
        
        refined_path: List[Tuple[float, float]] = path[:1]

        for i in range(1, len(path) - 1):
            p0, p1, p2 = path[i - 1], path[i], path[i + 1]
            x0, z0 = p0
            x1, z1 = p1
            x2, z2 = p2

            d1 = math.hypot(x1 - x0, z1 - z0) # 이전과 현재 포인트의 거리
            d2 = math.hypot(x2 - x1, z2 - z1) # 현재와 다음 포인트의 거리
            actual_margin = min(turn_margin, d1 * 0.4, d2 * 0.4)
            
            if actual_margin <= 0.2:
                refined_path.append(p1)
                continue

            t1, t2 = actual_margin / d1, actual_margin / d2
            ax, az = x1 - (x1 - x0) * t1, z1 - (z1 - z0) * t1
            bx, bz = x1 + (x2 - x1) * t2, z1 + (z2 - z1) * t2
            ax_pt, bx_pt = (ax, az), (bx, bz)

            ix1, iz1 = self.world_to_grid_index(ax, az)
            ix2, iz2 = self.world_to_grid_index(bx, bz)

            # _DStarNode 구조체 내의 walkable 플래그를 정확하게 참조하여 유효성 판정
            if self._grid[ix1][iz1].walkable and self._grid[ix2][iz2].walkable:
                refined_path.append(ax_pt)
                refined_path.append(bx_pt)
            else:
                refined_path.append(p1)

        last_item = path[-1]
        refined_path.append(last_item)
        
        unique_path = []
        for p in refined_path:
            if not unique_path:
                unique_path.append(p)
                continue
            prev_x, prev_z = unique_path[-1]
            curr_x, curr_z = p
            if math.hypot(curr_x - prev_x, curr_z - prev_z) > 0.1:
                unique_path.append(p)
        return unique_path

    def ultimate_one_pass_compression(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        경로의 맨 끝 점부터 역순으로 레이저를 쏘아, 장애물이 없는 가장 원거리의 점을 찾아 결합합니다.
        대괄호 문법이 차단되어 에러가 절대 발생하지 않는 고신뢰성 슬라이싱 버전입니다.
        """
        if len(path) <= 2: 
            return path
        
        working_list = list(path)
        # pop(0) 문법을 사용해 순수한 0번째 튜플 한 개를 꺼내 초기 리스트 세팅
        compressed: List[Tuple[float, float]] = [working_list.pop(0)]
        
        curr_idx = 0
        total_len = len(path)

        while curr_idx < total_len - 1:
            next_idx = curr_idx + 1
            
            # 대괄호 유실 문제를 차단하기 위해 변수 슬라이싱과 역순 튜플 해체 사용
            current_point_tuple = path[curr_idx]
            
            # 현재 인덱스 다음부터 리스트 끝까지의 요소를 인덱스와 함께 추출한 뒤 역순 정렬
            candidates = list(enumerate(path))
            target_candidates = candidates[curr_idx + 1:]
            reversed_candidates = target_candidates[::-1]
            
            for check_idx, check_point_tuple in reversed_candidates:
                # 두 좌표 간에 레이저를 가로막는 장애물 마진벽이 없는지 검사
                if self._is_straight_line_walkable(current_point_tuple, check_point_tuple):
                    next_idx = check_idx
                    break
            
            # 장애물이 없는 가장 원거리 변곡점을 찾아 추가하고 인덱스 이동
            target_point = path[next_idx]
            compressed.append(target_point)
            curr_idx = next_idx

        # 인접 중복 좌표 최종 소멸 처리
        unique_path = []
        for p in compressed:
            if not unique_path:
                unique_path.append(p)
                continue
            prev_x, prev_z = unique_path[-1]
            cx, cz = p
            if math.hypot(cx - prev_x, cz - prev_z) > 0.05:
                unique_path.append(p)

        return unique_path

    def _is_straight_line_walkable(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        x1, z1 = p1
        x2, z2 = p2
        dist = math.hypot(x2 - x1, z2 - z1)
        if dist == 0: 
            return True
        step_size = 0.5
        steps = int(dist / step_size)
        for i in range(1, steps):
            t = i / steps
            cx = x1 + (x2 - x1) * t
            cz = z1 + (z2 - z1) * t
            ix, iz = self.world_to_grid_index(cx, cz)
            if not self._grid[ix][iz].walkable: 
                return False
        return True


    def reset_planner(self) -> None:
        """
        D* Lite의 모든 비용 테이블과 우선순위 탐색 큐(U), 글로벌 보정 키값을 초기화합니다.
        """
        # 1. 우선순위 탐색 큐(U)를 완전히 비웁니다.
        self.U = []
        
        # 2. 전차 이동 거리에 따른 휴리스틱 글로벌 보정 키값을 0으로 리셋합니다.
        self.km = 0.0
        
        # 3. 이전 스텝의 기록 변수들을 깨끗하게 지웁니다.
        self.last_start = None
        self._current_goal_node = None

        # 4. 9만 칸 전체 격자를 순회하며 누적되어 있던 g비용과 rhs예측비용을 무한대로 초기화합니다.
        for col in self._grid:
            for node in col:
                node.g = float('inf')
                node.rhs = float('inf')
                # (만약 walkable 상태까지 완전 초기화하고 싶다면 하단 주석 해제)
                # node.walkable = True 
                
        # 5. 장애물 리스트를 비우고 그리드 유효성 플래그를 내립니다.
        # self._obstacles = [] # 일단 오브젝트 정보는 살려둔다.

        self._grid_valid = False
        print("리셋되었습니다.")


    def plot(self, path: Optional[List[Tuple[float, float]]] = None, show_grid: bool = False, figsize: Tuple[int, int] = (7, 7), title: Optional[str] = None, fname: str = 'path_result') -> None:
            if not _HAS_MPL: 
                return
            fig, ax = plt.subplots(figsize=figsize)
            for obs in self._obstacles:
                x_min = obs.x_min - self.obstacle_margin
                z_min = obs.z_min - self.obstacle_margin
                w = (obs.x_max + self.obstacle_margin) - x_min
                h = (obs.z_max + self.obstacle_margin) - z_min
                rect = plt.Rectangle((x_min, z_min), w, h, color='red', alpha=0.25, linewidth=0)
                ax.add_patch(rect)
    
            if show_grid:
                xs_grid = [self.grid_min_x + i * self.cell_size for i in range(self.grid_size_x + 1)]
                zs_grid = [self.grid_min_z + i * self.cell_size for i in range(self.grid_size_z + 1)]
                for i, x in enumerate(xs_grid):
                    if i % 10 == 0: 
                        ax.axvline(x, linewidth=0.2, color='gray', alpha=0.5)
                for i, z in enumerate(zs_grid):
                    if i % 10 == 0: 
                        ax.axhline(z, linewidth=0.2, color='gray', alpha=0.5)
    
            cnt = 0
            for i in range(len(path)-1):
                cnt += ((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)**0.5
    
            if path:
                xs = [point_tuple[0] for point_tuple in path]
                zs = [point_tuple[1] for point_tuple in path]
                ax.plot(xs, zs, color="blue", linewidth=2.5, label="Path", zorder=3)
                ax.scatter(xs, zs, color="black", s=30, zorder=5, label="Waypoints")
                ax.legend(loc="upper right")
    
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(self.grid_min_x, self.grid_max_x)
            ax.set_ylim(self.grid_min_z, self.grid_max_z)
            ax.set_xlabel("X (World)")
            ax.set_ylabel("Z (World)")
            ax.set_title(cnt)
            
            plt.tight_layout()
            plt.savefig(f"{fname}.png", dpi=150)
            plt.close()
