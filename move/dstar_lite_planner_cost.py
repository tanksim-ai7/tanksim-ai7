from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np

GridNode = Tuple[int, int]
WorldPoint = Tuple[float, float]
INF = float("inf")

@dataclass(frozen=True)
class ObstacleRect:
    x_min: float
    x_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_min_max(cls, x_min, x_max, z_min, z_max):
        return cls(
            x_min=min(float(x_min), float(x_max)),
            x_max=max(float(x_min), float(x_max)),
            z_min=min(float(z_min), float(z_max)),
            z_max=max(float(z_min), float(z_max)),
        )


class DStarPlanner:
    """
    server_sample v2.3.10(2).ipynb와 맞춘 D* Lite Planner.

    서버에서 사용하는 공개 메서드:
        world_to_grid()
        find_path()
        set_obstacles()
        get_path()
        get_path_cost()
        plot()
    """

    def __init__(
        self,
        start=(0, 0),
        goal=(299, 299),
        width=300,
        height=300,
        obstacles=None,
        allow_diagonal=True,
        obstacle_margin=2.0,
        clearance_radius=8.0,
        clearance_weight=4.0,
        clearance_decay=2.5,
    ):
        self.width = int(width)
        self.height = int(height)
        self.allow_diagonal = bool(allow_diagonal)

        # obstacle_margin: 차체 충돌 방지를 위한 hard padding (통행 불가)
        self.obstacle_margin = float(obstacle_margin)

        # clearance_*: hard padding 바깥의 soft cost 영역
        # 유한 비용이므로 좁은 길은 필요하면 통과할 수 있다.
        self.clearance_radius = max(0.0, float(clearance_radius))
        self.clearance_weight = max(0.0, float(clearance_weight))
        self.clearance_decay = max(1e-6, float(clearance_decay))

        self.start = tuple(map(int, start))
        self.goal = tuple(map(int, goal))
        self.last_start = self.start

        self.obstacle_rectangles: List[ObstacleRect] = []
        self.obstacles: Set[GridNode] = set(obstacles or [])

        # 각 자유 셀의 장애물 근접 추가 비용. 값이 없으면 추가 비용 0.
        self.clearance_costs: Dict[GridNode, float] = {}
        self.obstacle_distances: Dict[GridNode, float] = {}

        # y축 추가
        self.y: Dict[GridNode, float] = {}
        self.high: Set[GridNode] = set() # 특정 고도를 막기 위한 변수

        self.g: Dict[GridNode, float] = {}
        self.rhs: Dict[GridNode, float] = {self.goal: 0.0}
        self.km = 0.0

        self.open_heap = []
        self.open_entries = {}
        self._push_count = 0

        self.last_path: List[WorldPoint] = []

        # replan_if_needed()가 "목적지/시작 grid가 바뀌었는지"를
        # 판단하기 위해 마지막으로 재계획했던 grid를 기억해둔다.
        # 이 상태를 여기서 관리하기 때문에 서버(main) 쪽은
        # previous_pos 같은 걸 따로 들고 다닐 필요가 없다.
        self._replan_start_grid: Optional[GridNode] = None
        self._replan_goal_grid: Optional[GridNode] = None

        self._insert_open(self.goal, self.calculate_key(self.goal))

        # 고도 정보 insert
        loaded_data = np.load('move/risk_layers.npz')
        loaded_data = loaded_data['height']
        loaded_data = np.flipud(loaded_data)
        loaded_data = np.rot90(loaded_data, k=-1)
        self.update_entire_heightmap(loaded_data)

    # --------------------------------------------------
    # 좌표 변환
    # --------------------------------------------------

    def world_to_grid(self, position, clamp=False):
        """
        서버의 [x, z] 좌표를 Grid (x, z)로 변환.
        현재 Terrain이 300 x 300이고 cell size=1로 가정.
        """
        if position is None:
            return None

        x = int(round(float(position[0])))
        z = int(round(float(position[1])))

        if clamp:
            x = min(max(x, 0), self.width - 1)
            z = min(max(z, 0), self.height - 1)

        node = (x, z)

        if not self.in_bounds(node):
            raise ValueError(f"좌표 {position}가 Grid 범위를 벗어났습니다.")

        return node

    def grid_to_world(self, node):
        return float(node[0]), float(node[1])

    # --------------------------------------------------
    # 기본값 접근
    # --------------------------------------------------

    def get_g(self, node):
        return self.g.get(node, INF)

    def get_rhs(self, node):
        return self.rhs.get(node, INF)

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------

    def in_bounds(self, node):
        x, z = node
        return 0 <= x < self.width and 0 <= z < self.height

    def is_free(self, node):
        # return self.in_bounds(node) and node not in self.obstacles and node not in self.high
        return self.in_bounds(node) and node not in self.obstacles

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dz = abs(a[1] - b[1])

        # 1. 평면 최단 거리 계산 분기
        if not self.allow_diagonal:
            # 대각 이동 금지(4방향) 시: 맨해튼 거리 공식
            flat_dist = float(dx + dz)
        else:
            # 대각 이동 허용(8방향) 시: 옥타일 거리 공식
            flat_dist = max(dx, dz) + (math.sqrt(2.0) - 1.0) * min(dx, dz)
            
        # 2. 대각 이동 안 타더라도 y축 고도 차이는 무조건 정밀 합산
        y_a = self.y.get(a, 0.0)
        y_b = self.y.get(b, 0.0)
        return flat_dist + abs(y_b - y_a)

    def get_neighbors(self, node):
        x, z = node

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        if self.allow_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        result = []

        for dx, dz in directions:
            nxt = (x + dx, z + dz)

            if not self.is_free(nxt):
                continue

            if dx != 0 and dz != 0:
                side_x = (x + dx, z)
                side_z = (x, z + dz)

                if not self.is_free(side_x) or not self.is_free(side_z):
                    continue

            result.append(nxt)

        return result

    def rebuild_clearance_costs(self):
        """
        hard obstacle 셀에서 clearance_radius 안쪽 자유 셀까지의 거리를
        8방향 다중 시작점 Dijkstra로 계산하고 soft cost를 만든다.

        추가 비용:
            clearance_weight * exp(-distance / clearance_decay)

        장애물 셀 자체는 movement_cost에서 INF로 처리된다.
        """
        self.clearance_costs.clear()
        self.obstacle_distances.clear()

        if (
            not self.obstacles
            or self.clearance_radius <= 0.0
            or self.clearance_weight <= 0.0
        ):
            return

        distance = {}
        heap = []

        for node in self.obstacles:
            distance[node] = 0.0
            heapq.heappush(heap, (0.0, node))

        directions = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        ]

        while heap:
            current_distance, node = heapq.heappop(heap)

            if current_distance != distance.get(node):
                continue

            if current_distance > self.clearance_radius:
                continue

            x, z = node

            for dx, dz, step in directions:
                neighbor = (x + dx, z + dz)

                if not self.in_bounds(neighbor):
                    continue

                next_distance = current_distance + step

                if next_distance > self.clearance_radius:
                    continue

                if next_distance < distance.get(neighbor, INF):
                    distance[neighbor] = next_distance
                    heapq.heappush(heap, (next_distance, neighbor))

        for node, obstacle_distance in distance.items():
            if node in self.obstacles:
                continue

            self.obstacle_distances[node] = obstacle_distance
            self.clearance_costs[node] = (
                self.clearance_weight
                * math.exp(-obstacle_distance / self.clearance_decay)
            )

    def get_clearance_cost(self, node):
        return self.clearance_costs.get(node, 0.0)

    def refresh_costmap(self):
        """장애물 셀이 외부에서 직접 변경됐을 때 비용맵과 탐색 상태 갱신."""
        self.rebuild_clearance_costs()
        self._reset_search(self.start, self.goal)

    def movement_cost(self, a, b):
        if not self.is_free(a) or not self.is_free(b):
            return INF

        dx = abs(a[0] - b[0])
        dz = abs(a[1] - b[1])

        if dx + dz == 1:
            base_cost = 1.0

        elif self.allow_diagonal and dx == 1 and dz == 1:
            side_x = (b[0], a[1])
            side_z = (a[0], b[1])

            if not self.is_free(side_x) or not self.is_free(side_z):
                return INF

            base_cost = math.sqrt(2.0)

        else:
            return INF

        # 출발/도착 셀의 근접 비용 평균을 이동거리에 더한다.
        # 비용은 유한하므로 우회로가 없으면 좁은 통로도 통과 가능하다.
        proximity_cost = 0.5 * (
            self.get_clearance_cost(a)
            + self.get_clearance_cost(b)
        )

        return base_cost + proximity_cost

    # --------------------------------------------------
    # Priority Queue
    # --------------------------------------------------

    def calculate_key(self, node):
        min_cost = min(self.get_g(node), self.get_rhs(node))

        return (
            min_cost + self.heuristic(self.start, node) + self.km,
            min_cost,
        )

    # (?, ?, self._push_count, (x,z))
    def _insert_open(self, node, key):
        self._push_count += 1
        self.open_entries[node] = key

        heapq.heappush(
            self.open_heap,
            (key[0], key[1], self._push_count, node),
        )

    def _remove_open(self, node):
        self.open_entries.pop(node, None)

    def _clean_open(self):
        while self.open_heap:
            key = (self.open_heap[0][0], self.open_heap[0][1])
            node = self.open_heap[0][3]

            if self.open_entries.get(node) == key:
                break

            heapq.heappop(self.open_heap)

    def _top_key(self):
        self._clean_open()

        if not self.open_heap:
            return INF, INF

        return self.open_heap[0][0], self.open_heap[0][1]

    def _pop_open(self):
        self._clean_open()

        if not self.open_heap:
            return None, (INF, INF)

        key1, key2, _, node = heapq.heappop(self.open_heap)
        self.open_entries.pop(node, None)

        return node, (key1, key2)

    # --------------------------------------------------
    # D* Lite
    # --------------------------------------------------

    def _reset_search(self, start, goal):
        self.start = start
        self.goal = goal
        self.last_start = start

        self.g.clear()
        self.rhs.clear()
        self.rhs[self.goal] = 0.0

        self.km = 0.0
        self.open_heap.clear()
        self.open_entries.clear()
        self._push_count = 0

        self._insert_open(self.goal, self.calculate_key(self.goal))

    def move_start(self, new_start):
        new_start = tuple(new_start)

        if not self.is_free(new_start):
            raise ValueError(f"새 시작점 {new_start}이 장애물이거나 지도 밖입니다.")

        if new_start == self.start:
            return

        self.km += self.heuristic(self.last_start, new_start)
        self.start = new_start
        self.last_start = new_start

    def update_vertex(self, node):
        if node != self.goal:
            self.rhs[node] = min(
                (
                    self.movement_cost(node, neighbor) + self.get_g(neighbor)
                    for neighbor in self.get_neighbors(node)
                ),
                default=INF,
            )

        self._remove_open(node)

        if not math.isclose(
            self.get_g(node),
            self.get_rhs(node),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            self._insert_open(node, self.calculate_key(node))

    def compute_shortest_path(self, max_iterations=2_000_000):
        iteration = 0
        while (
            self._top_key() < self.calculate_key(self.start)
            or not math.isclose(
                self.get_rhs(self.start),
                self.get_g(self.start),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            if iteration >= max_iterations:
                raise RuntimeError("D* Lite 최대 반복 횟수를 초과했습니다.")

            node, old_key = self._pop_open()

            if node is None:
                break

            new_key = self.calculate_key(node)

            if old_key < new_key:
                self._insert_open(node, new_key)

            elif self.get_g(node) > self.get_rhs(node):
                self.g[node] = self.get_rhs(node)

                for predecessor in self.get_neighbors(node):
                    self.update_vertex(predecessor)

            else:
                self.g[node] = INF
                self.update_vertex(node)

                for predecessor in self.get_neighbors(node):
                    self.update_vertex(predecessor)

            iteration += 1
        
        return self.get_g(self.start) < INF

    # --------------------------------------------------
    # 서버 호환 공개 메서드
    # --------------------------------------------------

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
            ix, iz = self.world_to_grid((cx, cz), True)
            if not self.is_free((ix, iz)): 
                return False
        return True

    def ultimate_one_pass_compression(self, path: List[WorldPoint]) -> List[WorldPoint]:
        if len(path) <= 2: 
            return path
        
        working_list = list(path)
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

    def find_path(self, current_pos, dest):
        """
        서버 호출:
            current_path = planner.find_path(current_pos, dest)
        """
        start = self.world_to_grid(current_pos, clamp=True)
        goal = self.world_to_grid(dest, clamp=True)

        if not self.is_free(start):
            raise ValueError(f"시작 위치 {start}가 장애물에 포함됩니다.")

        if not self.is_free(goal):
            raise ValueError(f"목적지 {goal}가 장애물에 포함됩니다.")

        if goal != self.goal:
            self._reset_search(start, goal)
        else:
            self.move_start(start)

        if not self.compute_shortest_path():
            self.last_path = []
            return []

        grid_path = self._extract_grid_path()
        tmp = [self.grid_to_world(node) for node in grid_path]
        self.last_path = self.ultimate_one_pass_compression(tmp)

        return self.last_path

    # --------------------------------------------------
    # 시작점 hard obstacle 갇힘 비상 탈출
    # --------------------------------------------------

    def clear_start_area(self, position, radius=2):
        """
        position 주변 radius 칸 이내의 hard obstacle/통행 불가 지형을
        "영구적으로" 강제 해제하는 비상 도구.

        주의 (중요):
            obstacles/terrain_blocked 집합에서 셀을 영구적으로 지운다.
            그래서 정상 주행 중에 매 tick 호출하면 안 된다 — 그렇게 하면
            전차가 지뢰 등 진짜 위험 지역 옆을 지나갈 때마다 주변 셀이
            계속 안전지대로 지워져서, 실제로는 위험한 곳을 D* Lite가
            자유롭게 통행 가능하다고 착각할 수 있다.
            (_find_path_with_recovery()에서 "실제로 막혀 있을 때"만
            호출하는 이유)


        obstacle_rectangles(시각화용 원본 도형 목록)는 건드리지 않으므로
        plot() 이미지에는 위험 지역이 계속 표시된다 — 경로탐색용 grid
        판정만 patch되는 것이다.

        Returns
        -------
        set[GridNode]
            실제로 통행 가능 상태로 바뀐 셀들.
        """
        center = self.world_to_grid(position, clamp=True)
        changed_obstacles = set()
        changed_terrain = set()
        terrain_blocked = getattr(self, "terrain_blocked", None)

        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                cell = (center[0] + dx, center[1] + dz)

                if not self.in_bounds(cell):
                    continue

                if cell in self.obstacles:
                    self.obstacles.discard(cell)
                    changed_obstacles.add(cell)

                if terrain_blocked is not None and cell in terrain_blocked:
                    terrain_blocked.discard(cell)
                    changed_terrain.add(cell)

        if changed_obstacles or changed_terrain:
            # rebuild_clearance_costs()만으로는 부족하다 — D* Lite의 내부
            # g/rhs 그래프는 update_vertex()를 호출해야만 "이 셀이 이제
            # 통행 가능하다"는 사실을 실제로 반영한다.
            self.rebuild_clearance_costs()

            affected_nodes = set(changed_obstacles) | set(changed_terrain)
            for x, z in list(affected_nodes):
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (x + dx, z + dz)
                        if self.in_bounds(neighbor):
                            affected_nodes.add(neighbor)

            for node in affected_nodes:
                self.update_vertex(node)

        if changed_obstacles:
            print(
                f"⚠️ clear_start_area: {center} 주변 반경 {radius}칸 중 장애물 "
                f"{len(changed_obstacles)}칸을 비상 해제했습니다."
            )

        if changed_terrain:
            print(
                f"⚠️ clear_start_area: {center} 주변 반경 {radius}칸 중 통행 불가 지형 "
                f"{len(changed_terrain)}칸을 비상 해제했습니다 (경사 등 실제 위험 지역일 수 있음)."
            )

        return changed_obstacles | changed_terrain

    def _find_path_with_recovery(self, current_pos, dest, max_radius=6):
        """
        find_path()를 시도하고, 실패하면 clear_start_area()로 점점 더
        넓게 뚫어가며 재시도한다. 매 반경 단계마다 실제로 find_path()를
        다시 돌려서 "진짜로 경로가 나오는지"를 직접 확인한다 — is_free()
        판정만으로는 큰 장애물 덩어리 안의 작은 섬에 갇힌 경우를 놓칠
        수 있기 때문이다 (clear_start_area 참고).

        시작점/목적지 어느 쪽이 막혀 있는지 모두 확인해서 필요한 쪽을
        같이 뚫는다.
        """
        try:
            path = self.find_path(current_pos, dest)
        except ValueError:
            path = []

        if path:
            return path

        start_grid = self.world_to_grid(current_pos, clamp=True)
        goal_grid = self.world_to_grid(dest, clamp=True)
        start_blocked = not self.is_free(start_grid)
        goal_blocked = not self.is_free(goal_grid)

        if not (start_blocked or goal_blocked):
            # 둘 다 멀쩡한데 경로가 없다 -> clear_start_area로 고칠 수 있는
            # 문제가 아니라 진짜로 단절된 지형이다. 반경을 키워봐야 소용없다.
            return []

        for radius in range(1, max_radius + 1):
            if start_blocked:
                self.clear_start_area(current_pos, radius=radius)
            if goal_blocked:
                self.clear_start_area(dest, radius=radius)

            try:
                path = self.find_path(current_pos, dest)
            except ValueError:
                path = []

            if path:
                print(f"✅ _find_path_with_recovery: 반경 {radius}칸까지 비상 해제 후 경로를 찾았습니다.")
                return path

        print(
            f"⚠️ _find_path_with_recovery: 반경 {max_radius}칸까지 비상 해제해봤지만 "
            f"{start_grid} -> {goal_grid} 경로를 찾지 못했습니다. "
            "장애물이 반경보다 넓게 퍼져 있거나 진짜로 단절된 지역일 수 있습니다."
        )
        return []

    # --------------------------------------------------
    # 재계획 / 재플롯 판단 (서버 쪽 상태 관리를 없애기 위한 래퍼)
    # --------------------------------------------------

    def reset_replan_tracking(self):
        """
        /init 등 에피소드가 새로 시작될 때 호출한다.
        마지막으로 재계획했던 start/goal grid 기록을 지워
        다음 replan_if_needed() 호출이 무조건 재계획하도록 만든다.
        """
        self._replan_start_grid = None
        self._replan_goal_grid = None

    def replan_if_needed(self, current_pos, dest, save_path=None, force=False):
        """
        서버(main)가 매 tick 호출하는 단일 진입점.

        이 메서드가 하는 일:
            1. start grid 또는 goal(목적지) grid가 실제로 바뀌었을 때만
               (혹은 force=True일 때만) find_path()로 재계획한다.
            2. 시작점/목적지가 막혀서 실패하면 _find_path_with_recovery()가
               반경을 점점 늘려가며 실제로 경로가 나올 때까지 재시도한다.
            3. 목적지가 실제로 바뀐 경우(또는 최초/강제)에만 plot()으로 이미지를 갱신한다.

        Returns
        -------
        list[WorldPoint] | None
            재계획을 실제로 시도했으면 새 world-path (실패 시 빈 리스트).
            재계획이 필요 없었으면 None (호출부는 기존 path를 그대로 쓰면 된다).
        """
        if dest is None:
            return None

        new_start_grid = self.world_to_grid(current_pos, clamp=True)
        new_goal_grid = self.world_to_grid(dest, clamp=True)

        goal_changed = new_goal_grid != self._replan_goal_grid
        start_changed = new_start_grid != self._replan_start_grid

        if not (force or goal_changed or start_changed):
            return None

        path = self._find_path_with_recovery(current_pos, dest)

        self._replan_start_grid = new_start_grid
        self._replan_goal_grid = new_goal_grid


        if save_path and (force or goal_changed):
            self.plot_async(path, save_path=save_path)

        return path

    def _extract_grid_path(self, max_path_length=None):
        if self.start == self.goal:
            return [self.start]

        if self.get_g(self.start) == INF:
            return []

        if max_path_length is None:
            max_path_length = self.width * self.height * 4

        current = self.start
        path = [current]
        visited = {current}

        while current != self.goal and len(path) < max_path_length:
            neighbors = self.get_neighbors(current) # 갈 수 있는 경로

            if not neighbors:
                return []

            next_node = min(
                neighbors,
                key=lambda neighbor: (
                    self.movement_cost(current, neighbor) + self.get_g(neighbor),
                    self.heuristic(neighbor, self.goal),
                ),
            )

            if (
                self.movement_cost(current, next_node) + self.get_g(next_node)
                == INF
            ):
                return []

            if next_node in visited:
                return []

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path if current == self.goal else []

    def get_path(self, current_pos=None):
        """
        서버 시작 부분의 planner.get_path(current_pos) 호출과 호환.
        계산된 경로가 없으면 [] 반환.
        """
        if current_pos is None:
            return list(self.last_path)

        if self.last_path:
            return list(self.last_path)

        return []

    def get_path_cost(self, path=None):
        active_path = self.last_path if path is None else path

        if not active_path or len(active_path) < 2:
            return 0.0

        grid_path = [
            self.world_to_grid(point, clamp=True)
            for point in active_path
        ]

        return sum(
            self.movement_cost(grid_path[i], grid_path[i + 1])
            for i in range(len(grid_path) - 1)
        )

    # y축 값 추가
    def update_entire_heightmap(self, heightmap: np.ndarray) -> None:
        """
        300x300 등 넘파이 2D 고도화 배열을 전체 그리드 노드에 한 번에 주입합니다.
        """
        
        if heightmap.shape != (self.width, self.height):
            raise ValueError(f"지형 매트릭스 크기 {heightmap.shape}가 플래너 격자 크기 "
                             f"({self.width}, {self.height})와 일치하지 않습니다.")
        
        height_flat = heightmap.flat
        idx = 0
        for ix in range(self.width):
            for iz in range(self.height):
                self.y[(ix,iz)] = float(height_flat[idx])

                if float(height_flat[idx]) >= 13.0:
                    self.high.add((ix,iz))

                idx += 1
                
        print(f"총 {idx}개 노드의 고도 데이터가 넘파이 배열로부터 일괄 주입되었습니다.")

    # --------------------------------------------------
    # 장애물
    # --------------------------------------------------

    def set_obstacles(self, obs_list: Iterable[ObstacleRect]):
        """
        서버 호출:
            changed_cells = planner.set_obstacles(obs_list)
        """
        self.obstacle_rectangles = list(obs_list)
        new_obstacles = set()

        for obs in self.obstacle_rectangles:
            x_min = math.floor(obs.x_min - self.obstacle_margin)
            x_max = math.ceil(obs.x_max + self.obstacle_margin)
            z_min = math.floor(obs.z_min - self.obstacle_margin)
            z_max = math.ceil(obs.z_max + self.obstacle_margin)

            x_min = max(0, x_min)
            x_max = min(self.width - 1, x_max)
            z_min = max(0, z_min)
            z_max = min(self.height - 1, z_max)

            for x in range(x_min, x_max + 1):
                for z in range(z_min, z_max + 1):
                    new_obstacles.add((x, z))

        changed_cells = self.obstacles.symmetric_difference(new_obstacles)
        affected_nodes = set(changed_cells)

        for x, z in changed_cells:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    node = (x + dx, z + dz)

                    if self.in_bounds(node):
                        affected_nodes.add(node)

        self.obstacles = new_obstacles

        # 장애물 주변의 edge cost도 함께 바뀌므로 soft costmap을 다시 만들고
        # 현재 시작점/목적지 기준으로 D* Lite 상태를 안전하게 재초기화한다.
        self.rebuild_clearance_costs()

        for node in changed_cells:
            if self.in_bounds(node):
                self.update_vertex(node)
                for neighbor in self.get_neighbors(node):
                    self.update_vertex(neighbor)
        # self._reset_search(self.start, self.goal)

        return changed_cells

    # --------------------------------------------------
    # 시각화
    # --------------------------------------------------

    def plot(
        self,
        path=None,
        show_grid=True,
        title="D* Lite",
        save_path=None,
        show=False,
    ):
        """
        서버 호출:
            planner.plot(path=current_path, show_grid=True, title="...")
        """
        active_path = self.last_path if path is None else path

        fig, ax = plt.subplots(figsize=(8, 8))

        height_matrix = np.zeros((self.width, self.height))
        for ix in range(self.width):
            for iz in range(self.height):
                height_matrix[ix, iz] = self.y.get((ix, iz), 0.0)
        height_matrix = height_matrix.T

        # 2. 플롯 설정 및 그리드 히트맵 드로잉
        im = ax.imshow(
            height_matrix, 
            cmap='terrain', 
            origin='lower',
            extent=[0, self.width, 0, self.height],
            zorder=1
        )

        # (ix + 0.5)

        # for x,z in self.high:
        #     rect_grid = plt.Rectangle(
        #         ((x+0.5) - 1.0 * 0.5, (z+0.5) - 1.0 * 0.5), # 사각형의 시작점(좌측 하단)
        #         1.0, 1.0,
        #         facecolor='magenta', 
        #         edgecolor='none', 
        #         alpha=0.3, # 다른 표시와 겹쳐도 다 보이도록 투명도 설정
        #         zorder=2
        #     )
        #     ax.add_patch(rect_grid)
        # ax.plot([], [], color='magenta', alpha=0.3, label="Blocked Terrain", linestyle='-', linewidth=5)

        for obs in self.obstacle_rectangles:
            patch = Rectangle(
                (
                    obs.x_min - self.obstacle_margin,
                    obs.z_min - self.obstacle_margin,
                ),
                (obs.x_max - obs.x_min) + 2 * self.obstacle_margin,
                (obs.z_max - obs.z_min) + 2 * self.obstacle_margin,
                alpha=0.5,
            )
            ax.add_patch(patch)

        if active_path:
            px = [point[0] for point in active_path]
            pz = [point[1] for point in active_path]

            ax.plot(px, pz, marker="o", markersize=2, label="D* Lite Path")
            ax.scatter(px[0], pz[0], s=80, marker="o", label="Start")
            ax.scatter(px[-1], pz[-1], s=120, marker="*", label="Goal")

        else:
            ax.scatter(
                self.start[0],
                self.start[1],
                s=80,
                marker="o",
                label="Start",
            )
            ax.scatter(
                self.goal[0],
                self.goal[1],
                s=120,
                marker="*",
                label="Goal",
            )

        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Height / Altitude (meters)", rotation=275, labelpad=15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title("Path Map")

        if show_grid:
            ax.grid(True, alpha=0.3)

        ax.legend()
        fig.tight_layout()

        if save_path:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # --------------------------------------------------
    # 논블로킹 렌더링 (RiskDStarPlanner.plot_async와 동일한 목적)
    # --------------------------------------------------

    def plot_async(self, path=None, show_grid=True, title="D* Lite", save_path=None):
        """
        plot()과 같은 그림을 그리지만, 실제 렌더링은 별도 스레드에서
        수행하고 이 함수는 즉시 리턴한다.
        """
        active_path = list(self.last_path if path is None else path)
        obstacle_rectangles_snapshot = list(self.obstacle_rectangles)
        start_snapshot = self.start
        goal_snapshot = self.goal
        obstacle_margin_snapshot = self.obstacle_margin

        thread = threading.Thread(
            target=self._render_and_save,
            args=(
                active_path,
                obstacle_rectangles_snapshot,
                start_snapshot,
                goal_snapshot,
                obstacle_margin_snapshot,
                show_grid,
                title,
                save_path,
            ),
            daemon=True,
        )
        thread.start()

    def _render_and_save(self, active_path, obstacle_rectangles, start, goal,
                          obstacle_margin, show_grid, title, save_path):
        """
        plot_async()가 뜬 스냅샷으로 실제 렌더링을 수행한다.
        pyplot을 쓰지 않고 Figure를 직접 만들어서 스레드 간 간섭을 피한다.
        """
        fig = Figure(figsize=(8, 8))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        height_matrix = np.zeros((self.width, self.height))
        for ix in range(self.width):
            for iz in range(self.height):
                height_matrix[ix, iz] = self.y.get((ix, iz), 0.0)
        height_matrix = height_matrix.T

        im = ax.imshow(
            height_matrix,
            cmap='terrain',
            origin='lower',
            extent=[0, self.width, 0, self.height],
            zorder=1,
        )

        for obs in obstacle_rectangles:
            patch = Rectangle(
                (obs.x_min - obstacle_margin, obs.z_min - obstacle_margin),
                (obs.x_max - obs.x_min) + 2 * obstacle_margin,
                (obs.z_max - obs.z_min) + 2 * obstacle_margin,
                alpha=0.5,
            )
            ax.add_patch(patch)

        if active_path:
            px = [point[0] for point in active_path]
            pz = [point[1] for point in active_path]
            ax.plot(px, pz, marker="o", markersize=2, label="D* Lite Path")
            ax.scatter(px[0], pz[0], s=80, marker="o", label="Start")
            ax.scatter(px[-1], pz[-1], s=120, marker="*", label="Goal")
        else:
            ax.scatter(start[0], start[1], s=80, marker="o", label="Start")
            ax.scatter(goal[0], goal[1], s=120, marker="*", label="Goal")

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Height / Altitude (meters)", rotation=275, labelpad=15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(title)

        if show_grid:
            ax.grid(True, alpha=0.3)

        ax.legend()
        fig.tight_layout()

        if save_path:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.print_figure(output, dpi=150, bbox_inches="tight")
            print("grid 저장 완료 (백그라운드)")

        return fig, ax


# 기존 이름 호환
DStarLite = DStarPlanner