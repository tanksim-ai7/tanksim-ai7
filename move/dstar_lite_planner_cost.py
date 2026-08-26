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

# 자연물이 아닌 오브젝트에 대해서 값을 가지고 있어야 한다.
# 최초에 자연물이 아닌 모든 오브젝트의 값은 unknown_list에 있어야한다.
# 시뮬레이션을 진행하면서 unknown이였던 오브젝트가 team, enemy 등과 같이 특정이 된다면
# unknown_list에서 빼주고
# 해당하는 list에 넣어줘야한다.
# (각 오브젝트 type에 따라 각자 다른 로직을 타야하기 때문에)
unknown_list = [] # 최초에 여기에 자연물이 아닌 모든 오브젝트들을 넣어줘야 함
enemy_list = []
enemy_tank_list = []
team_list = []
team_tank_list = []

@dataclass(frozen=True)
class ObstacleRect:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    type: str = 'nature'
    # nature: 자연물
    # unknown: 아군/적군 여부를 알 수 없는 오브젝트
    # enemy: 적군 오브젝트
    # enemy_tank: 적군 탱크
    # team: 아군 오브젝트
    # team_tank: 아군 탱크

    @classmethod
    def from_min_max(cls, x_min, x_max, z_min, z_max):

        # 모든 좌표가 ±0.2 오차 범위 안에 있는지 검사하는 보조 함수
        def _is_match(target_list):
            for tx_min, tx_max, tz_min, tz_max in target_list:
                if (abs(tx_min - x_min) <= 0.2 and 
                    abs(tx_max - x_max) <= 0.2 and 
                    abs(tz_min - z_min) <= 0.2 and 
                    abs(tz_max - z_max) <= 0.2):
                    return True
            return False
        
        type = 'nature'
        if _is_match(unknown_list):
            type = 'unknown'
        elif _is_match(enemy_list):
            type = 'enemy'
        elif _is_match(team_list):
            type = 'team'
        elif _is_match(enemy_tank_list):
            type = 'enemy_tank'
        elif _is_match(team_tank_list):
            type = 'team_tank'


        return cls(
            x_min=min(float(x_min), float(x_max)),
            x_max=max(float(x_min), float(x_max)),
            z_min=min(float(z_min), float(z_max)),
            z_max=max(float(z_min), float(z_max)),
            type=type,
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

        # 움직일 수 있는 적 전차에 대한 변수
        self.movable_enemy_tank: Set[GridNode] = set([])

        # 각 자유 셀의 장애물 근접 추가 비용. 값이 없으면 추가 비용 0.
        self.clearance_costs: Dict[GridNode, float] = {}
        self.obstacle_distances: Dict[GridNode, float] = {}

        # y축 추가
        self.y: Dict[GridNode, float] = {}
        self.updated_y: List[GridNode] = [] # 자연물에 의해 고도가 업데이트 된 좌표를 위한 공간

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

        # update_y_from_nature()가 매번 디스크에서 다시 읽지 않도록 캐싱.
        # 지형 자체는 게임 도중 안 바뀌는 정적 데이터라 한 번만 읽으면 된다.
        self._cached_base_heightmap = loaded_data

        # update_y_from_nature()가 "지난번과 nature 배치가 똑같으면
        # 재계산을 건너뛰기" 위해 기억해두는 서명. None이면 아직 한 번도
        # 계산 안 한 상태라는 뜻이라 최초 1회는 무조건 계산한다.
        self._last_nature_signature = None


    def update_dstar_obstacles_from_payload(self, payload: dict):
        obs_list = []
        for item in payload.get("obstacles", []):
            obs = ObstacleRect.from_min_max(
                x_min=item["x_min"],
                x_max=item["x_max"],
                z_min=item["z_min"],
                z_max=item["z_max"],
            )
            obs_list.append(obs)
        self.set_obstacles(obs_list)

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
        return self.in_bounds(node) and node not in self.obstacles and node not in self.movable_enemy_tank

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

    def rebuild_clearance_costs_incremental(self, newly_obstacle_cells, newly_freed_cells):
        """
        rebuild_clearance_costs()의 증분(incremental) 버전.

        기존 rebuild_clearance_costs()는 오브젝트 하나만 추가돼도 매번
        self.obstacles 전체를 소스로 삼아 300x300 격자 전체에 다시
        multi-source Dijkstra를 돌렸다. 실제로 값이 바뀌는 범위는
        "이번에 새로 막히거나/뚫린 셀 주변, clearance_radius 이내"뿐이라
        대부분 낭비였다.

        Parameters
        ----------
        newly_obstacle_cells : Iterable[GridNode]
            이번 set_obstacles() 호출로 새로 '막힌' 셀들
            (changed_cells 중 지금 self.obstacles에 있는 것).
        newly_freed_cells : Iterable[GridNode]
            이번 호출로 새로 '뚫린' 셀들
            (changed_cells 중 지금 self.obstacles에 없는 것).

        정확성에 대한 설명
        ------------------
        1) 새로 막힌 셀(newly_obstacle_cells):
           이 셀들만 소스로 삼아 clearance_radius까지 bounded Dijkstra로
           확장하되, 어떤 free 셀에 대해 "새로 계산한 거리가 기존에
           저장된 거리보다 짧을 때만" 갱신한다. 장애물 추가는 거리를
           줄일 수만 있지(더 가까운 장애물이 새로 생기는 것) 늘릴 수는
           없으므로, 이 방식이 전체 재계산과 결과가 100% 동일하다.

        2) 새로 뚫린 셀(newly_freed_cells):
           장애물이 없어지면 주변 셀들의 '가장 가까운 장애물까지 거리'가
           멀어질 수 있어서(비용이 낮아짐), 국소적으로 처음부터 다시
           계산해야 한다. 다만 그 영향 범위가 clearance_radius를 넘을 수
           없으므로, 뚫린 셀 주변 clearance_radius*2 반경 안의 '진짜
           남아있는 장애물'만 소스로 삼아 그 창(window) 안쪽만 다시
           계산한다 (radius*2인 이유: 윈도우 경계에 있는 셀의 최근접
           장애물이 윈도우 밖, 최대 radius만큼 더 떨어진 곳에 있을 수
           있어서 소스 탐색 범위를 그만큼 더 넓혀야 정확하다).
        """
        if (
            self.clearance_radius <= 0.0
            or self.clearance_weight <= 0.0
        ):
            self.clearance_costs.clear()
            self.obstacle_distances.clear()
            return

        directions = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        ]

        # ---- 1) 새로 막힌 셀: bounded 확장, 기존 값보다 짧을 때만 갱신 ----
        if newly_obstacle_cells:
            heap = []
            seed_distance = {}

            for node in newly_obstacle_cells:
                seed_distance[node] = 0.0
                heapq.heappush(heap, (0.0, node))
                # 이제 이 셀 자신은 장애물이므로 clearance 대상에서 제외.
                self.obstacle_distances.pop(node, None)
                self.clearance_costs.pop(node, None)

            while heap:
                current_distance, node = heapq.heappop(heap)

                if current_distance != seed_distance.get(node):
                    continue
                if current_distance > self.clearance_radius:
                    continue

                x, z = node

                for dx, dz, step in directions:
                    neighbor = (x + dx, z + dz)

                    if not self.in_bounds(neighbor):
                        continue
                    if neighbor in self.obstacles:
                        continue

                    next_distance = current_distance + step

                    if next_distance > self.clearance_radius:
                        continue

                    # 새로 생긴 장애물들 사이에서 계속 퍼져나가기 위한 갱신.
                    if next_distance < seed_distance.get(neighbor, INF):
                        seed_distance[neighbor] = next_distance
                        heapq.heappush(heap, (next_distance, neighbor))

                    # 전역 저장값 갱신은 "더 가까워졌을 때만".
                    if next_distance < self.obstacle_distances.get(neighbor, INF):
                        self.obstacle_distances[neighbor] = next_distance
                        self.clearance_costs[neighbor] = (
                            self.clearance_weight
                            * math.exp(-next_distance / self.clearance_decay)
                        )

        # ---- 2) 새로 뚫린 셀: 국소 윈도우만 실제 장애물 기준 재계산 ----
        if newly_freed_cells:
            radius_cells = int(math.ceil(self.clearance_radius))

            window = set()
            for (fx, fz) in newly_freed_cells:
                for dx in range(-radius_cells, radius_cells + 1):
                    for dz in range(-radius_cells, radius_cells + 1):
                        node = (fx + dx, fz + dz)
                        if self.in_bounds(node):
                            window.add(node)

            source_radius = radius_cells * 2
            sources = set()
            for (fx, fz) in newly_freed_cells:
                for dx in range(-source_radius, source_radius + 1):
                    for dz in range(-source_radius, source_radius + 1):
                        node = (fx + dx, fz + dz)
                        if self.in_bounds(node) and node in self.obstacles:
                            sources.add(node)

            # 윈도우 안 값은 오래된 값이 남지 않도록 일단 지운다.
            for node in window:
                self.obstacle_distances.pop(node, None)
                self.clearance_costs.pop(node, None)

            if sources:
                heap = []
                local_distance = {}

                for node in sources:
                    local_distance[node] = 0.0
                    heapq.heappush(heap, (0.0, node))

                while heap:
                    current_distance, node = heapq.heappop(heap)

                    if current_distance != local_distance.get(node):
                        continue
                    if current_distance > self.clearance_radius:
                        continue

                    x, z = node

                    for dx, dz, step in directions:
                        neighbor = (x + dx, z + dz)

                        if not self.in_bounds(neighbor):
                            continue
                        if neighbor in self.obstacles:
                            continue

                        next_distance = current_distance + step

                        if next_distance > self.clearance_radius:
                            continue

                        if next_distance < local_distance.get(neighbor, INF):
                            local_distance[neighbor] = next_distance
                            heapq.heappush(heap, (next_distance, neighbor))

                            if neighbor in window:
                                self.obstacle_distances[neighbor] = next_distance
                                self.clearance_costs[neighbor] = (
                                    self.clearance_weight
                                    * math.exp(-next_distance / self.clearance_decay)
                                )

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

    def get_occupied_space(self, corners):
        """
        4개의 꼭짓점(실수 좌표) 내부 및 경계선에 위치한 모든 정수 격자 좌표(x, z)를 set으로 반환합니다.
        """
        occupied = set()
        n = len(corners)
        if n < 3:
            return occupied

        # 1. 바운딩 박스(최소/최대 정수 범위) 계산하여 탐색 구역 제한
        xs = [c[0] for c in corners]
        zs = [c[1] for c in corners]
        min_x = math.floor(min(xs))
        max_x = math.ceil(max(xs))
        min_z = math.floor(min(zs))
        max_z = math.ceil(max(zs))

        # 2. Ray Casting 알고리즘으로 내부 점 판별
        for z in range(min_z, max_z + 1):
            for x in range(min_x, max_x + 1):
                inside = False
                p1x, p1z = corners[0]
                
                for i in range(n + 1):
                    p2x, p2z = corners[i % n]
                    # 현재 정수 좌표 (x, z)가 다각형 변과 교차하는지 검사
                    if min(p1z, p2z) < z <= max(p1z, p2z):
                        if p1z != p2z:
                            x_inters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                            if p1x == p2x or x <= x_inters:
                                inside = not inside
                    p1x, p1z = p2x, p2z
                    
                if inside:
                    occupied.add((x, z))
                    
        return occupied

    def get_bb_corners(self, cx, cz, width, height, angle_degrees):
        rad = math.radians(angle_degrees)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        hx, hz = width / 2.0, height / 2.0
        
        local_corners = [(hx, hz), (hx, -hz), (-hx, -hz), (-hx, hz)]
        global_corners = []
        for lx, lz in local_corners:
            gx = lx * cos_a + lz * sin_a + cx
            gz = -lx * sin_a + lz * cos_a + cz
            global_corners.append((gx, gz))
        return global_corners


    def find_path(self, current_pos, dest, latest_info=None):
        """
        서버 호출:
            current_path = planner.find_path(current_pos, dest)
        """

        # self.movable_enemy_tank
        # 움직일 수 있는 적 전차 위치에 대한 변수 업데이트
        if latest_info != None:
            cx, cz = latest_info["enemyPos"]["x"], latest_info["enemyPos"]["z"]
            body_angle = latest_info["enemyBodyX"]
            turret_angle = latest_info["enemyTurretX"]

            body_corners = self.get_bb_corners(cx, cz, 3.303, 6.339, body_angle)
            turret_corners = self.get_bb_corners(cx, cz, 2.681, 2.822, turret_angle)

            body_tiles = self.get_occupied_space(body_corners)
            turret_tiles = self.get_occupied_space(turret_corners)

            base_enemy_space = body_tiles.union(turret_tiles)

            padded_enemy_space = set()
            for x, z in base_enemy_space:
                for dx in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        padded_enemy_space.add((x + dx, z + dz))

            self.movable_enemy_tank = padded_enemy_space

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

    def pad_object(
        self,
        x_min, x_max, z_min, z_max,
        obj_type='enemy_tank',
        dedupe_radius_m=None,
    ):
        """
        Unity의 /update_obstacle 콜백과 무관하게, 우리 쪽 인지 파이프라인이
        직접 확보한 오브젝트 world 좌표를 맵에 추가로 패딩 처리한다.

        set_obstacles()는 넘겨받은 리스트로 obstacle_rectangles 전체를
        교체하는 방식이라, 기존 목록에 이번 오브젝트만 append한 뒤
        다시 통째로 넘겨서 '추가' 효과를 낸다.

        중복 방지(dedupe):
            인식 파이프라인은 같은 오브젝트를 프레임마다 계속 다시
            보내주므로, 매번 무조건 append하면 obstacle_rectangles가
            무한정 늘어나 set_obstacles() 호출(=Dijkstra 재계산)이
            점점 느려진다.

            그래서 같은 obj_type의 기존 rect 중 "중심이 dedupe_radius_m
            이내로 가까운" 것이 있으면 새로 추가하지 않고, 그 rect를
            이번 좌표로 '갱신'한다(오브젝트가 살짝 움직였을 수 있으니
            최신 좌표를 반영). 진짜 새 오브젝트로 봐야 하면(다른 type,
            또는 멀리 떨어진 같은 type) 새로 append된다.

            트래킹 ID를 인식 파이프라인에서 넘겨줄 수 있게 되면, 거리
            기반 추정보다 ID 기반 매칭이 훨씬 정확하니 그쪽으로
            바꾸는 걸 권장한다.

        Parameters
        ----------
        dedupe_radius_m : float, optional
            같은 오브젝트로 볼 중심 간 거리 기준 [m]. 생략하면 이번에
            들어온 바운딩 박스의 대각선 길이를 기준으로 삼는다(오브젝트
            크기에 비례해서 자연스럽게 커지도록).

        Returns
        -------
        set[GridNode]
            이번 호출로 실제로 통행 가능/불가 상태가 바뀐 grid 셀들.
        """
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        z_min, z_max = min(z_min, z_max), max(z_min, z_max)

        new_cx = (x_min + x_max) / 2.0
        new_cz = (z_min + z_max) / 2.0

        if dedupe_radius_m is None:
            # 바운딩 박스 대각선 길이의 절반 정도를 기준으로 잡는다.
            # 너무 좁으면(고정값) 큰 오브젝트가 프레임마다 조금씩 흔들려
            # 잡힐 때 다른 오브젝트로 오인해서 계속 append될 수 있고,
            # 너무 넓으면 실제로 가까이 있는 서로 다른 오브젝트 둘을
            # 하나로 합쳐버릴 수 있다.
            dedupe_radius_m = math.hypot(x_max - x_min, z_max - z_min) / 2.0

        combined = list(self.obstacle_rectangles)
        matched_index = None

        for i, obs in enumerate(combined):
            if obs.type != obj_type:
                continue

            obs_cx = (obs.x_min + obs.x_max) / 2.0
            obs_cz = (obs.z_min + obs.z_max) / 2.0

            if math.hypot(new_cx - obs_cx, new_cz - obs_cz) <= dedupe_radius_m:
                matched_index = i
                break

        new_rect = ObstacleRect(
            x_min=x_min, x_max=x_max,
            z_min=z_min, z_max=z_max,
            type=obj_type,
        )

        if matched_index is None:
            combined.append(new_rect)
        else:
            # append가 아니라 '갱신' -> obstacle_rectangles 길이가
            # 늘어나지 않는다.
            combined[matched_index] = new_rect

        return self.set_obstacles(combined)

    def is_path_blocked(self, world_path):
        """
        world_path(waypoint 리스트)가 현재 obstacles/terrain_blocked 기준으로
        여전히 통행 가능한지 검사한다.

        world_path는 ultimate_one_pass_compression()이 만든 코너점만 남긴
        희소한 리스트이므로, waypoint 자체가 아니라 waypoint '사이 직선 구간'을
        _is_straight_line_walkable()로 재검사한다 (경로 압축과 동일 로직 재사용).
        pad_object() 등으로 obstacles가 바뀐 '직후'에 호출해야 최신 상태를 반영한다.
        """
        if not world_path or len(world_path) < 2:
            return False
        return any(
            not self._is_straight_line_walkable(p1, p2)
            for p1, p2 in zip(world_path, world_path[1:])
        )

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
                idx += 1
                
        print(f"총 {idx}개 노드의 고도 데이터가 넘파이 배열로부터 일괄 주입되었습니다.")

    # --------------------------------------------------
    # 장애물
    # --------------------------------------------------

    def update_y_from_nature(self):
        """
            자연물이 배치된 좌표에 고도를 업데이트 해준다.
            
            self.updated_y = List:[Tuple:(int, int, int)] (x, z, add_num)
            (자연물에 의해 고도가 업데이트 된 좌표를 위한 공간)

            나무와 바위에 대한 type 구분이 필요
            대충 사이즈로 판단하자 -> max-min >= 3.5 면 Rock으로 판단하고 진행하자

        최적화:
            set_obstacles()가 호출될 때마다(예: 적 전차 1대 새로 탐지될 때마다)
            매번 무조건 다시 불렸었는데, 원래 여기서 매번 (1) 디스크에서
            risk_layers.npz를 다시 읽고 (2) 90000칸 고도맵 전체를 리셋하고
            (3) nature(나무/바위) 오브젝트의 높이 보정을 처음부터 다시
            계산하고 있었다. 근데 지형(nature) 배치는 게임 도중 거의 안
            바뀌고, enemy_tank 같은 다른 타입 오브젝트가 추가된 것뿐이면
            높이 데이터는 지난번 계산 결과와 100% 동일하다.

            그래서 "지난번 호출 이후 nature 오브젝트 배치 자체가 실제로
            바뀌었는지"를 서명(시그니처) 비교로 먼저 확인하고, 안 바뀌었으면
            디스크 재로드도, 90000칸 순회도, 재계산도 전부 건너뛴다.
        """
        nature_signature = tuple(
            sorted(
                (obs.x_min, obs.x_max, obs.z_min, obs.z_max)
                for obs in self.obstacle_rectangles
                if obs.type == 'nature'
            )
        )

        if nature_signature == self._last_nature_signature:
            return

        self._last_nature_signature = nature_signature

        # 디스크에서 다시 읽지 않고 __init__에서 캐싱해둔 배열을 재사용한다.
        # (지형 자체는 정적 데이터라 파일이 실행 중 바뀔 일이 없다.)
        self.update_entire_heightmap(self._cached_base_heightmap)
        self.updated_y = []

        for obs in self.obstacle_rectangles: # Tree
            if obs.type == 'nature':
                if max((obs.x_max-obs.x_min), (obs.z_max-obs.z_min)) >= 3.5:
                    continue

                add_num = 5

                x_min = math.floor(obs.x_min)
                x_max = math.ceil(obs.x_max)
                z_min = math.floor(obs.z_min)
                z_max = math.ceil(obs.z_max)
    
                x_min = max(0, x_min)
                x_max = min(self.width - 1, x_max)
                z_min = max(0, z_min)
                z_max = min(self.height - 1, z_max)

                # 일단 무조건 insert
                for x in range(x_min, x_max + 1):
                    for z in range(z_min, z_max + 1):
                        self.y[(x, z)] = float(self.y.get((x, z), 0)+add_num)
                        self.updated_y.append((x,z))

        for obs in self.obstacle_rectangles: # Rock
            if obs.type == 'nature':
                if max((obs.x_max-obs.x_min), (obs.z_max-obs.z_min)) < 3.5:
                    continue

                add_num = 2
                
                x_min = math.floor(obs.x_min)
                x_max = math.ceil(obs.x_max)
                z_min = math.floor(obs.z_min)
                z_max = math.ceil(obs.z_max)
    
                x_min = max(0, x_min)
                x_max = min(self.width - 1, x_max)
                z_min = max(0, z_min)
                z_max = min(self.height - 1, z_max)

                # tree에 대한 고도가 더 높게 더해지기 때문에 
                # if (x, z) in self.updated_y로 확인하여 insert
                for x in range(x_min, x_max + 1):
                    for z in range(z_min, z_max + 1):
                        if (x, z) in self.updated_y:
                            continue

                        self.y[(x, z)] = float(self.y.get((x, z), 0)+add_num)
                        self.updated_y.append((x,z))

    def update_obstacles_type(self, target_list:List[Tuple[float, float, float, str]]):
        """
            오브젝트 타입 업데이트 (detect된 오브젝트들에 대해서 type update)
            target_list = [(x, y, z, name), ...]

            우리 인지 파이프라인이 탐지한 (x, y, z, class_name) 목록을 받아서,
            이미 등록된 obstacle_rectangles(Unity /update_obstacle로 온, 고정된
            지형/오브젝트) 중 이 좌표를 포함하는 게 있으면 그 타입을 name
            기준으로 재분류한다.

            좌표를 포함하는 기존 obstacle이 하나도 없으면(예: 실시간으로
            움직이는 적 전차처럼 Unity가 애초에 등록해주지 않는 동적
            오브젝트) 그 항목은 unmatched로 반환한다 -- 호출부가
            pad_object() 등으로 새 장애물을 만들어 대응해야 하는 대상이라는
            뜻이다.

            주의: ObstacleRect는 frozen dataclass라 기존 인스턴스의 .type을
            직접 못 바꾼다. 그래서 좌표는 그대로 두고 ObstacleRect.from_min_max()로
            다시 만들어서(그 안에서 아래에서 갱신한 전역 분류 리스트를 다시
            조회해 올바른 타입이 매겨진다) set_obstacles()에 반영한다.

        Returns
        -------
        changed_cells : set[GridNode]
            이번 재분류로 실제로 통행 가능/불가 상태가 바뀐 grid 셀들.
            재분류가 하나도 없었으면 빈 set.
        unmatched : List[Tuple[float, float, float, str]]
            좌표를 포함하는 기존 obstacle을 못 찾은 탐지 항목들.
        """
        global unknown_list, enemy_list, enemy_tank_list, team_list, team_tank_list

        chg_list = [] # 재분류된 (x_min, x_max, z_min, z_max, type) 목록
        unmatched = [] # 매칭되는 기존 obstacle이 없었던 탐지 항목들

        name_to_type = {
            'Human1': 'enemy',
            'Human2': 'enemy',
            'Tank1': 'enemy_tank',
            'Human3': 'team',
            'Tank2': 'team_tank',
        }

        for x, y, z, name in target_list:
            desired_type = name_to_type.get(name)
            matched = False

            for idx in range(len(self.obstacle_rectangles)):
                rect = self.obstacle_rectangles[idx]
                x_min = max(rect.x_min, 0)
                x_max = min(rect.x_max, self.width)
                z_min = max(rect.z_min, 0)
                z_max = min(rect.z_max, self.height)

                if not (x_min <= x <= x_max and z_min <= z <= z_max):
                    continue

                # 이 좌표를 포함하는 첫 obstacle을 기준으로 판단한다
                # (여러 개가 겹쳐 있는 경우는 고려하지 않음).
                matched = True

                if desired_type is not None and rect.type != desired_type:
                    chg_list.append((rect.x_min, rect.x_max, rect.z_min, rect.z_max, desired_type))

                break

            if not matched:
                unmatched.append((x, y, z, name))

        if not chg_list:
            return set(), unmatched

        # 전역 분류 리스트 갱신: 이번에 바뀐 좌표는 기존에 어느 리스트에
        # 있었든 일단 빼고, 새로 확정된 타입의 리스트에 다시 넣는다.
        chg_set = {dt[:4] for dt in chg_list}
        unknown_list = [dt for dt in unknown_list if dt not in chg_set]
        enemy_list = [dt for dt in enemy_list if dt not in chg_set]
        enemy_tank_list = [dt for dt in enemy_tank_list if dt not in chg_set]
        team_list = [dt for dt in team_list if dt not in chg_set]
        team_tank_list = [dt for dt in team_tank_list if dt not in chg_set]

        for dt in chg_list:
            if dt[4] == 'enemy':
                enemy_list.append(dt[:4])
            elif dt[4] == 'enemy_tank':
                enemy_tank_list.append(dt[:4])
            elif dt[4] == 'team':
                team_list.append(dt[:4])
            elif dt[4] == 'team_tank':
                team_tank_list.append(dt[:4])

        # 좌표는 그대로 두고, 방금 갱신한 전역 분류 리스트를 다시 조회해서
        # 타입만 새로 확정되도록 전체 obstacle_rectangles를 재생성한다.
        rebuilt = [
            ObstacleRect.from_min_max(r.x_min, r.x_max, r.z_min, r.z_max)
            for r in self.obstacle_rectangles
        ]
        changed_cells = self.set_obstacles(rebuilt)

        return changed_cells, unmatched


    def _is_visible_under_height(self, p1: Tuple[float, float], p2: Tuple[float, float], max_allowed_y: float, num: int) -> bool:
        """
        적에 대해서 기존 고도에 따른 이동 가능/불가능한 영역 설정을 위한 함수
        p1: 적 탱크의 중심 좌표(min, max 값의 중간값)
        p2: 반복문으로 계속 받아오는 값으로(범위를 의미) min~max까지의 좌표값
        max_allowed_y: min~max안의 중심 좌표에 대한 고도 값
        num: +해줄 상수

        성능 참고:
            enemy_tank 오브젝트 하나당 패딩 반경(±49칸)에 걸리는 후보 셀이
            최대 1만 개 안팎이고, 셀 하나마다 이 함수가 0.5 간격 raymarch로
            길게는 100회 넘게 world_to_grid()를 호출한다 (탐지 1건당
            world_to_grid 호출이 수십만 번까지 발생 -> 실측 dominant cost).
            여기서는 매번 clamp=True로 호출하는데, 클램핑 이후 좌표는
            반드시 grid 범위 안이라 world_to_grid() 내부의 in_bounds 검사가
            항상 통과하는 중복 작업이다. 그래서 그 부분만 인라인으로 풀어
            함수 호출/중복 검사 오버헤드를 없앴다 — 결과값은 완전히 동일하다.
        """
        x1, z1 = p1
        x2, z2 = p2
        dist = math.hypot(x2 - x1, z2 - z1)
        if dist == 0: 
            return True
        
        width = self.width
        height = self.height
        y_get = self.y.get

        target_ix = min(max(int(round(x2)), 0), width - 1)
        target_iz = min(max(int(round(z2)), 0), height - 1)
        target_grid_y = y_get((target_ix, target_iz), 0)
        
        step_size = 0.5
        steps = int(dist / step_size)
        dx = x2 - x1
        dz = z2 - z1
        int_x1 = int(x1)
        int_z1 = int(z1)

        for i in range(1, steps):
            t = i / steps
            cx = x1 + dx * t
            cz = z1 + dz * t

            ix = min(max(int(round(cx)), 0), width - 1)
            iz = min(max(int(round(cz)), 0), height - 1)

            if (ix == target_ix and iz == target_iz) or (ix == int_x1 and iz == int_z1):
                continue

            curr_y = y_get((ix, iz), 0)

            # 고도에 따른 시야에 대한 부분
            # max_allowed_y: 오브젝트의 센터에 대한 고도값
            # target_grid_y: 오브젝트의 범위 만큼 반복하는 (x,z)에 대한 고도값
            # curr_y: 오브젝트와 반복문의 좌표를 이은 선에 대해 중간에 존재하는 고도값
            if curr_y >= (max_allowed_y+num): # 중간에 존재하는 고도값이 오브젝트가 존재하는 고도값 + num 보다 높다면
                if (target_grid_y+num-1) <= curr_y:
                    return False
                if target_grid_y <= (curr_y+0.5) and math.hypot(x2 - ix, z2 - iz) >= 6:
                    # 중간의 고도가 target(x,z)의 고도보다 살짝 낮아도 거리 차이가 6이상이라면 지나갈 수 있도록
                    return False
                    
        return True
    
    def set_obstacles(self, obs_list: Iterable[ObstacleRect]):
        """
        서버 호출:
            changed_cells = planner.set_obstacles(obs_list)
        """
        self.obstacle_rectangles = list(obs_list)
        new_obstacles = set()

        self.update_y_from_nature() # 자연물에 의한 고도 정보 업데이트

        # TODO
        # obs.type에 따른 로직이 이부분에 들어가야 한다.
        for obs in self.obstacle_rectangles:
            x_min = math.floor(obs.x_min - self.obstacle_margin)
            x_max = math.ceil(obs.x_max + self.obstacle_margin)
            z_min = math.floor(obs.z_min - self.obstacle_margin)
            z_max = math.ceil(obs.z_max + self.obstacle_margin)

            add_num = 0 # 오브젝트 범위에 대한 설정
            if obs.type == 'enemy':
                add_num = 25 # 일단 단순히 오브젝트에 대한 범위를 +-25
            elif obs.type == 'enemy_tank':
                add_num = 49 # 일단 단순히 오브젝트에 대한 범위를 +-49
            elif obs.type == 'team':
                pass
            elif obs.type == 'team_tank':
                pass

            x_min = max(0, x_min)
            x_max = min(self.width - 1, x_max)
            z_min = max(0, z_min)
            z_max = min(self.height - 1, z_max)

            # 타입에 따른 로직에 대한 설정
            if obs.type == 'enemy': 
                # target_y = max(self.y[(x, z)] for x in range(x_min, x_max + 1) for z in range(z_min, z_max + 1))

                cx = int((x_min + x_max) / 2)
                cz = int((z_min + z_max) / 2)
                target_y = self.y.get((cx, cz), 0) # 중심 좌표에 대한 고도

                tmp_x_min = max(0, x_min-add_num)
                tmp_x_max = min(self.width - 1, x_max+add_num)
                tmp_z_min = max(0, z_min-add_num)
                tmp_z_max = min(self.height - 1, z_max+add_num)
                for x in range(tmp_x_min, tmp_x_max + 1):
                    for z in range(tmp_z_min, tmp_z_max + 1):
                        # 기본 enemy 사이즈만큼은 무조건 add
                        if x_min <= x <= x_max and z_min <= z <= z_max:
                            new_obstacles.add((x, z))
                            continue

                        if self._is_visible_under_height((cx, cz), (x, z), target_y, 2):
                            new_obstacles.add((x, z))
            elif obs.type == 'enemy_tank':
                # target_y = max(self.y[(x, z)] for x in range(x_min, x_max + 1) for z in range(z_min, z_max + 1))
                
                cx = int((x_min + x_max) / 2)
                cz = int((z_min + z_max) / 2)
                target_y = self.y.get((cx, cz), 0) # 중심 좌표에 대한 고도

                tmp_x_min = max(0, x_min-add_num)
                tmp_x_max = min(self.width - 1, x_max+add_num)
                tmp_z_min = max(0, z_min-add_num)
                tmp_z_max = min(self.height - 1, z_max+add_num)
                for x in range(tmp_x_min, tmp_x_max + 1):
                    for z in range(tmp_z_min, tmp_z_max + 1):
                        # 기본 enemy 사이즈만큼은 무조건 add
                        if x_min <= x <= x_max and z_min <= z <= z_max:
                            new_obstacles.add((x, z))
                            continue

                        if self._is_visible_under_height((cx, cz), (x, z), target_y, 3):
                            new_obstacles.add((x, z))
            # elif obs.type == 'team':
            #     pass
            # elif obs.type == 'team_tank':
            #     pass
            else:
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

        # 장애물 주변의 edge cost도 함께 바뀌므로 soft costmap을 갱신한다.
        # 예전엔 매번 self.obstacles 전체로 rebuild_clearance_costs()를
        # 새로 돌렸는데(300x300 전체 Dijkstra), changed_cells는 이미 위에서
        # 정확히 계산돼 있으니 그 증분만 처리하는 게 훨씬 빠르다.
        newly_obstacle_cells = changed_cells & self.obstacles
        newly_freed_cells = changed_cells - self.obstacles
        self.rebuild_clearance_costs_incremental(
            newly_obstacle_cells, newly_freed_cells,
        )

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

        # loc='best'(기본값)는 obstacle_rectangles 수백 개 + imshow 전체를
        # 상대로 겹치지 않는 위치를 전수 탐색해서 실제로 수십 초 이상
        # 걸릴 수 있다(legend에 'Creating legend with loc="best" can be
        # slow' 경고가 뜨는 이유). 위치를 고정해서 그 탐색 자체를 없앤다.
        ax.legend(loc='upper right')
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

        for x,z in self.obstacles:
            patch = Rectangle(
                ((x+0.5) - 1.0 * 0.5, (z+0.5) - 1.0 * 0.5), # 사각형의 시작점(좌측 하단)
                1.0, 1.0,
                alpha=0.5,
            )
            ax.add_patch(patch)

        color = '#228B22'
        for obs in obstacle_rectangles:
            if obs.type == 'nature':
                color = '#228B22'
            elif obs.type == 'unknown':
                color = '#000000'
            elif obs.type == 'enemy_tank':
                color = '#800020'
            elif obs.type == 'enemy':
                color = '#FF8C00'
            elif obs.type == 'team':
                color = '#8A2BE2'
            elif obs.type == 'team_tank':
                color = '#000080'

            patch = Rectangle(
                (
                    obs.x_min - self.obstacle_margin,
                    obs.z_min - self.obstacle_margin,
                ),
                (obs.x_max - obs.x_min) + 2 * self.obstacle_margin,
                (obs.z_max - obs.z_min) + 2 * self.obstacle_margin,
                facecolor=color, 
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

        # loc='best'(기본값)는 obstacle_rectangles 수백 개 + imshow 전체를
        # 상대로 겹치지 않는 위치를 전수 탐색해서 실제로 수십 초 이상
        # 걸릴 수 있다(legend에 'Creating legend with loc="best" can be
        # slow' 경고가 뜨는 이유). 위치를 고정해서 그 탐색 자체를 없앤다.
        ax.legend(loc='upper right')
        fig.tight_layout()

        if save_path:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.print_figure(output, dpi=150, bbox_inches="tight")
            print("grid 저장 완료 (백그라운드)")

        return fig, ax


# 기존 이름 호환
DStarLite = DStarPlanner