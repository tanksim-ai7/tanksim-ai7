"""
navigation_controller.py
========================

D* Lite 경로계획 상태를 서버 코드에서 분리하기 위한 모듈.

이 모듈이 담당하는 것
--------------------
- DStarPlanner 객체 생성/보관
- 현재 위치(current_pos) 보관
- 목적지(dest) 보관
- 현재 경로(current_path) 보관
- 목적지 설정 후 D* Lite 경로 계산
- 장애물 갱신 후 D* Lite 재계획
- 시작점이 장애물 영역에 포함된 경우 예외 처리
- 경로 맵 저장


따라서 기본 server_sample은 그대로 두고,
통합할 때 이 모듈의 NavigationController 객체만 생성해서 사용하면 된다.
"""

from move.dstar_lite_planner_cost import DStarPlanner, ObstacleRect

# 목적지가 새로 설정될 때 이전 PID 누적 상태를 초기화하기 위해 사용한다.
# PID 계산 자체는 navigation_controller가 하지 않는다.
from move.pid_controller import reset_pid_controllers


class NavigationController:
    """
    D* Lite navigation 상태와 관련 함수를 하나의 객체로 관리한다.

    기존 서버 코드의 전역변수:
        planner
        current_pos
        dest
        current_path

    기존 서버 함수:
        apply_destination()
        clear_start_area()
        render_map()
        update_obstacles_from_payload()

    위 항목들을 이 클래스 안으로 이동시킨 구조다.
    """

    def __init__(
        self,
        start=(0, 0),
        goal=(299, 299),
        width=300,
        height=300,
        allow_diagonal=True,
        obstacle_margin=2.0,
        clearance_radius=8.0,
        clearance_weight=4.0,
        clearance_decay=2.5,
        map_image_path="dstar_map.png",
    ):
        """
        NavigationController 초기화.

        Parameters
        ----------
        start:
            D* Lite 초기 시작 grid 좌표.
            실제 차량 위치가 들어오면 find_path() 호출 시 다시 반영된다.

        goal:
            D* Lite 초기 goal grid 좌표.
            실제 목적지가 설정되면 다시 반영된다.

        width, height:
            전체 grid map 크기.

        allow_diagonal:
            대각선 이동 허용 여부.

        obstacle_margin:
            장애물 주변을 통행 불가로 만드는 hard margin.

        clearance_radius:
            hard margin 바깥에서 soft clearance cost를 적용하는 반경.

        clearance_weight:
            장애물 근처를 피하려는 추가 비용의 강도.

        clearance_decay:
            장애물과 멀어질수록 clearance cost가 감소하는 정도.

        map_image_path:
            planner.plot() 결과를 저장할 파일 경로.
        """

        # D* Lite 경로계획 객체.
        self.planner = DStarPlanner(
            start=start,
            goal=goal,
            width=width,
            height=height,
            allow_diagonal=allow_diagonal,
            obstacle_margin=obstacle_margin,
            clearance_radius=clearance_radius,
            clearance_weight=clearance_weight,
            clearance_decay=clearance_decay,
        )

        # 서버 /info 또는 /get_action에서 받은 현재 전차 위치 [x, z].
        self.current_pos = None

        # 현재 목적지 [x, z].
        self.dest = None

        # planner.find_path()가 계산한 현재 D* Lite 경로.
        self.current_path = []

        # 경로 시각화 파일 저장 위치.
        self.map_image_path = map_image_path

    # ========================================================
    # 현재 위치
    # ========================================================

    def set_current_position(self, position):
        """
        서버가 받은 현재 전차 위치를 navigation module에 저장한다.

        Parameters
        ----------
        position:
            [x, z] 또는 (x, z)

        Returns
        -------
        list[float]
            저장된 [x, z]
        """

        if position is None:
            self.current_pos = None
            return None

        if len(position) < 2:
            raise ValueError(
                "position must contain x and z"
            )

        self.current_pos = [
            float(position[0]),
            float(position[1]),
        ]

        return self.current_pos

    # ========================================================
    # 시작점 장애물 예외 처리
    # ========================================================

    def clear_start_area(
        self,
        position=None,
        radius=2,
    ):
        """
        현재 시작점이 hard obstacle 안에 포함된 경우에만
        시작점 주변 obstacle cell을 제거한다.

        기존 server 코드의 clear_start_area()를
        planner 전역변수 없이 사용할 수 있도록 메서드화했다.
        """

        # position을 따로 주지 않으면 현재 위치 사용.
        if position is None:
            position = self.current_pos

        if position is None:
            return set()

        # world 좌표를 D* Lite grid 좌표로 변환.
        center = self.planner.world_to_grid(
            position,
            clamp=True,
        )

        # 실제 제거된 obstacle cell 기록.
        changed = set()

        # 현재 위치 주변 radius 범위를 검사.
        for dx in range(
            -radius,
            radius + 1,
        ):
            for dz in range(
                -radius,
                radius + 1,
            ):
                cell = (
                    center[0] + dx,
                    center[1] + dz,
                )

                # map 안에 존재하면서 obstacle로 등록된 cell만 제거.
                if (
                    self.planner.in_bounds(cell)
                    and cell in self.planner.obstacles
                ):
                    self.planner.obstacles.discard(
                        cell
                    )
                    changed.add(cell)

        # obstacle cell이 실제 변경됐으면
        # soft clearance cost도 다시 계산해야 한다.
        if changed:
            self.planner.refresh_costmap()

        return changed

    # ========================================================
    # 경로 시각화
    # ========================================================

    def render_map(self, title):
        """
        현재 planner 상태와 current_path를 이미지 파일로 저장한다.

        주행 제어 자체에는 필수가 아니며,
        디버깅/경로 확인용 기능이다.
        """

        self.planner.plot(
            path=(
                self.current_path
                if self.current_path
                else None
            ),
            show_grid=True,
            title=title,
            save_path=self.map_image_path,
            show=False,
        )

        return self.map_image_path

    # ========================================================
    # 목적지 설정 + 경로 계산
    # ========================================================

    def apply_destination(
        self,
        x,
        y,
        z,
        render=True,
    ):
        """
        목적지를 설정하고 현재 위치에서 목적지까지
        D* Lite 경로를 계산한다.

        기존 server 코드의 apply_destination() 역할을 담당한다.

        y는 시뮬레이터 API 형식을 유지하기 위해 받지만
        현재 D* Lite는 XZ 평면을 사용하므로 경로 계산에는 사용하지 않는다.
        """

        if self.current_pos is None:
            raise ValueError(
                "Current position is not received yet"
            )

        # D* Lite는 XZ 평면에서 동작하므로 x, z만 저장.
        self.dest = [
            float(x),
            float(z),
        ]

        # 새 목적지에서는 이전 PID의 적분/미분 상태가
        # 남아 있으면 안 되므로 PID 상태 초기화.
        reset_pid_controllers()

        # 현재 위치가 obstacle margin 내부로 잡힌 경우를 방지.
        self.clear_start_area(
            self.current_pos,
            radius=2,
        )

        # 현재 위치 -> 목적지 경로 계산.
        self.current_path = (
            self.planner.find_path(
                self.current_pos,
                self.dest,
            )
        )

        # 필요할 때만 경로 이미지 생성.
        if render:
            self.render_map(
                "D* Lite Demo (300X300)"
            )

        return {
            "status": "OK",
            "destination": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            },
            "path_done_count": len(
                self.current_path
            ),
            "path_cost": (
                self.planner.get_path_cost(
                    self.current_path
                )
            ),
            "map_path": self.map_image_path,
        }

    # ========================================================
    # 장애물 갱신
    # ========================================================

    def update_obstacles_from_payload(
        self,
        payload,
        replan=True,
        render=True,
    ):
        """
        Unity /update_obstacle 형식의 payload를 받아
        D* Lite planner obstacle로 변환한다.

        payload 예상 형식:
        {
            "obstacles": [
                {
                    "x_min": ...,
                    "x_max": ...,
                    "z_min": ...,
                    "z_max": ...
                }
            ]
        }

        replan=True:
            현재 위치와 목적지가 존재하면
            장애물 반영 직후 경로를 다시 계산한다.

        render=True:
            갱신 후 경로 맵을 이미지로 저장한다.
        """

        if payload is None:
            raise ValueError(
                "Obstacle payload is None"
            )

        # Unity payload의 사각형 장애물을
        # D* Lite가 사용하는 ObstacleRect 목록으로 변환.
        obs_list = []

        for item in payload.get(
            "obstacles",
            [],
        ):
            obs = ObstacleRect.from_min_max(
                x_min=item["x_min"],
                x_max=item["x_max"],
                z_min=item["z_min"],
                z_max=item["z_max"],
            )

            obs_list.append(obs)

        # planner 내부 obstacle map 갱신.
        changed_cells = (
            self.planner.set_obstacles(
                obs_list
            )
        )

        # obstacle 갱신 후 시작점이 obstacle 안에 들어간
        # 예외상황을 다시 정리한다.
        self.clear_start_area(
            self.current_pos,
            radius=2,
        )

        replanned = False

        # 현재 위치와 목적지가 모두 있을 때만
        # 새 장애물을 반영한 경로 재계산 가능.
        if (
            replan
            and self.current_pos is not None
            and self.dest is not None
        ):
            try:
                self.current_path = (
                    self.planner.find_path(
                        self.current_pos,
                        self.dest,
                    )
                )
                replanned = True

            except ValueError:
                # 경로를 찾지 못하면 이전 경로를 계속 쓰지 않도록 비운다.
                self.current_path = []

        elif replan:
            # 위치 또는 목적지가 없는 상태에서는
            # 아직 계산할 경로가 없음.
            self.current_path = []

        # 경로/장애물 상태 확인용 이미지 저장.
        if render:
            if replanned:
                title = "D* Lite Replanning"
            else:
                title = "D* Lite Obstacle Map"

            self.render_map(title)

        return {
            "status": "success",
            "changed_cells": changed_cells,
            "changed_cell_count": len(
                changed_cells
            ),
            "path_length": len(
                self.current_path
            ),
            "obstacle_count": len(
                self.planner.obstacle_rectangles
            ),
            "replanned": replanned,
            "map_path": self.map_image_path,
        }

    # ========================================================
    # 명시적 재계획
    # ========================================================

    def replan(self, render=False):
        """
        현재 current_pos와 dest를 이용해
        D* Lite 경로를 다시 계산한다.
        """

        if self.current_pos is None:
            raise ValueError(
                "Current position is not received yet"
            )

        if self.dest is None:
            raise ValueError(
                "Destination is not set"
            )

        self.clear_start_area(
            self.current_pos,
            radius=2,
        )

        self.current_path = (
            self.planner.find_path(
                self.current_pos,
                self.dest,
            )
        )

        if render:
            self.render_map(
                "D* Lite Replanning"
            )

        return self.current_path

    # ========================================================
    # 서버에서 읽기 쉽게 getter 제공
    # ========================================================

    def get_path(self):
        """
        현재 D* Lite 경로 반환.
        """
        return self.current_path

    def get_destination(self):
        """
        현재 목적지 [x, z] 반환.
        """
        return self.dest

    def get_current_position(self):
        """
        현재 전차 위치 [x, z] 반환.
        """
        return self.current_pos

    def get_path_cost(self):
        """
        현재 경로의 D* Lite 비용 반환.
        """
        if not self.current_path:
            return 0.0

        return self.planner.get_path_cost(
            self.current_path
        )

    def get_map_path(self):
        """
        render_map()이 저장하는 이미지 파일 경로 반환.
        """
        return self.map_image_path

    # ========================================================
    # 전체 Navigation 상태 초기화
    # ========================================================

    def reset_navigation(
        self,
        clear_position=False,
    ):
        """
        목적지/현재 경로/PID 상태를 초기화한다.

        clear_position=False:
            현재 위치는 유지하고 목적지와 경로만 초기화.

        clear_position=True:
            current_pos까지 None으로 초기화.
        """

        self.dest = None
        self.current_path = []

        if clear_position:
            self.current_pos = None

        reset_pid_controllers()
