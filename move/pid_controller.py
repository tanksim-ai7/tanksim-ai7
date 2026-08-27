"""
pid_controller_refactored.py
============================

Tank Challenge 데모용 주행 제어 모듈.

목적
----
기존 server_sample_v2.3.10_dstarlite_corner_proportional_v7 노트북에
직접 들어가 있던 D* Lite 경로 추종, 속도 PID, 조향 PID, 코너 감속,
목적지 관리, 장애물 재계획 관련 코드를 Flask 서버에서 분리한다.

Flask 서버는 TankDriveController 인스턴스를 하나 만든 뒤 아래 메서드만 호출하면 된다.

    handle_info(data)
    get_action(data)
    handle_set_destination(data)
    handle_update_obstacles(data)
    initialize(start_position, destination)
    render_map(title)

현재 제어식과 주요 파라미터는 업로드된
server_sample_v2.3.10_dstarlite_corner_proportional_v7 (6)(2).ipynb 기준으로 유지했다.
"""

import math
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from move.dstar_lite_planner_cost import ObstacleRect
from move.risk_planner import RiskDStarPlanner as DStarLitePlanner


# ============================================================
# 공통 타입 설명
# ============================================================

# X-Z 평면상의 월드 좌표 [m].
PointXZ = Tuple[float, float]


# ============================================================
# 공통 수학 보조 함수
# ============================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    값을 minimum ~ maximum 범위로 제한한다.

    Args:
        value:
            제한할 원본 값.
        minimum:
            허용할 최소값.
        maximum:
            허용할 최대값.

    Returns:
        minimum <= 결과 <= maximum 인 실수.
    """
    return max(minimum, min(maximum, value))


def normalize_angle_deg(angle: float) -> float:
    """
    각도를 -180 ~ +180 deg 범위로 정규화한다.

    Args:
        angle:
            정규화할 각도 [deg].

    Returns:
        -180 <= angle < 180 범위의 각도 [deg].
    """
    return (float(angle) + 180.0) % 360.0 - 180.0


def extract_speed_from_info(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """
    /info JSON에서 시뮬레이터가 직접 제공하는 속도값을 읽는다.

    시뮬레이터가 제공하는 속도값의 단위를 m/s로 간주하고 km/h로 변환한다.

    Args:
        data:
            /info endpoint에서 받은 JSON dictionary.

    Returns:
        (speed_kmh, source_key)
        사용할 수 있는 속도 필드가 없으면 (None, None).
    """
    # 시뮬레이터 버전에 따라 속도 key 이름이 다를 수 있어 순서대로 확인한다.
    keys = ("PlayerSpeed", "playerSpeed", "speed", "velocity")

    for key in keys:
        if key in data and data[key] is not None:
            try:
                # 후진 속도가 음수여도 속력의 크기만 사용하고 m/s -> km/h로 변환한다.
                return abs(float(data[key])) * 3.6, key
            except (TypeError, ValueError):
                pass

    return None, None


def project_position_to_segment(
    current_position: Sequence[float],
    segment_start: Sequence[float],
    segment_end: Sequence[float],
) -> Tuple[PointXZ, float, float]:
    """
    현재 차량 위치를 D* Lite path segment 위에 투영한다.

    Args:
        current_position:
            현재 차량 X-Z 월드 좌표 [m].
        segment_start:
            segment 시작 X-Z 좌표 [m].
        segment_end:
            segment 끝 X-Z 좌표 [m].

    Returns:
        projection:
            segment 위 투영점 [m].
        t:
            segment 내부 비율. 0은 시작점, 1은 끝점.
        distance:
            현재 위치와 projection 사이 거리 [m].
    """
    cx, cz = map(float, current_position)
    x1, z1 = map(float, segment_start)
    x2, z2 = map(float, segment_end)

    # segment 방향 벡터.
    segment_dx = x2 - x1
    segment_dz = z2 - z1

    # segment 길이의 제곱 [m^2].
    segment_length_sq = (
        segment_dx * segment_dx
        + segment_dz * segment_dz
    )

    # 시작점과 끝점이 같은 비정상 segment 처리.
    if segment_length_sq <= 1e-12:
        projection = (x1, z1)
        distance = math.hypot(cx - x1, cz - z1)
        return projection, 0.0, distance

    # 무한 직선 기준 투영 비율.
    raw_t = (
        (cx - x1) * segment_dx
        + (cz - z1) * segment_dz
    ) / segment_length_sq

    # 실제 유한 segment 내부로 제한한다.
    t = clamp(raw_t, 0.0, 1.0)

    projection_x = x1 + t * segment_dx
    projection_z = z1 + t * segment_dz

    distance = math.hypot(
        cx - projection_x,
        cz - projection_z,
    )

    return (projection_x, projection_z), t, distance


def has_passed_segment_end(
    current_position: Sequence[float],
    segment_start: Sequence[float],
    segment_end: Sequence[float],
) -> bool:
    """
    차량이 segment 진행 방향 기준으로 끝점을 실제로 통과했는지 판단한다.

    Args:
        current_position:
            현재 차량 X-Z 좌표 [m].
        segment_start:
            segment 시작점 [m].
        segment_end:
            segment 끝점 [m].

    Returns:
        True:
            segment 끝점을 진행 방향으로 통과함.
        False:
            아직 끝점 이전에 있음.
    """
    cx, cz = map(float, current_position)
    x1, z1 = map(float, segment_start)
    x2, z2 = map(float, segment_end)

    segment_dx = x2 - x1
    segment_dz = z2 - z1

    # 현재 위치가 끝점보다 진행방향 쪽에 있는지 dot product로 판정한다.
    pass_dot = (
        (cx - x2) * segment_dx
        + (cz - z2) * segment_dz
    )

    return pass_dot > 0.0


def select_lookahead_point(
    path: Sequence[Sequence[float]],
    current_position: Sequence[float],
    lookahead_distance: float,
) -> Tuple[Optional[PointXZ], Optional[int]]:
    """
    D* Lite 압축 경로에서 현재 active segment를 찾고 look-ahead target을 계산한다.

    현재 서버 복구본과 동일하게 look-ahead target은 현재 segment의 다음 vertex를
    넘어가지 않는다.

    Args:
        path:
            D* Lite가 반환한 압축 X-Z 경로.
        current_position:
            현재 차량 X-Z 좌표 [m].
        lookahead_distance:
            현재 위치에서 경로 진행 방향으로 바라볼 거리 [m].

    Returns:
        target_point:
            조향 PID가 바라볼 X-Z 좌표 [m].
        target_index:
            현재 active segment의 다음 D* Lite vertex index.
    """
    if path is None or len(path) == 0:
        return None, None

    if len(path) == 1:
        return (
            (float(path[0][0]), float(path[0][1])),
            0,
        )

    # 차량과 가장 가까운 path segment를 찾는다.
    nearest_segment_index = None
    nearest_segment_distance = float("inf")

    for segment_index in range(len(path) - 1):
        _, _, distance_to_segment = project_position_to_segment(
            current_position,
            path[segment_index],
            path[segment_index + 1],
        )

        if distance_to_segment < nearest_segment_distance:
            nearest_segment_distance = distance_to_segment
            nearest_segment_index = segment_index

    if nearest_segment_index is None:
        return None, None

    segment_index = nearest_segment_index

    # 기하학적으로 다음 segment가 더 가까워졌더라도
    # 이전 vertex를 실제 진행방향으로 통과하기 전에는 이전 segment를 유지한다.
    while segment_index > 0:
        previous_start = path[segment_index - 1]
        previous_end = path[segment_index]

        if has_passed_segment_end(
            current_position,
            previous_start,
            previous_end,
        ):
            break

        segment_index -= 1

    # 현재 segment 끝점을 이미 통과했다면 다음 segment로 이동한다.
    while segment_index < len(path) - 2:
        segment_start = path[segment_index]
        segment_end = path[segment_index + 1]

        if not has_passed_segment_end(
            current_position,
            segment_start,
            segment_end,
        ):
            break

        segment_index += 1

    segment_start = path[segment_index]
    segment_end = path[segment_index + 1]

    x1, z1 = map(float, segment_start)
    x2, z2 = map(float, segment_end)

    projection, _, _ = project_position_to_segment(
        current_position,
        segment_start,
        segment_end,
    )

    projection_x, projection_z = projection

    segment_dx = x2 - x1
    segment_dz = z2 - z1
    segment_length = math.hypot(
        segment_dx,
        segment_dz,
    )

    if segment_length <= 1e-12:
        return (
            (x2, z2),
            segment_index + 1,
        )

    unit_dx = segment_dx / segment_length
    unit_dz = segment_dz / segment_length

    # projection에서 현재 segment 끝점까지 남은 거리 [m].
    distance_to_segment_end = math.hypot(
        x2 - projection_x,
        z2 - projection_z,
    )

    # 복구된 현재 로직:
    # look-ahead가 길어도 다음 D* Lite vertex를 넘어가지 않는다.
    target_advance_distance = min(
        max(0.0, float(lookahead_distance)),
        distance_to_segment_end,
    )

    target_x = (
        projection_x
        + unit_dx * target_advance_distance
    )

    target_z = (
        projection_z
        + unit_dz * target_advance_distance
    )

    target_point = (
        target_x,
        target_z,
    )

    target_index = segment_index + 1

    return target_point, target_index


def calculate_next_vertex_corner(
    path: Sequence[Sequence[float]],
    current_position: Sequence[float],
    target_index: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    현재 조향이 추종하는 다음 D* Lite vertex의 코너 정보를 계산한다.

    Args:
        path:
            압축 D* Lite 경로.
        current_position:
            현재 차량 X-Z 좌표 [m].
        target_index:
            select_lookahead_point()가 반환한 다음 vertex index.

    Returns:
        None:
            앞/뒤 segment를 만들 수 없을 때.
        dict:
            distance [m], angle [deg], radius [m], index, point를 포함한다.
    """
    if path is None or target_index is None:
        return None

    if (
        target_index <= 0
        or target_index >= len(path) - 1
    ):
        return None

    previous_point = path[target_index - 1]
    corner_point = path[target_index]
    next_point = path[target_index + 1]

    previous_x, previous_z = map(float, previous_point)
    corner_x, corner_z = map(float, corner_point)
    next_x, next_z = map(float, next_point)

    # 현재 차량 위치를 코너 진입 segment 위에 투영한다.
    projection, _, _ = project_position_to_segment(
        current_position,
        previous_point,
        corner_point,
    )

    projection_x, projection_z = projection

    # 현재 segment 투영점에서 코너 vertex까지 남은 거리 [m].
    distance_to_corner = math.hypot(
        corner_x - projection_x,
        corner_z - projection_z,
    )

    # +Z = 0 deg, +X = +90 deg 좌표계 기준 진입/진출 heading.
    heading_before = math.degrees(
        math.atan2(
            corner_x - previous_x,
            corner_z - previous_z,
        )
    )

    heading_after = math.degrees(
        math.atan2(
            next_x - corner_x,
            next_z - corner_z,
        )
    )

    turn_angle = abs(
        normalize_angle_deg(
            heading_after - heading_before
        )
    )

    # 세 경로점을 이용한 기하학적 외접원 반경 계산.
    # 현재 제어에서는 진단값으로만 사용하고 코너 속도식에는 직접 사용하지 않는다.
    segment_before_length = math.hypot(
        corner_x - previous_x,
        corner_z - previous_z,
    )

    segment_after_length = math.hypot(
        next_x - corner_x,
        next_z - corner_z,
    )

    chord_length = math.hypot(
        next_x - previous_x,
        next_z - previous_z,
    )

    double_triangle_area = abs(
        (corner_x - previous_x)
        * (next_z - previous_z)
        - (corner_z - previous_z)
        * (next_x - previous_x)
    )

    if double_triangle_area <= 1e-9:
        corner_radius_m = float("inf")
    else:
        corner_radius_m = (
            segment_before_length
            * segment_after_length
            * chord_length
            / (
                2.0
                * double_triangle_area
            )
        )

    return {
        "distance": distance_to_corner,
        "angle": turn_angle,
        "radius": corner_radius_m,
        "index": target_index,
        "point": (
            corner_x,
            corner_z,
        ),
    }


def make_stop_command() -> Dict[str, Any]:
    """
    모든 주행/조향 입력을 해제한 안전 정지 명령을 반환한다.

    Returns:
        Flask 서버가 그대로 jsonify할 수 있는 command dictionary.
    """
    return {
        "moveWS": {"command": "", "weight": 0.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }


def make_longitudinal_command(pid_output: float) -> Dict[str, Any]:
    """
    속도 PID 출력을 W/S 명령으로 변환한다.

    Args:
        pid_output:
            -1 ~ +1 범위의 속도 PID 출력.
            양수는 W 전진 가속, 음수는 S 제동/후진 방향 입력.

    Returns:
        moveWS가 채워진 command dictionary.
    """
    # PID 출력이 아주 작을 때 W/S가 반복 전환되는 것을 막는 명령 deadband.
    deadband = 0.02

    if pid_output > deadband:
        ws_command = "W"
        ws_weight = clamp(pid_output, 0.0, 1.0)

    elif pid_output < -deadband:
        ws_command = "S"
        ws_weight = clamp(abs(pid_output), 0.0, 1.0)

    else:
        ws_command = ""
        ws_weight = 0.0

    return {
        "moveWS": {
            "command": ws_command,
            "weight": round(ws_weight, 4),
        },
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }


# ============================================================
# 범용 PID 제어기
# ============================================================

class PIDController:
    """
    속도 PID와 조향 PD에서 공통으로 사용하는 PID Controller.

    모든 상태는 객체 내부에 저장하므로 서버 전역변수로 둘 필요가 없다.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -1.0,
        output_max: float = 1.0,
        integral_min: float = -10.0,
        integral_max: float = 10.0,
    ) -> None:
        """
        PID gain과 출력/적분 제한값을 초기화한다.

        Args:
            kp:
                비례 gain.
            ki:
                적분 gain.
            kd:
                미분 gain.
            output_min:
                최종 PID 출력 최소값.
            output_max:
                최종 PID 출력 최대값.
            integral_min:
                integral wind-up 방지용 적분 누적 최소값.
            integral_max:
                integral wind-up 방지용 적분 누적 최대값.
        """
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.output_min = float(output_min)
        self.output_max = float(output_max)

        self.integral_min = float(integral_min)
        self.integral_max = float(integral_max)

        # 누적된 error * dt.
        self.integral = 0.0

        # 직전 제어 주기의 error. D항 계산에 사용한다.
        self.previous_error: Optional[float] = None

    def reset(self) -> None:
        """
        목적지 변경/초기화 시 PID 내부 상태를 비운다.
        """
        self.integral = 0.0
        self.previous_error = None

    def update(self, error: float, dt: float) -> float:
        """
        현재 error와 dt로 PID 출력을 계산한다.

        Args:
            error:
                목표값 - 현재값.
            dt:
                제어 주기 [sec].

        Returns:
            output_min ~ output_max 범위의 PID 출력.
        """
        # 지나치게 작은 dt에서 derivative가 폭발하지 않도록 최소값을 둔다.
        dt = max(float(dt), 1e-3)

        self.integral = clamp(
            self.integral + error * dt,
            self.integral_min,
            self.integral_max,
        )

        derivative = (
            0.0
            if self.previous_error is None
            else (error - self.previous_error) / dt
        )

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.previous_error = error

        return clamp(
            output,
            self.output_min,
            self.output_max,
        )


# ============================================================
# Flask 서버에서 직접 사용하는 facade 클래스
# ============================================================

class TankDriveController:
    """
    D* Lite 경로 계획과 PID 주행 제어를 한 객체 안에 묶는 facade 클래스.

    서버 측에서는 내부 계산 함수나 상태변수를 알 필요가 없다.

    권장 서버 사용법:

        drive_controller = TankDriveController()

        @app.route('/info', methods=['POST'])
        def info():
            response, status = drive_controller.handle_info(request.get_json(force=True))
            return jsonify(response), status

        @app.route('/get_action', methods=['POST'])
        def get_action():
            return jsonify(drive_controller.get_action(request.get_json(force=True)))
    """

    # --------------------------------------------------------
    # 현재 데모 제어 파라미터
    # --------------------------------------------------------

    # 직선 기준 최고 목표속도 [km/h].
    MAX_SPEED_KMH = 60.0

    # 목적지/코너 제동거리 계산에서 가정하는 계획 감속도 [m/s^2].
    PLANNED_BRAKE_DECEL_MPS2 = 1.5

    # 목적지 중심에서 이 거리 이내면 도착 단계로 진입한다 [m].
    STOP_DISTANCE_M = 1.0

    # 도착 영역에서 이 속도 이하가 되면 이동 입력을 해제한다 [km/h].
    STOP_SPEED_KMH = 1.0

    # 속도 PID P gain.
    SPEED_KP = 0.12

    # 속도 PID D gain. 현재 I gain은 0이므로 실질적으로 PD 제어다.
    SPEED_KD = 0.02

    # 정지/직선 구간의 기본 look-ahead 거리 [m].
    LOOKAHEAD_BASE_M = 3.0

    # 현재 속도 [km/h]에 따라 look-ahead를 증가시키는 비례계수 [m/(km/h)].
    LOOKAHEAD_SPEED_GAIN = 0.08

    # 조향 PID P gain.
    STEER_KP = 0.025

    # 조향 PID D gain. 현재 I gain은 0이다.
    STEER_KD = 0.003

    # moveAD 조향 weight 최대값.
    STEER_MAX_WEIGHT = 0.85

    # /info 위치 기반 속도 fallback의 EMA 신규 측정값 비율.
    INFO_SPEED_EMA_ALPHA = 0.35

    def __init__(
        self,
        path_planner: DStarLitePlanner,
        map_image_path: str = "dstar_map.png",
    ) -> None:
        """
        주행 제어기 상태를 초기화한다.

        Args:
            path_planner:
                서버에서 생성한 DStarLitePlanner 또는 RiskDStarPlanner.
                경로계획 파라미터는 planner가 소유하며
                PID Controller에서 중복 정의하지 않는다.

            map_image_path:
                D* Lite 경로 시각화 PNG 저장 경로.
        """
        # 서버에서 생성한 planner 하나를 그대로 공유한다.
        # PID Controller 내부에서 planner 설정값을 다시 하드코딩하지 않는다.
        if path_planner is None:
            raise ValueError(
                "TankDriveController requires path_planner."
            )

        self.stop_flag = False

        self.planner = path_planner

        # Flask threaded 모드에서 obstacle/path 갱신이 겹치지 않도록 planner 접근을 보호한다.
        self.planner_lock = threading.RLock()

        # D* Lite 경로 시각화 파일 경로.
        self.map_image_path = str(map_image_path)

        # 현재 설정 목적지 [x, z] [m].
        self.dest: Optional[List[float]] = None

        # 가장 최근 차량 위치 [x, z] [m].
        self.current_pos: Optional[List[float]] = None

        # 현재 D* Lite가 반환한 압축 경로.
        self.current_path: List[PointXZ] = []

        # 가장 최근 /info JSON 전체.
        self.latest_info: Dict[str, Any] = {}

        # /info에서 계산/필터링한 현재 속도 [km/h].
        self.info_speed_kmh: Optional[float] = None

        # 위치 기반 속도 fallback 계산에 사용하는 직전 차량 위치 [x, z] [m].
        self.info_previous_position: Optional[List[float]] = None

        # 위치 기반 속도 fallback 계산에 사용하는 직전 /info 시간 [sec, monotonic].
        self.info_previous_time: Optional[float] = None

        # 사격팀 보정용 실제 차체 각속도 계산에 사용하는
        # 직전 /info 차체 yaw [deg].
        self.fire_previous_body_yaw_deg: Optional[float] = None

        # 사격팀 보정용 실제 차체 각속도 계산에 사용하는
        # 직전 /info 수신 시간 [sec, monotonic].
        self.fire_previous_body_yaw_time: Optional[float] = None

        # 연속된 /info의 playerBodyX 변화량으로 계산한
        # 현재 차체 yaw rate [deg/s].
        self.fire_body_rate_dps: float = 0.0

        # 직전 /get_action 제어 실행 시간 [sec, monotonic].
        self.last_control_time: Optional[float] = None

        # 목적지 반경에 한 번 진입한 뒤 다시 일반주행으로 복귀하지 않도록 유지하는 latch.
        self.arrival_latched = False

        # PID가 마지막으로 추종하던 목적지 signature.
        self.last_pid_destination: Optional[Tuple[float, float]] = None

        # 속도 제어용 PID. I gain은 현재 notebook과 동일하게 0.
        self.speed_pid = PIDController(
            self.SPEED_KP,
            0.0,
            self.SPEED_KD,
        )

        # 조향 제어용 PID. 출력은 실제 moveAD 허용 범위로 제한한다.
        self.steering_pid = PIDController(
            self.STEER_KP,
            0.0,
            self.STEER_KD,
            output_min=-self.STEER_MAX_WEIGHT,
            output_max=self.STEER_MAX_WEIGHT,
            integral_min=-30.0,
            integral_max=30.0,
        )

        # --------------------------------------------------------
        # 자체 인지(우리 쪽 파이프라인) 기반 회피/후퇴용 상태
        # --------------------------------------------------------
        # vehicle_mode:
        #   'advance' -> 평소 상태. get_action()이 매 tick grid 변화에 따라
        #                D* Lite 전진 재탐색을 돌린다.
        #   'retreat' -> 후퇴 중. 전진 재탐색을 멈추고 retreat 경로를
        #                self.current_path에 넣은 채 그대로 추종한다.
        #                목표(retreat 경로의 마지막 점)에 가까워지면
        #                자동으로 'advance'로 복귀한다.
        self.vehicle_mode = 'advance'

        # 실제로 지나온 world 좌표 breadcrumb. 후퇴 경로의 재료가 된다.
        # self.current_path는 항상 '미래로 갈 경로'만 들고 있어서 과거 이동
        # 기록이 따로 없기 때문에 이 리스트를 새로 둔다.
        self.position_history: List[PointXZ] = []
        self._history_min_step_m = 3.0    # 이 거리 이상 움직였을 때만 기록 (노이즈/중복 방지)
        self._history_max_points = 100    # 무한정 쌓이지 않게 상한
        # 한 번 후퇴할 때 되돌아갈 거리.
        # set_obstacles()의 obj_type별 padding 반경(예: enemy_tank는
        # grid 49칸)보다 짧으면 breadcrumb을 아무리 되짚어도 여전히
        # 패딩 구역 안이라 안전한 구간을 못 찾고 강제 재탐색으로 빠진다.
        # 실제 cell_size(격자 1칸당 미터 수)를 곱해서 padding 반경보다
        # 확실히 크게 잡아야 한다 — 이 값은 시작점일 뿐이니 실제 맵/
        # cell_size 기준으로 다시 튜닝해야 한다.
        self._retreat_distance_m = 60.0
        self._retreat_arrival_tolerance_m = 3.0  # 후퇴 목표점에 이만큼 가까워지면 후퇴 종료

    # --------------------------------------------------------
    # 자체 인지(우리 쪽 파이프라인) 기반 회피/후퇴
    # --------------------------------------------------------

    def _record_position_history(
        self,
        current_xz: Optional[Sequence[float]],
    ) -> None:
        """
        get_action()이 매 tick 호출한다.
        일정 거리 이상 움직였을 때만 breadcrumb을 남긴다.
        """
        if current_xz is None:
            return

        current_xz = (float(current_xz[0]), float(current_xz[1]))

        if not self.position_history:
            self.position_history.append(current_xz)
            return

        last = self.position_history[-1]

        if math.hypot(
            current_xz[0] - last[0],
            current_xz[1] - last[1],
        ) >= self._history_min_step_m:
            self.position_history.append(current_xz)

            if len(self.position_history) > self._history_max_points:
                self.position_history.pop(0)

    def _check_retreat_arrival(
        self,
        current_position: Optional[Sequence[float]],
    ) -> None:
        """
        후퇴 목표점(retreat 경로의 마지막 점)에 충분히 가까워졌는지 확인하고,
        가까워졌으면 advance 모드로 복귀시켜 다음 tick부터 다시 목적지를
        향해 전진 재탐색하게 한다.
        """
        if (
            self.vehicle_mode != 'retreat'
            or not self.current_path
            or current_position is None
        ):
            return

        retreat_target = self.current_path[-1]

        dist_to_retreat_target = math.hypot(
            retreat_target[0] - current_position[0],
            retreat_target[1] - current_position[1],
        )

        if dist_to_retreat_target <= self._retreat_arrival_tolerance_m:
            self.vehicle_mode = 'advance'

            # retreat 중 그대로 유지되던 D* Lite 재계획 추적 상태를 지워서
            # 다음 get_action() tick이 grid 변화 여부와 무관하게 무조건
            # 한 번 새로 전진 경로를 계산하도록 만든다.
            with self.planner_lock:
                if hasattr(self.planner, "reset_replan_tracking"):
                    self.planner.reset_replan_tracking()

                if self.dest is not None:
                    try:
                        self.current_path = self.planner.find_path(
                            current_position,
                            self.dest,
                            self.latest_info,
                        )
                    except ValueError as exc:
                        print(
                            "D* Lite 후퇴->전진 복귀 재계획 실패:",
                            exc,
                        )
                        self.current_path = []

                    if not self.current_path:
                        # find_path()가 예외 없이 그냥 빈 경로만 반환하는
                        # 경우(시작점/목적지 자체는 안 막혔는데 그 사이 경로가
                        # 없는 경우 -- 예: 방금 재분류된 오브젝트의 거대한
                        # 안전 반경이 두 지점 사이를 완전히 갈라놓은 경우).
                        # 이건 ValueError가 안 나서 위 except로도 안 걸리고,
                        # 그대로 두면 다음 tick에 또 find_path()를 불러도
                        # 똑같이 빈 경로만 나오는 게 무한 반복된다.
                        # clear_start_area로 점점 넓혀가며 재시도하는
                        # _find_path_with_recovery()로 한 번 더 시도한다.
                        print(
                            "D* Lite 후퇴->전진 복귀: find_path()가 빈 경로를 "
                            "반환함(시작/목적지 자체는 안 막혔지만 그 사이 경로가 "
                            "없는 상태) -> 비상 탈출 재시도"
                        )
                        self.current_path = (
                            self.planner._find_path_with_recovery(
                                current_position, self.dest,
                            )
                        )

            # 후퇴 경로 추종 중 쌓인 조향/속도 PID 오차가 새 전진 경로에
            # 그대로 이어지면 튀는 값이 나올 수 있어 초기화한다.
            self.speed_pid.reset()
            self.steering_pid.reset()

    def _build_retreat_path(
        self,
        retreat_distance_m: Optional[float] = None,
    ) -> Optional[List[PointXZ]]:
        """
        position_history를 거꾸로 따라가는 후퇴 경로를 만든다.
        이미 한 번 통과해서 안전이 검증된 구간이므로 D* Lite 재탐색 없이 바로
        쓸 수 있지만, 그 사이 또 다른 오브젝트가 잡혔을 수 있으니 구간마다
        다시 통행 가능 여부를 검사하며 안전한 만큼만 잘라서 반환한다.

        planner_lock을 이미 잡고 있는 컨텍스트에서만 호출한다.
        (self.planner_lock은 RLock이라 같은 스레드에서 재진입해도 안전하다.)
        """
        if retreat_distance_m is None:
            retreat_distance_m = self._retreat_distance_m

        if len(self.position_history) < 2:
            return None

        raw = [self.position_history[-1]]
        accumulated = 0.0

        for i in range(len(self.position_history) - 2, -1, -1):
            p_prev = self.position_history[i + 1]
            p_curr = self.position_history[i]

            accumulated += math.hypot(
                p_curr[0] - p_prev[0],
                p_curr[1] - p_prev[1],
            )

            raw.append(p_curr)

            if accumulated >= retreat_distance_m:
                break

        # raw[0]은 현재 위치(지금 막혀 있는 지점) 그 자체다. 방금 생긴
        # 패딩 반경이 breadcrumb 간격보다 넓으면(예: enemy_tank 안전
        # 반경 49칸처럼 큰 경우), 바로 다음 breadcrumb 몇 개도 같이
        # 덮여 있을 수 있다. _is_straight_line_walkable()은 시작점
        # 바로 다음 지점부터 촘촘히 검사하는 함수라 그대로 쓰면 항상
        # 실패하니, 실제로 다시 통행 가능한 첫 breadcrumb을 찾을 때까지
        # 건너뛴 다음 거기서부터 정상적으로 구간별 검사를 이어간다.
        first_free_idx = None

        for idx in range(1, len(raw)):
            grid = self.planner.world_to_grid(raw[idx], clamp=True)
            if self.planner.is_free(grid):
                first_free_idx = idx
                break

        if first_free_idx is None:
            # breadcrumb 전체가 여전히 막혀 있다 -> 이 기록으로는 후퇴가
            # 불가능하다 (retreat_distance_m을 늘리거나 상위 호출부의
            # 비상 탈출(_find_path_with_recovery)에 맡겨야 한다).
            return None

        safe = [raw[0], raw[first_free_idx]]

        for p1, p2 in zip(raw[first_free_idx:], raw[first_free_idx + 1:]):
            if not self.planner._is_straight_line_walkable(p1, p2):
                break

            safe.append(p2)

        return safe if len(safe) >= 2 else None

    # 인지 파이프라인이 보내는 class_name -> (half_width_m, half_length_m).
    # 실측 하단 몸체 크기(x, y, z) 기준 절반값. y(높이)는 2D 평면(x-z)에서
    # 도는 D* Lite 충돌판정엔 안 쓴다.
    #   적 전차 몸체 (3.303, 1.131, 6.339) -> half_width=1.6515, half_length=3.1695
    # Tank2(아군 전차)는 정확한 실측치가 따로 없어 우선 동일 차체로 가정한다.
    # 실측치가 확인되면 여기 값만 바꾸면 된다.
    _OBJECT_HALF_EXTENTS_M: Dict[str, Tuple[float, float]] = {
        'Tank1': (1.6515, 3.1695),
        'Tank2': (1.6515, 3.1695),
    }

    # update_obstacles_type()에서 매칭 실패해 pad_object()로 새로 등록할 때
    # 쓸 obj_type. class_name -> obj_type 매핑.
    _OBJECT_TYPE_FOR_PAD: Dict[str, str] = {
        'Tank1': 'enemy_tank',
        'Tank2': 'team_tank',
    }

    def handle_objects_detected(
        self,
        detections,
    ) -> Dict[str, Any]:
        """
        인지 파이프라인이 한 프레임에 탐지한 오브젝트 여러 개를 한 번에 처리하는
        진입점.

            drive_controller.handle_objects_detected([
                (102.0, 5.0, 103.0, "Tank1"),
                (200.0, 5.0, 200.0, "Tank1"),
            ])

        처리 순서(백그라운드 스레드):
            1) planner.update_obstacles_type()으로 Unity /update_obstacle가
               이미 등록해둔 고정 오브젝트(예: 고정 배치된 Tank1 모형)와
               좌표가 겹치는지 먼저 확인 -> 겹치면 타입만 재분류(좌표는
               이미 정확하니 새로 만들 필요 없음)
            2) 겹치는 기존 오브젝트가 없는 탐지(예: 실시간으로 움직이는
               적 전차처럼 Unity가 애초에 등록 안 해주는 동적 오브젝트)는
               pad_object() 기반의 기존 흐름(_process_object_detected)으로
               새 장애물을 등록

        Returns
        -------
        dict: {"status": "queued"} 고정. 실제 처리는 비동기라 이 반환값으론
              결과를 알 수 없고, 필요하면 서버 콘솔 로그로 확인한다.
        """
        thread = threading.Thread(
            target=self._process_objects_detected,
            args=(list(detections),),
            daemon=True,
        )
        thread.start()
        return {"status": "queued"}

    def _process_objects_detected(self, detections) -> None:
        """
        handle_objects_detected()가 백그라운드 스레드에서 실행하는 실제 로직.
        직접 호출하지 말고 handle_objects_detected()를 통해서만 사용한다.

        설계 변경(중요):
            예전엔 매칭 안 된(unmatched) 탐지를 pad_object()로 새 장애물을
            만들었는데, 이걸 없앴다. 이유:

            1) 실시간으로 움직이는 적 전차 1대는 이미 latest_info["enemyPos"]
               기반 movable_enemy_tank(find_path() 안에서 매번 갱신)로 별도
               처리되고 있어서, 우리 비전 탐지가 새로 장애물을 만들어줄
               필요가 원래 없다.
            2) 우리가 매핑해야 하는 진짜 대상은 "이미 /update_obstacle로
               등록된 고정 장애물의 타입 확정"뿐이다. 즉 매칭이 되어야
               정상이고, 매칭이 안 됐다는 건 새 오브젝트를 찾은 게
               아니라 대부분 신뢰할 수 없는 탐지(먼 거리에서 스테레오
               삼각측량 오차가 커진 경우 등)라는 신호에 가깝다.
            3) 실측(디버그 로그)으로 확인됨: 거리 100m 이상에서
               baseline 1m 스테레오는 disparity 1~2px 차이만으로도
               depth가 100m 넘게 요동친다. 이런 신뢰 못 할 좌표를 매번
               pad_object로 새로 등록하면, 매 프레임 다른 위치에
               유령 장애물이 계속 쌓여서 실제 경로 탐색을 방해했다.

            그래서 이제 unmatched는 그냥 로그만 남기고 아무 것도
            등록하지 않는다. (나중에 "Unity가 아직 등록 안 해준 진짜
            새 오브젝트"를 다뤄야 하는 상황이 생기면 이 부분을 다시
            설계해야 한다.)
        """
        try:
            with self.planner_lock:
                changed_cells, unmatched = self.planner.update_obstacles_type(
                    detections,
                )

            if changed_cells:
                self.render_map("D* Lite Map (오브젝트 타입 갱신)")

            if unmatched:
                print(
                    f"[_process_objects_detected] 매칭 안 된 탐지 {len(unmatched)}건 "
                    f"무시함(신뢰도 낮은 좌표로 간주): {unmatched}"
                )

        except Exception as exc:
            # 백그라운드 스레드라 예외가 호출자에게 안 올라간다. 콘솔에
            # 남겨서 조용히 묻히지 않게 한다.
            print(f"[_process_objects_detected] 처리 실패: {exc}")

    def handle_object_detected(
        self,
        x_min: float,
        x_max: float,
        z_min: float,
        z_max: float,
        obj_type: str = 'enemy_tank',
    ) -> Dict[str, Any]:
        """
        우리 쪽 인지 파이프라인(팀원 작업, world 좌표 확보 후)이 직접 호출하는
        진입점. Flask 라우트가 아니라 일반 메서드라서:

            drive_controller.handle_object_detected(40, 46, 100, 106, obj_type='enemy_tank')

        실제 처리(_process_object_detected)는 별도 데몬 스레드에서 돈다.
        pad_object() -> set_obstacles()가 고도 데이터 재주입(90000칸 순회) +
        clearance cost 재계산(Dijkstra)까지 하느라 실측 1~2초가 걸리는데,
        이 함수가 planner_lock을 잡은 채로 그 시간만큼 호출자를 막으면
        (특히 인식 파이프라인이 매 프레임 이걸 호출하는 상황이면) /get_action이
        같은 락을 기다리다 실시간 제어 루프에 지연이 생길 수 있다. 그래서
        이 메서드 자체는 스레드만 띄우고 즉시 리턴하고, 실제 상태 변경
        (self.current_path/self.vehicle_mode 교체 등)은 그 스레드 안에서
        여전히 planner_lock을 잡고 안전하게 수행한다.

        Returns
        -------
        dict: {"status": "queued"} 고정. 처리 결과 자체는 비동기라 이
              반환값으로는 알 수 없고, 필요하면 서버 콘솔 로그로 확인한다.
        """
        thread = threading.Thread(
            target=self._process_object_detected,
            args=(x_min, x_max, z_min, z_max, obj_type),
            daemon=True,
        )
        thread.start()
        return {"status": "queued"}

    def _process_object_detected(
        self,
        x_min: float,
        x_max: float,
        z_min: float,
        z_max: float,
        obj_type: str = 'enemy_tank',
    ) -> Dict[str, Any]:
        """
        handle_object_detected()가 백그라운드 스레드에서 실행하는 실제 로직.
        직접 호출하지 말고 handle_object_detected()를 통해서만 사용한다.

        처리 순서:
            1) 오브젝트 패딩 처리 (planner.pad_object)
            2) 그로 인해 지금 self.current_path가 막혔는지 판단 (planner.is_path_blocked)
            3-a) 안 막혔으면 -> 아무것도 안 하고 그대로 진행
            3-b) 막혔으면 -> 현재 위치가 그 패딩(통행 불가 셀) 위에 서 있는지 판단
                 - 서 있음   -> 후퇴 모드로 전환, retreat 경로로 self.current_path 교체
                 - 안 서 있음 -> 전진 방향으로 강제 재탐색해서 self.current_path 교체,
                                 advance 모드 유지

        Returns
        -------
        dict: 상태 설명용. 백그라운드 스레드에서 도니 호출부로 리턴되지
              않고, 예외 발생 시 콘솔에 출력한다.
        """
        try:
            # 패딩 자체는 목적지/현재 위치와 무관하게 항상 먼저 반영한다.
            # (예: restart 직후, 아직 목적지를 안 정한 상태에서 화면에 적 전차가
            # 잡혀도 맵에는 바로 반영되어야 한다.) 그 아래 "지금 경로가 막혔는지
            # -> 후퇴/재탐색" 판단만 목적지/현재 위치가 있어야 의미가 있는
            # 부분이라 그 시점에 가서 갈린다.
            with self.planner_lock:
                changed_cells = self.planner.pad_object(
                    x_min, x_max, z_min, z_max, obj_type,
                )

            if changed_cells:
                # 실제로 뭔가 바뀌었을 때만 다시 그린다. 탐지는 매 프레임(예:
                # /stereo_image가 계속 들어오는 상황) 반복 호출될 수 있는데,
                # 매번 아무 변화 없어도 렌더 스레드를 새로 띄우면 낭비다.
                self.render_map("D* Lite Map (오브젝트 탐지 갱신)")

            current_position = self.current_pos
            destination_xz = self.dest

            if current_position is None or destination_xz is None:
                return {
                    "status": "padded_only",
                    "reason": "position/destination not ready",
                    "changed_cells": len(changed_cells or []),
                }

            with self.planner_lock:
                # current_path가 이미 비어있는 상태(예: 직전 재탐색 실패로 멈춰있는
                # 상황)도 '막힘'으로 간주해야 한다 — 그렇지 않으면 is_path_blocked()가
                # 빈 리스트에 대해 False를 반환해서 멈춰있는 차량을 그대로 방치한다.
                blocked = (
                    not self.current_path
                    or self.planner.is_path_blocked(self.current_path)
                )

                if not blocked:
                    return {
                        "status": "clear",
                        "path_blocked": False,
                        "changed_cells": len(changed_cells or []),
                    }

                current_grid = self.planner.world_to_grid(
                    current_position, clamp=True,
                )
                standing_on_blocked_cell = not self.planner.is_free(current_grid)

                if standing_on_blocked_cell:
                    retreat_path = self._build_retreat_path()

                    if retreat_path is None:
                        # 후퇴할 기록이 없다(예: 시작하자마자 막힘) -> 최후 수단으로
                        # 그 자리에서 강제 재탐색을 시도한다.
                        #
                        # 이 분기에 들어왔다는 것 자체가 "지금 서 있는 셀이 막혀
                        # 있다"는 뜻이라 find_path()를 그대로 부르면 시작점 검증에서
                        # 무조건 ValueError가 난다. 그래서 clear_start_area()로
                        # 반경을 넓혀가며 실제로 뚫어주는 _find_path_with_recovery()를
                        # 써야 한다. (단, latest_info 기반 적 전차 마스킹은 이 경로에서
                        # 지원되지 않는다 — _find_path_with_recovery가 내부적으로
                        # find_path(pos, dest)를 latest_info 없이 호출하기 때문.)
                        new_path = self.planner._find_path_with_recovery(
                            current_position, destination_xz,
                        )

                        self.current_path = new_path or []
                        self.vehicle_mode = 'advance'
                        self.speed_pid.reset()
                        self.steering_pid.reset()

                        return {
                            "status": "forced_replan_no_history",
                            "path_blocked": True,
                            "changed_cells": len(changed_cells or []),
                        }

                    self.current_path = retreat_path
                    self.vehicle_mode = 'retreat'
                    self.speed_pid.reset()
                    self.steering_pid.reset()

                    return {
                        "status": "retreating",
                        "path_blocked": True,
                        "changed_cells": len(changed_cells or []),
                        "retreat_points": len(retreat_path),
                    }

                else:
                    try:
                        new_path = self.planner.find_path(
                            current_position, destination_xz, self.latest_info,
                        )
                    except ValueError as exc:
                        print("D* Lite 강제 재탐색 실패:", exc)
                        new_path = []

                    self.current_path = new_path or []
                    self.vehicle_mode = 'advance'

                    return {
                        "status": "replanned",
                        "path_blocked": True,
                        "changed_cells": len(changed_cells or []),
                    }

        except Exception as exc:
            # 백그라운드 스레드라 예외가 호출자에게 안 올라간다. 콘솔에
            # 남겨서 조용히 묻히지 않게 한다.
            print(f"[_process_object_detected] 처리 실패: {exc}")
            return {"status": "error", "message": str(exc)}

    # --------------------------------------------------------
    # 내부 상태 관리
    # --------------------------------------------------------

    def _reset_control_state(
        self,
        reset_destination_signature: bool = True,
    ) -> None:
        """
        PID와 시간/도착 상태를 초기화한다.

        Args:
            reset_destination_signature:
                True이면 마지막 목적지 signature도 제거한다.
        """
        self.speed_pid.reset()
        self.steering_pid.reset()

        self.last_control_time = None
        self.arrival_latched = False

        if reset_destination_signature:
            self.last_pid_destination = None

    def _reset_info_state(self) -> None:
        """
        /info 기반 속도 필터 상태를 초기화한다.
        """
        self.info_speed_kmh = None
        self.info_previous_position = None
        self.info_previous_time = None

        # 이전 episode의 yaw 변화량이 새 episode에 섞이지 않도록 초기화한다.
        self.fire_previous_body_yaw_deg = None
        self.fire_previous_body_yaw_time = None
        self.fire_body_rate_dps = 0.0

        self.latest_info = {}

    # --------------------------------------------------------
    # /info 처리
    # --------------------------------------------------------

    def _update_info_speed(
        self,
        data: Dict[str, Any],
        player_position: Sequence[float],
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        /info에서 현재 속도를 갱신한다.

        우선순위:
            1. playerSpeed 등 명시적 속도 필드.
            2. 없으면 연속 위치 변화량 / dt.

        Args:
            data:
                /info JSON.
            player_position:
                현재 차량 [x, z] 위치 [m].

        Returns:
            (필터링된 속도 [km/h], 측정 source 문자열)
        """
        now = time.monotonic()

        explicit_speed_kmh, speed_key = extract_speed_from_info(data)

        measured_speed_kmh = explicit_speed_kmh
        speed_source = speed_key

        if (
            measured_speed_kmh is None
            and self.info_previous_position is not None
            and self.info_previous_time is not None
        ):
            dt = now - self.info_previous_time

            if 0.01 <= dt <= 1.0:
                dx = (
                    float(player_position[0])
                    - float(self.info_previous_position[0])
                )

                dz = (
                    float(player_position[1])
                    - float(self.info_previous_position[1])
                )

                measured_speed_kmh = (
                    math.hypot(dx, dz)
                    / dt
                    * 3.6
                )

                speed_source = "playerPos/dt"

        self.info_previous_position = [
            float(player_position[0]),
            float(player_position[1]),
        ]

        self.info_previous_time = now

        if measured_speed_kmh is not None:
            if self.info_speed_kmh is None:
                self.info_speed_kmh = measured_speed_kmh

            else:
                self.info_speed_kmh = (
                    self.INFO_SPEED_EMA_ALPHA
                    * measured_speed_kmh
                    + (
                        1.0
                        - self.INFO_SPEED_EMA_ALPHA
                    )
                    * self.info_speed_kmh
                )

        return self.info_speed_kmh, speed_source

    def _read_player_body_yaw_deg(self) -> Optional[float]:
        """
        최신 /info의 playerBodyX를 차체 yaw [deg]로 읽는다.

        Returns:
            0 ~ 360 deg yaw.
            값이 없거나 숫자로 변환할 수 없으면 None.
        """
        value = self.latest_info.get("playerBodyX")

        if value is None:
            value = self.latest_info.get("PlayerBodyX")

        if value is None:
            return None

        try:
            return float(value) % 360.0
        except (TypeError, ValueError):
            return None

    def _update_fire_body_rate(
        self,
        body_yaw_deg: Optional[float],
    ) -> float:
        """
        연속된 /info의 차체 yaw 변화량으로 실제 차체 각속도를 계산한다.

        Args:
            body_yaw_deg:
                현재 /info의 playerBodyX [deg].

        Returns:
            현재 차체 yaw rate [deg/s].

        계산식:
            body_rate_dps = normalize_angle(current_yaw - previous_yaw) / dt

        역할:
            PID 조향 weight에 임의의 선회율 상수를 곱하지 않고,
            시뮬레이터에서 실제로 변한 playerBodyX를 사용해
            사격팀의 차체 회전 보정값을 만든다.
        """
        # 현재 /info 수신 시각 [sec, monotonic].
        now = time.monotonic()

        # yaw 데이터가 없으면 실제 각속도를 계산할 수 없으므로 0으로 둔다.
        if body_yaw_deg is None:
            self.fire_previous_body_yaw_deg = None
            self.fire_previous_body_yaw_time = None
            self.fire_body_rate_dps = 0.0
            return self.fire_body_rate_dps

        # 첫 번째 yaw 샘플은 이전 값이 없으므로 기준값만 저장한다.
        if (
            self.fire_previous_body_yaw_deg is None
            or self.fire_previous_body_yaw_time is None
        ):
            self.fire_previous_body_yaw_deg = float(body_yaw_deg)
            self.fire_previous_body_yaw_time = now
            self.fire_body_rate_dps = 0.0
            return self.fire_body_rate_dps

        # 직전 /info와 현재 /info 사이의 실제 시간 차 [sec].
        dt = (
            now
            - self.fire_previous_body_yaw_time
        )

        # 359 -> 0 deg처럼 360 deg 경계를 넘어도
        # 실제 최단 회전량을 얻도록 -180~+180 deg로 정규화한다.
        yaw_delta_deg = normalize_angle_deg(
            float(body_yaw_deg)
            - self.fire_previous_body_yaw_deg
        )

        # 정상적인 /info 시간 간격에서만 각속도를 계산한다.
        if 0.01 <= dt <= 1.0:
            self.fire_body_rate_dps = (
                yaw_delta_deg
                / dt
            )
        else:
            self.fire_body_rate_dps = 0.0

        # 다음 /info 계산을 위한 현재 yaw/시간 저장.
        self.fire_previous_body_yaw_deg = float(body_yaw_deg)
        self.fire_previous_body_yaw_time = now

        return self.fire_body_rate_dps

    def get_fire_control_inputs(
        self,
        drive_command: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        FireModule.get_turret_command()에 필요한 주행팀 입력을 반환한다.

        기존 PID 주행 제어식과 Jupyter Notebook의 비례식은 변경하지 않는다.
        PID Controller가 이미 /info에서 관리하는 실제 속도와 yaw,
        실제 yaw 변화량으로 계산한 차체 각속도만 사격팀에 제공한다.

        Args:
            drive_command:
                이번 /get_action에서 계산한 주행 명령 dictionary.
                현재 계산에서는 별도 선회율 상수나 moveAD 비례식을 만들지 않는다.

        Returns:
            my_vel:
                현재 차량 월드 좌표계 속도 벡터 [m/s].

            body_rate_dps:
                실제 playerBodyX 변화량 / dt로 계산한 차체 각속도 [deg/s].

            hull_settled:
                현재 차량이 거의 정지하고 차체 회전도 거의 없는지 여부.
        """
        # PID 속도 제어에 실제로 사용 중인 필터링 속력 [km/h].
        speed_kmh = (
            float(self.info_speed_kmh)
            if self.info_speed_kmh is not None
            else 0.0
        )

        # FireModule 입력 단위인 m/s로 변환한 현재 속력.
        speed_mps = (
            speed_kmh
            / 3.6
        )

        # PID 조향 제어에 실제로 사용 중인 차체 yaw [deg].
        body_yaw_deg = (
            self._read_player_body_yaw_deg()
        )

        # 아직 yaw가 들어오지 않았으면 속도벡터 방향 계산에서만 0 deg를 사용한다.
        velocity_yaw_deg = (
            float(body_yaw_deg)
            if body_yaw_deg is not None
            else 0.0
        )

        # 속도 벡터 계산을 위한 yaw [rad].
        velocity_yaw_rad = math.radians(
            velocity_yaw_deg
        )

        # Unity 기준 yaw 0 deg = +Z이므로 X축 속도는 sin(yaw)를 사용한다.
        velocity_x_mps = (
            speed_mps
            * math.sin(velocity_yaw_rad)
        )

        # Unity 기준 yaw 0 deg = +Z이므로 Z축 속도는 cos(yaw)를 사용한다.
        velocity_z_mps = (
            speed_mps
            * math.cos(velocity_yaw_rad)
        )

        # FireModule에 넘길 자기 차량의 월드 속도 벡터 [m/s].
        my_vel = (
            velocity_x_mps,
            0.0,
            velocity_z_mps,
        )

        # /info의 실제 yaw 변화로 계산해 둔 차체 각속도 [deg/s].
        body_rate_dps = float(
            self.fire_body_rate_dps
        )

        # 기존 FireModule 연동에서 사용하던 정지 판정 기준을 유지한다.
        hull_settled = (
            speed_mps < 0.3
            and abs(body_rate_dps) < 1e-6
        )

        return {
            "my_vel": my_vel,
            "body_rate_dps": body_rate_dps,
            "hull_settled": hull_settled,
        }


    def handle_info(
        self,
        data: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        """
        Flask /info endpoint의 전체 주행 상태 처리를 담당한다.

        Args:
            data:
                request.get_json(force=True) 결과.

        Returns:
            (response dictionary, HTTP status code)
        """
        data = data or {}

        if not data:
            return {"error": "No JSON received"}, 400

        self.latest_info = data

        # 이번 /info에서 받은 실제 차체 yaw [deg].
        body_yaw_deg = (
            self._read_player_body_yaw_deg()
        )

        # 연속 /info의 yaw 변화량으로 사격팀 보정용
        # 실제 차체 각속도 [deg/s]를 갱신한다.
        self._update_fire_body_rate(
            body_yaw_deg
        )

        player_pos = data.get("playerPos", {})

        if player_pos:
            fallback_x = (
                self.current_pos[0]
                if self.current_pos is not None
                else 0.0
            )

            fallback_z = (
                self.current_pos[1]
                if self.current_pos is not None
                else 0.0
            )

            self.current_pos = [
                float(player_pos.get("x", fallback_x)),
                float(player_pos.get("z", fallback_z)),
            ]

        # playerPos가 없는 경우에는 기존 current_pos가 있을 때만 속도 상태를 갱신한다.
        if self.current_pos is not None:
            speed_kmh, speed_source = self._update_info_speed(
                data,
                self.current_pos,
            )
        else:
            speed_kmh = self.info_speed_kmh
            speed_source = None

        print("[/info] 현재 위치:", self.current_pos)
        print("[/info] 차체 방향:", data.get("playerBodyX"))
        print(
            "[/info] 현재 속도:",
            "계산 대기"
            if speed_kmh is None
            else f"{speed_kmh:.2f} km/h",
            f"(source={speed_source})",
        )
        print("[/info] 설정 목적지:", self.dest)

        return {
            "status": "success",
            "control": "",
        }, 200

    # --------------------------------------------------------
    # 목적지 / 초기화
    # --------------------------------------------------------

    def initialize(
        self,
        start_position: Sequence[float],
        destination: Optional[Sequence[float]] = None,
    ) -> None:
        """
        새 episode 시작 시 제어 상태를 초기화한다.

        Args:
            start_position:
                시작 X-Z 좌표 [m].
            destination:
                선택적 초기 목적지.
                (x, z) 또는 (x, y, z)를 받을 수 있다.
        """
        if len(start_position) < 2:
            raise ValueError(
                "start_position must contain x and z"
            )

        self.current_pos = [
            float(start_position[0]),
            float(start_position[-1]),
        ]

        self.current_path = []

        # 새 episode에서는 이전 episode의 breadcrumb/후퇴 상태가
        # 섞이지 않도록 항상 advance로 초기화한다.
        self.vehicle_mode = 'advance'
        self.position_history = []

        self._reset_control_state(
            reset_destination_signature=True,
        )

        self._reset_info_state()

        # _reset_info_state가 current_pos를 지우지 않으므로 시작 위치는 그대로 유지된다.
        if destination is not None:
            if len(destination) >= 3:
                x = float(destination[0])
                y = float(destination[1])
                z = float(destination[2])
            elif len(destination) == 2:
                x = float(destination[0])
                y = 0.0
                z = float(destination[1])
            else:
                raise ValueError(
                    "destination must contain (x,z) or (x,y,z)"
                )

            self.apply_destination(
                x,
                y,
                z,
            )

    def apply_destination(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Dict[str, Any]:
        """
        /set_destination과 /init이 공통으로 사용하는 목적지 설정 함수.

        Args:
            x:
                목적지 X 좌표 [m].
            y:
                목적지 Y 좌표 [m]. D* Lite 평면 경로에는 사용하지 않지만 응답 형식을 유지한다.
            z:
                목적지 Z 좌표 [m].

        Returns:
            목적지, 경로 길이/비용, map URL 정보를 포함하는 response dictionary.
        """
        if self.current_pos is None:
            raise ValueError(
                "Current position is not received yet"
            )

        self.dest = [
            float(x),
            float(z),
        ]

        # 새 목적지가 들어오면 후퇴 중이었더라도 전진 상태로 복귀한다.
        self.vehicle_mode = 'advance'

        self._reset_control_state(
            reset_destination_signature=True,
        )

        print(
            "[DEST RESET]",
            f"dest={self.dest}",
            f"arrival_latched={self.arrival_latched}",
            f"last_pid_destination={self.last_pid_destination}",
        )

        self.clear_start_area(
            self.current_pos,
            radius=2,
        )

        with self.planner_lock:
            self.current_path = self.planner.find_path(
                self.current_pos,
                self.dest,
                self.latest_info
            )

        self.render_map(
            "D* Lite Demo (300X300)"
        )

        print(
            f"🎯 Destination set to: "
            f"x={x}, y={y}, z={z}"
        )

        return {
            "status": "OK",
            "destination": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            },
            "path_done_count": len(self.current_path),
            "path_cost": self.planner.get_path_cost(
                self.current_path
            ),
            "map_url": "/path_map",
        }

    def handle_set_destination(
        self,
        data: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        """
        Flask /set_destination endpoint의 입력검사와 목적지 적용을 담당한다.

        Args:
            data:
                request.get_json() 결과.

        Returns:
            (response dictionary, HTTP status code)
        """
        self.stop_flag = True
        self.current_path = []

        data = data or {}

        if "destination" not in data:
            return {
                "status": "ERROR",
                "message": "Missing destination data",
            }, 400

        try:
            x, y, z = map(
                float,
                data["destination"].split(","),
            )
        except Exception as exc:
            # 좌표 파싱 자체가 실패한 경우(콤마 개수, 숫자 변환 등)만
            # "Invalid format"으로 분류한다.
            return {
                "status": "ERROR",
                "message": f"Invalid format: {str(exc)}",
            }, 400

        try:
            return self.apply_destination(
                x,
                y,
                z,
            ), 200

        except ValueError as exc:
            # apply_destination() -> find_path()가 던지는 ValueError는
            # 좌표 형식 문제가 아니라 "시작점/목적지가 장애물(패딩 포함)에
            # 막혀서 경로를 못 만든다"는 뜻이다(혹은 아직 /info를 못 받아
            # current_pos가 없는 경우). 원인을 구분해서 응답하고 서버
            # 콘솔에도 그대로 남겨서 디버깅이 가능하게 한다.
            print(f"[/set_destination] 경로 계산 실패: {exc}")
            return {
                "status": "ERROR",
                "message": f"Path planning failed: {str(exc)}",
            }, 400

        except Exception as exc:
            # 그 외 예상 못한 예외는 원인을 감추지 않고 traceback까지 남긴다.
            import traceback
            traceback.print_exc()
            return {
                "status": "ERROR",
                "message": f"Unexpected error: {str(exc)}",
            }, 500

    # --------------------------------------------------------
    # D* Lite obstacle / map 관리
    # --------------------------------------------------------

    def clear_start_area(
        self,
        position: Optional[Sequence[float]],
        radius: int = 2,
    ) -> None:
        """
        시작점 주변 hard obstacle을 예외적으로 해제한다.

        현재 복구된 서버와 동일하게 기본 radius=2를 유지한다.

        Args:
            position:
                현재 차량 X-Z 위치 [m].
            radius:
                시작점 주변에서 obstacle을 해제할 grid 반경 [cell].
        """
        if position is None:
            return

        with self.planner_lock:
            center = self.planner.world_to_grid(
                position,
                clamp=True,
            )

            changed = set()

            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    cell = (
                        center[0] + dx,
                        center[1] + dz,
                    )

                    if (
                        self.planner.in_bounds(cell)
                        and cell in self.planner.obstacles
                    ):
                        self.planner.obstacles.discard(cell)
                        changed.add(cell)

            if changed:
                self.planner.refresh_costmap()

    def _update_obstacles_from_payload(
        self,
        payload: Dict[str, Any],
    ):
        """
        /update_obstacle payload를 ObstacleRect 목록으로 변환해 planner에 반영한다.

        Args:
            payload:
                obstacles 배열을 포함하는 JSON dictionary.

        Returns:
            DStarLitePlanner.set_obstacles()가 반환한 changed_cells.
        """
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

        with self.planner_lock:
            changed_cells = self.planner.set_obstacles(
                obs_list
            )

        self.clear_start_area(
            self.current_pos,
            radius=2,
        )

        print(
            "변경된 장애물 Grid 수:",
            len(changed_cells),
        )
        print(
            "등록된 장애물 사각형 수:",
            len(self.planner.obstacle_rectangles),
        )
        print(
            "Clearance cost 설정:",
            f"hard_margin={self.planner.obstacle_margin},",
            f"radius={self.planner.clearance_radius},",
            f"weight={self.planner.clearance_weight},",
            f"decay={self.planner.clearance_decay}",
        )
        print(
            "Soft cost 적용 셀 수:",
            len(self.planner.clearance_costs),
        )

        return changed_cells

    def handle_update_obstacles(
        self,
        data: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        """
        Flask /update_obstacle endpoint의 obstacle 갱신과 재계획을 담당한다.

        Args:
            data:
                request.get_json() 결과.

        Returns:
            (response dictionary, HTTP status code)
        """
        if not data:
            return {
                "status": "error",
                "message": "No data received",
            }, 400

        print(
            "🪨 Obstacle Data:",
            data,
        )

        changed_cells = self._update_obstacles_from_payload(
            data
        )

        if (
            self.current_pos is not None
            and self.dest is not None
        ):
            # Unity 쪽 /update_obstacle로 전체 장애물 목록이 새로 온 경우이므로
            # 자체 인지 기반 후퇴 중이었더라도 전진 상태로 복귀해 새 맵 기준으로
            # 다시 계획한다.
            self.vehicle_mode = 'advance'

            try:
                with self.planner_lock:
                    self.current_path = self.planner.find_path(
                        self.current_pos,
                        self.dest,
                        self.latest_info
                    )

            except ValueError as exc:
                print(
                    "D* Lite 재계획 실패:",
                    exc,
                )
                self.current_path = []

            self.render_map(
                "D* Lite Replanning"
            )

        else:
            self.current_path = []

            self.render_map(
                "D* Lite Obstacle Map"
            )

        return {
            "status": "success",
            "message": "Obstacle data received",
            "changed_cell_count": len(changed_cells),
            "path_length": len(self.current_path),
            "obstacle_count": len(
                self.planner.obstacle_rectangles
            ),
            "map_url": "/path_map",
        }, 200

    def render_map(
        self,
        title: str,
    ) -> str:
        """
        현재 D* Lite obstacle/path 상태를 PNG로 저장한다.

        주의:
            planner.plot()(동기/blocking)이 아니라 plot_async()를 쓴다.
            plot()은 matplotlib legend(loc='best')가 obstacle_rectangles
            수백 개 + 90000픽셀 imshow를 상대로 '겹치지 않는 위치'를
            전수 탐색하느라 실제로 수십 초~분 단위로 걸릴 수 있는데,
            이 함수가 apply_destination()/handle_update_obstacles() 안에서
            request 처리 스레드를 그대로 막고 있어서(게다가
            planner_lock까지 잡은 채로) 목적지 설정/장애물 갱신 응답
            자체가 오래 걸리는 원인이었다. plot_async()는 그리는 데
            필요한 데이터만 스냅샷 떠서 별도 데몬 스레드에 넘기고
            즉시 리턴하므로 요청 스레드를 막지 않는다.

        Args:
            title:
                plot 제목.

        Returns:
            저장될 예정인 map 이미지 파일 경로. plot_async()는 비동기라
            이 시점에는 아직 파일이 안 만들어져 있을 수 있다.
        """
        with self.planner_lock:
            self.planner.plot_async(
                path=(
                    self.current_path
                    if self.current_path
                    else None
                ),
                show_grid=True,
                title=title,
                save_path=self.map_image_path,
            )

        return self.map_image_path

    def get_map_path(self) -> str:
        """
        /path_map endpoint에서 send_file에 넘길 map 이미지 경로를 반환한다.

        render_map()은 이제 비동기라 파일이 아직 안 만들어졌을 수 있다.
        여긴 사람이 브라우저로 이미지를 열어보는 디버그용 endpoint라
        핫패스(주행 루프)와 달리 한 번쯤 느려도 괜찮으므로, 파일이 아예
        없는 최초 1회에 한해 동기 plot()으로 확실히 만들어서 돌려준다.

        Returns:
            map PNG 경로.
        """
        if not Path(
            self.map_image_path
        ).exists():
            with self.planner_lock:
                self.planner.plot(
                    path=(
                        self.current_path
                        if self.current_path
                        else None
                    ),
                    show_grid=True,
                    title="D* Lite Map",
                    save_path=self.map_image_path,
                    show=False,
                )

        return self.map_image_path

    # --------------------------------------------------------
    # 속도 / 코너 / 조향 계산
    # --------------------------------------------------------

    def _calculate_target_speed_kmh(
        self,
        usable_distance: float,
    ) -> float:
        """
        남은 거리에서 계획 감속도로 정지 가능한 속도 상한을 계산한다.

        v^2 = 2ad 를 사용한다.

        Args:
            usable_distance:
                반응거리를 제외한 실제 감속 가능 거리 [m].

        Returns:
            목적지 접근 목표속도 상한 [km/h].
        """
        usable_distance = max(
            0.0,
            float(usable_distance),
        )

        braking_speed_mps = math.sqrt(
            2.0
            * self.PLANNED_BRAKE_DECEL_MPS2
            * usable_distance
        )

        max_speed_mps = (
            self.MAX_SPEED_KMH / 3.6
        )

        target_speed_mps = min(
            max_speed_mps,
            braking_speed_mps,
        )

        return target_speed_mps * 3.6

    def _calculate_corner_speed_limit(
        self,
        corner: Optional[Dict[str, Any]],
        current_speed_kmh: float,
        dt: float,
        normal_speed_kmh: Optional[float] = None,
    ) -> float:
        """
        코너 각도와 남은 거리로 제한속도를 연속적으로 계산한다.

        Args:
            corner:
                calculate_next_vertex_corner() 결과.
            current_speed_kmh:
                현재 차량 속도 [km/h].
            dt:
                현재 제어 주기 [sec].
            normal_speed_kmh:
                직선 기준 목표속도 [km/h].
                None이면 MAX_SPEED_KMH를 사용한다.

        Returns:
            현재 코너에서 허용할 속도 상한 [km/h].
        """
        if normal_speed_kmh is None:
            normal_speed_kmh = self.MAX_SPEED_KMH

        if corner is None:
            return float(
                normal_speed_kmh
            )

        distance = max(
            0.0,
            float(corner["distance"]),
        )

        angle = max(
            0.0,
            float(corner["angle"]),
        )

        # 코너 각도 0~90 deg를 0~1 severity로 변환한다.
        corner_severity = clamp(
            angle / 90.0,
            0.0,
            1.0,
        )

        # 실제 차량 회전반경 모델이 아직 없으므로
        # 현재 복구된 서버에서 사용하던 코너 최소 통과속도 [km/h]를 유지한다.
        min_corner_speed_kmh = 8.0

        corner_target_speed_kmh = (
            normal_speed_kmh
            - corner_severity
            * (
                normal_speed_kmh
                - min_corner_speed_kmh
            )
        )

        current_speed_mps = max(
            0.0,
            current_speed_kmh / 3.6,
        )

        # 현재 속도와 제어주기만큼의 동적 반응거리 [m].
        reaction_distance = (
            current_speed_mps * dt
        )

        usable_distance = max(
            0.0,
            distance - reaction_distance,
        )

        corner_target_speed_mps = (
            corner_target_speed_kmh / 3.6
        )

        allowed_speed_mps = math.sqrt(
            corner_target_speed_mps ** 2
            + 2.0
            * self.PLANNED_BRAKE_DECEL_MPS2
            * usable_distance
        )

        allowed_speed_kmh = (
            allowed_speed_mps * 3.6
        )

        return min(
            float(normal_speed_kmh),
            allowed_speed_kmh,
        )

    def _calculate_alignment_speed_limit(
        self,
        heading_error_deg: float,
        normal_speed_kmh: Optional[float] = None,
    ) -> float:
        """
        현재 차체와 조향 목표 방향의 오차로 속도 상한을 계산한다.

        현재 복구된 서버 식을 그대로 유지한다.

        Args:
            heading_error_deg:
                현재 차체와 조향 target heading 사이 오차 [deg].
            normal_speed_kmh:
                완전히 정렬됐을 때 최대 허용속도 [km/h].

        Returns:
            alignment 기반 목표속도 상한 [km/h].
        """
        if normal_speed_kmh is None:
            normal_speed_kmh = self.MAX_SPEED_KMH

        angle = abs(
            float(heading_error_deg)
        )

        alignment_ratio = clamp(
            1.0 - angle / 90.0,
            0.0,
            1.0,
        )

        speed_ratio = (
            alignment_ratio
            * alignment_ratio
        )

        return (
            float(normal_speed_kmh)
            * speed_ratio
        )

    def _calculate_steering_command(
        self,
        current_position: Sequence[float],
        body_yaw_deg: float,
        path: Sequence[Sequence[float]],
        current_speed_kmh: float,
        dt: float,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        D* Lite look-ahead target과 현재 yaw 오차로 moveAD 명령을 계산한다.

        Args:
            current_position:
                현재 차량 X-Z 좌표 [m].
            body_yaw_deg:
                현재 차체 yaw [deg].
            path:
                현재 압축 D* Lite 경로.
            current_speed_kmh:
                현재 차량 속도 [km/h].
            dt:
                제어 주기 [sec].

        Returns:
            (steering_command, steering_info)
        """
        lookahead_distance = (
            self.LOOKAHEAD_BASE_M
            + self.LOOKAHEAD_SPEED_GAIN
            * current_speed_kmh
        )

        target_point, target_index = (
            select_lookahead_point(
                path,
                current_position,
                lookahead_distance,
            )
        )

        if target_point is None:
            self.steering_pid.reset()

            return {
                "command": "",
                "weight": 0.0,
            }, None

        cx, cz = map(
            float,
            current_position,
        )

        tx, tz = map(
            float,
            target_point,
        )

        dx = tx - cx
        dz = tz - cz

        # 현재 시뮬레이터 좌표계: +Z = 0deg, +X = +90deg.
        target_heading_deg = (
            math.degrees(
                math.atan2(
                    dx,
                    dz,
                )
            )
            % 360.0
        )

        heading_error_deg = (
            normalize_angle_deg(
                target_heading_deg
                - body_yaw_deg
            )
        )

        steer_output = (
            self.steering_pid.update(
                heading_error_deg,
                dt,
            )
        )

        print(
            "[STEER DEBUG]",
            f"target_index={target_index}",
            f"heading_error={heading_error_deg:.2f}",
            f"steer_output={steer_output:.3f}",
            f"speed={current_speed_kmh:.2f}",
        )

        if steer_output > 0.0:
            ad_command = "D"
            ad_weight = min(
                abs(steer_output),
                self.STEER_MAX_WEIGHT,
            )

        elif steer_output < 0.0:
            ad_command = "A"
            ad_weight = min(
                abs(steer_output),
                self.STEER_MAX_WEIGHT,
            )

        else:
            ad_command = ""
            ad_weight = 0.0

        info = {
            "target_point": (
                tx,
                tz,
            ),
            "target_index": target_index,
            "lookahead_m": lookahead_distance,
            "target_heading_deg": target_heading_deg,
            "body_yaw_deg": body_yaw_deg,
            "heading_error_deg": heading_error_deg,
            "steer_output": steer_output,
        }

        print(
            "[STEER]",
            "pos=",
            current_position,
            "target=",
            target_point,
            "target_index=",
            target_index,
            "heading_error=",
            heading_error_deg,
        )

        return {
            "command": ad_command,
            "weight": round(
                ad_weight,
                4,
            ),
        }, info

    # --------------------------------------------------------
    # /get_action
    # --------------------------------------------------------

    def get_action(
        self,
        data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Flask /get_action endpoint의 전체 주행 제어를 수행한다.

        서버에서는 이 메서드 한 번만 호출하면
        D* Lite 경로 갱신, 조향, 코너 감속, 목적지 감속,
        속도 PID, 도착 정지가 모두 처리된다.

        Args:
            data:
                request.get_json(force=True) 결과.

        Returns:
            시뮬레이터에 보낼 command dictionary.
        """
        if self.stop_flag:
            self.stop_flag = False
            return {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "", "weight": 0.0},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        data = data or {}

        position = data.get(
            "position",
            {},
        )

        fallback_x = (
            self.current_pos[0]
            if self.current_pos is not None
            else 0.0
        )

        fallback_z = (
            self.current_pos[1]
            if self.current_pos is not None
            else 0.0
        )

        pos_x = float(
            position.get(
                "x",
                fallback_x,
            )
        )

        pos_z = float(
            position.get(
                "z",
                fallback_z,
            )
        )

        # D* Lite grid 이동 여부를 판단하기 위해 직전 위치를 보존한다.
        previous_pos = self.current_pos

        self.current_pos = [
            pos_x,
            pos_z,
        ]

        # 자체 인지 기반 회피/후퇴 상태 갱신.
        # dest가 아직 없어도 breadcrumb 자체는 계속 쌓아 둔다.
        self._record_position_history(self.current_pos)
        self._check_retreat_arrival(self.current_pos)

        if self.dest is None:
            self.speed_pid.reset()
            self.steering_pid.reset()
            self.last_control_time = None
            return make_stop_command()

        current_speed_kmh = (
            self.info_speed_kmh
        )

        if current_speed_kmh is None:
            self.speed_pid.reset()
            self.steering_pid.reset()

            print(
                "[/get_action] "
                "/info 속도 계산 전이므로 "
                "제어 명령을 보내지 않습니다."
            )

            return make_stop_command()

        body_yaw_deg = (
            self._read_player_body_yaw_deg()
        )

        if body_yaw_deg is None:
            self.speed_pid.reset()
            self.steering_pid.reset()

            print(
                "[/get_action] "
                "/info의 playerBodyX가 없어 "
                "제어 명령을 보내지 않습니다."
            )

            return make_stop_command()

        destination_signature = (
            round(
                float(self.dest[0]),
                3,
            ),
            round(
                float(self.dest[1]),
                3,
            ),
        )

        if (
            destination_signature
            != self.last_pid_destination
        ):
            self.speed_pid.reset()
            self.steering_pid.reset()

            self.last_control_time = None
            self.arrival_latched = False
            self.last_pid_destination = (
                destination_signature
            )

        now = time.monotonic()

        dt = (
            0.05
            if self.last_control_time is None
            else clamp(
                now - self.last_control_time,
                0.01,
                0.25,
            )
        )

        self.last_control_time = now

        # ----------------------------------------------------
        # 1) 현재 위치 기준 D* Lite 경로 갱신
        # ----------------------------------------------------
        try:
            old_grid = (
                self.planner.world_to_grid(
                    previous_pos,
                    clamp=True,
                )
                if previous_pos
                else None
            )

            new_grid = (
                self.planner.world_to_grid(
                    self.current_pos,
                    clamp=True,
                )
            )

            # 현재 복구된 서버와 동일하게 grid가 바뀌면 경로를 다시 계산한다.
            # 단, 후퇴 중(vehicle_mode == 'retreat')에는 grid가 바뀌어도
            # 전진 재탐색을 하지 않는다 -> retreat 경로가 그대로 유지된다.
            # (전진 복귀는 _check_retreat_arrival()이 도착 시점에 처리한다.)
            if (
                self.vehicle_mode == 'advance'
                and (
                    old_grid != new_grid
                    or not self.current_path
                )
            ):
                with self.planner_lock:
                    self.current_path = (
                        self.planner.find_path(
                            self.current_pos,
                            self.dest,
                            self.latest_info
                        )
                    )

                    if not self.current_path:
                        # find_path()가 예외 없이 빈 경로만 반환한 경우
                        # (시작점/목적지 자체는 안 막혔는데 그 사이에 경로가
                        # 없는 상태 -- 예: 방금 재분류된 오브젝트의 거대한
                        # 안전 반경이 두 지점 사이를 완전히 갈라놓은 경우).
                        # 그대로 두면 다음 tick에도 똑같은 조건(경로 없음)이라
                        # 다시 find_path()만 반복 호출하고 매번 빈 경로만
                        # 나오는 게 무한 반복된다. clear_start_area로 점점
                        # 넓혀가며 재시도하는 _find_path_with_recovery()로
                        # 한 번 더 시도한다.
                        print(
                            "[/get_action] find_path()가 빈 경로를 반환함"
                            "(시작/목적지 자체는 안 막혔지만 그 사이 경로가 "
                            "없는 상태) -> 비상 탈출 재시도"
                        )
                        self.current_path = (
                            self.planner._find_path_with_recovery(
                                self.current_pos, self.dest,
                            )
                        )

        except ValueError as exc:
            print(
                "D* Lite 경로 계산 실패:",
                exc,
            )
            self.current_path = []

            with self.planner_lock:
                current_grid = self.planner.world_to_grid(
                    self.current_pos, clamp=True,
                )
                standing_on_blocked_cell = not self.planner.is_free(
                    current_grid,
                )

                if standing_on_blocked_cell:
                    retreat_path = self._build_retreat_path()

                    if retreat_path is not None:
                        self.current_path = retreat_path
                        self.vehicle_mode = 'retreat'
                        print(
                            f"[/get_action] 발밑이 막혀서 후퇴 경로로 전환합니다 ({len(retreat_path)}개 지점)."
                        )
                    else:
                        self.current_path = (
                            self.planner._find_path_with_recovery(
                                self.current_pos, self.dest,
                            )
                        )
                        self.vehicle_mode = 'advance'

                    if self.current_path:
                        self.speed_pid.reset()
                        self.steering_pid.reset()

        if not self.current_path:
            self.speed_pid.reset()
            self.steering_pid.reset()

            print(
                "[/get_action] "
                "D* Lite 경로가 없어 차량을 정지합니다."
            )

            return make_stop_command()

        # ----------------------------------------------------
        # 2) 최종 목적지까지 거리
        # ----------------------------------------------------
        goal_dx = (
            float(self.dest[0])
            - pos_x
        )

        goal_dz = (
            float(self.dest[1])
            - pos_z
        )

        distance_to_goal = math.hypot(
            goal_dx,
            goal_dz,
        )

        # ----------------------------------------------------
        # 3) 조향 계산
        # ----------------------------------------------------
        (
            steering_command,
            steering_info,
        ) = self._calculate_steering_command(
            current_position=self.current_pos,
            body_yaw_deg=body_yaw_deg,
            path=self.current_path,
            current_speed_kmh=current_speed_kmh,
            dt=dt,
        )

        heading_error_deg = (
            0.0
            if steering_info is None
            else steering_info[
                "heading_error_deg"
            ]
        )

        # ----------------------------------------------------
        # 4) 목적지 제동거리 / 속도 PID
        # ----------------------------------------------------
        current_speed_mps = (
            current_speed_kmh / 3.6
        )

        brake_reaction_distance = (
            current_speed_mps * dt
        )

        physical_braking_distance = (
            current_speed_mps ** 2
            / (
                2.0
                * self.PLANNED_BRAKE_DECEL_MPS2
            )
        )

        brake_trigger_distance = (
            physical_braking_distance
            + brake_reaction_distance
        )

        print(
            "[ARRIVAL CHECK]",
            f"distance={distance_to_goal:.2f}m",
            f"stop_distance={self.STOP_DISTANCE_M:.2f}m",
            f"arrival_latched={self.arrival_latched}",
        )

        # 목적지 반경 진입 후에는 다시 일반 주행으로 돌아가지 않는다.
        if (
            self.arrival_latched
            or distance_to_goal
            <= self.STOP_DISTANCE_M
        ):
            self.arrival_latched = True

            if (
                current_speed_kmh
                > self.STOP_SPEED_KMH
            ):
                # 목표속도 0 km/h.
                final_speed_error_kmh = (
                    -current_speed_kmh
                )

                final_pid_output = (
                    self.speed_pid.update(
                        final_speed_error_kmh,
                        dt,
                    )
                )

                # 목적지에서는 전진 PID 출력을 허용하지 않는다.
                final_pid_output = min(
                    0.0,
                    final_pid_output,
                )

                command = (
                    make_longitudinal_command(
                        final_pid_output
                    )
                )

                state = "FINAL_BRAKING"

            else:
                self.speed_pid.reset()
                self.steering_pid.reset()

                command = (
                    make_stop_command()
                )

                state = "ARRIVED_STOP"

            # 최종 제동 중에는 조향 입력을 해제한다.
            command["moveAD"] = {
                "command": "",
                "weight": 0.0,
            }

            print(
                f"[/get_action CTRL] {state} | "
                f"pos=({pos_x:.2f},{pos_z:.2f}) "
                f"dest=({self.dest[0]:.2f},{self.dest[1]:.2f}) "
                f"distance={distance_to_goal:.2f}m "
                f"speed={current_speed_kmh:.2f}km/h "
                f"WS={command['moveWS']} "
                f"AD={command['moveAD']}"
            )

            return command

        # 반응거리 이후 실제로 사용할 수 있는 목적지 감속 거리 [m].
        destination_usable_distance = max(
            0.0,
            distance_to_goal
            - brake_reaction_distance,
        )

        destination_target_speed_kmh = (
            self._calculate_target_speed_kmh(
                destination_usable_distance
            )
        )

        # 조향 target과 동일한 다음 vertex를 코너 속도 기준으로 사용한다.
        target_index = (
            None
            if steering_info is None
            else steering_info[
                "target_index"
            ]
        )

        upcoming_corner = (
            calculate_next_vertex_corner(
                path=self.current_path,
                current_position=self.current_pos,
                target_index=target_index,
            )
        )

        corner_speed_limit_kmh = (
            self._calculate_corner_speed_limit(
                corner=upcoming_corner,
                current_speed_kmh=current_speed_kmh,
                dt=dt,
                normal_speed_kmh=self.MAX_SPEED_KMH,
            )
        )

        alignment_speed_limit_kmh = (
            self._calculate_alignment_speed_limit(
                heading_error_deg=heading_error_deg,
                normal_speed_kmh=self.MAX_SPEED_KMH,
            )
        )

        # 목적지/코너/정렬/최고속도 중 가장 낮은 값을 실제 목표속도로 사용한다.
        target_speed_kmh = min(
            destination_target_speed_kmh,
            corner_speed_limit_kmh,
            alignment_speed_limit_kmh,
            self.MAX_SPEED_KMH,
        )

        speed_error_kmh = (
            target_speed_kmh
            - current_speed_kmh
        )

        pid_output = (
            self.speed_pid.update(
                speed_error_kmh,
                dt,
            )
        )

        if (
            distance_to_goal
            <= brake_trigger_distance
        ):
            # 목적지 제동 영역에서는 PID가 양수여도 전진하지 않는다.
            braking_pid_output = min(
                0.0,
                pid_output,
            )

            command = (
                make_longitudinal_command(
                    braking_pid_output
                )
            )

            state = "BRAKING_FOR_GOAL"

        else:
            command = (
                make_longitudinal_command(
                    pid_output
                )
            )

            state = (
                "ACCELERATING"
                if pid_output > 0.02
                else "BRAKING"
                if pid_output < -0.02
                else "COASTING"
            )

        # 종방향 명령에 조향 명령을 결합한다.
        command["moveAD"] = (
            steering_command
        )

        if upcoming_corner is not None:
            corner_x, corner_z = (
                upcoming_corner["point"]
            )

            radius_value = (
                upcoming_corner["radius"]
            )

            radius_text = (
                "inf"
                if math.isinf(
                    float(radius_value)
                )
                else f"{float(radius_value):.2f}"
            )

            print(
                f"[/get_action CORNER] "
                f"target_index={upcoming_corner['index']} "
                f"point=({corner_x:.2f},{corner_z:.2f}) "
                f"distance={upcoming_corner['distance']:.2f}m "
                f"angle={upcoming_corner['angle']:.2f}deg "
                f"radius={radius_text}m "
                f"limit={corner_speed_limit_kmh:.2f}km/h"
            )

        else:
            print(
                f"[/get_action CORNER] "
                f"target_index={target_index} "
                f"corner=None | "
                f"limit={corner_speed_limit_kmh:.2f}km/h"
            )

        if steering_info is not None:
            target_x, target_z = (
                steering_info["target_point"]
            )

            print(
                f"[/get_action STEER] "
                f"yaw={body_yaw_deg:.2f}deg "
                f"target=({target_x:.2f},{target_z:.2f}) "
                f"targetYaw={steering_info['target_heading_deg']:.2f}deg "
                f"error={heading_error_deg:.2f}deg "
                f"lookahead={steering_info['lookahead_m']:.2f}m "
                f"AD={command['moveAD']}"
            )

        print(
            f"[/get_action CTRL] {state} | "
            f"pos=({pos_x:.2f},{pos_z:.2f}) "
            f"dest=({self.dest[0]:.2f},{self.dest[1]:.2f}) "
            f"distance={distance_to_goal:.2f}m "
            f"brake_at={brake_trigger_distance:.2f}m "
            f"targetSpeed={target_speed_kmh:.2f}km/h "
            f"cornerLimit={corner_speed_limit_kmh:.2f}km/h "
            f"speed={current_speed_kmh:.2f}km/h "
            f"speedError={speed_error_kmh:.2f} "
            f"speedPID={pid_output:.3f} "
            f"WS={command['moveWS']} "
            f"AD={command['moveAD']}"
        )

        return command