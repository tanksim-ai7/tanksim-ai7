import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import datetime
import time
from flask import Flask, request, jsonify

import torch
from ultralytics import YOLO
import math

from move.dstar_lite_planner import DStarLitePlanner, ObstacleRect

import matplotlib
matplotlib.use("Agg")  # 서버 환경(디스플레이 없음)에서도 안전하게 이미지 파일로만 저장하기 위한 백엔드
import matplotlib.pyplot as plt

from pathfinding.astar_planner import AStarPlanner, ObstacleRect
from pathfinding.dstar_lite_planner import DStarLitePlanner

# True로 두면 D* Lite 사용, False로 두면 기존 A* 그대로 사용
# (A*/D*Lite 둘 다 find_path()/set_obstacles() 인터페이스가 동일하므로
#  이 스위치 하나로 알고리즘만 갈아끼울 수 있음 - 비교 실험용)
USE_DSTAR_LITE = True


app = Flask(__name__)
model = YOLO('yolov8n.pt')

all_info = None # info 정보
idx = 0 # 경로 point의 idx
path = None # 경로에 대한 list
dest = None # 목적지에 대한 변수
path_flag = False # path finding 최초 여부 flag

planner = DStarLitePlanner(
    grid_min_x=0.0,
    grid_max_x=300.0,
    grid_min_z=0.0,
    grid_max_z=300.0,
    cell_size=1.0,
    obstacle_margin=2.0,
    allow_diagonal=True,
)

print(model.names)


@app.before_request
def _log_every_request():
    """
    /update_obstacle 요청이 서버에 도달하는지 자체를 의심하는 상황이라,
    라우트별 로직과 무관하게 '들어오는 모든 요청'을 무조건 한 줄씩 남긴다.
    이 로그에 /update_obstacle 이 단 한 번도 안 찍히면 Flask/서버 코드 문제가 아니라
    Unity(클라이언트) 쪽에서 이 요청 자체를 안 보내고 있다는 뜻.
    """
    print(f"🌐 [REQUEST] {request.method} {request.path} "
          f"content_type={request.content_type} content_length={request.content_length}")

# ----------------------------------------------------------------------
# 경로 플래너 설정 (A* / D* Lite 공용 - 인터페이스가 동일함)
# ----------------------------------------------------------------------
# 맵 크기는 실제 Terrain 크기에 맞게 수정 필요 (여기서는 300x300 가정)
_PLANNER_KWARGS = dict(
    grid_min_x=0.0,
    grid_max_x=300.0,
    grid_min_z=0.0,
    grid_max_z=300.0,
    cell_size=1.0,
    obstacle_margin=3.0,   # 전차 크기 감안한 여유 마진
    allow_diagonal=True,
)

if USE_DSTAR_LITE:
    planner = DStarLitePlanner(**_PLANNER_KWARGS)
    print("🧭 Planner: D* Lite (incremental replanning)")
else:
    planner = AStarPlanner(**_PLANNER_KWARGS)
    print("🧭 Planner: A* (full recompute every time)")


def timed_find_path(start, goal):
    """
    planner.find_path() 를 호출하면서, 알고리즘 종류와 무관하게 항상 동일한 방식으로
    걸린 시간을 재고 출력하는 공통 wrapper.

    - D* Lite: find_path() 내부에 이미 자체 타이머(full_init/incremental 구분)가 있어서
      "⏱️ [D*Lite] ..." 로그가 한 번 더 찍히는데, 그건 탐색(ComputeShortestPath)만 잰 값이고
      여기 wrapper가 재는 시간은 경로 재구성(_reconstruct_path)까지 포함한 "find_path() 전체"
      시간이라 서로 다른 걸 보는 것. 두 로그를 같이 보면 "탐색 자체" vs "경로 좌표 리스트로
      만드는 것까지 포함한 총 시간"을 나눠서 볼 수 있음.
    - A*: find_path() 안에 타이머가 없으므로, 알고리즘 비교를 위해선 이 wrapper가 유일한
      시간 측정 지점이 됨.

    두 알고리즘 모두 "server가 planner.find_path()를 부르기 직전 ~ 결과(waypoint 리스트)를
    돌려받은 직후"를 기준으로 재기 때문에, USE_DSTAR_LITE 스위치만 바꿔가며 비교해도
    측정 기준이 동일함.
    """
    algo_name = "D*Lite" if USE_DSTAR_LITE else "A*"
    t0 = time.perf_counter()
    path = planner.find_path(start, goal)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"⏱️ [{algo_name}] find_path() 총 소요시간: {elapsed_ms:.3f} ms "
          f"(path_len={len(path)})")
    return path

# 경로 추종 관련 전역 상태
current_destination = None     # (x, z) 튜플
current_path = []              # planner.find_path() 결과: [(x, z), ...]
path_index = 0                 # 현재 추종 중인 waypoint 인덱스
last_position = None           # 이전 프레임의 (x, z)
last_valid_heading = None      # 마지막으로 "의미 있다고 판단한" 진행 방향 벡터 (노이즈 필터링용)
heading_history = []           # 최근 유효 heading 벡터들 (평균 내서 노이즈를 더 줄이기 위함)
steering_mode = "straight"     # "straight"(직진) / "turning"(조향) 상태 기계 - 매 프레임 반응 방지용
turning_frame_count = 0        # 조향 모드가 연속 몇 프레임째인지 - 원운동에 갇히는 것 탈출용
forced_straight_countdown = 0  # 원운동 탈출 시 "몇 프레임 동안 강제로 직진 유지할지" 카운트다운
forced_straight_start_dist = None  # 강제 직진 시작 시점의 목표 거리 (방향이 틀렸는지 판단용)
last_angle_diff = None         # 오버슈트(목표를 지나쳐버림) 감지용 - 직전 프레임의 각도차

# /info 엔드포인트로 들어오는 실제 차체 회전각 (get_action에는 이 정보가 없음 - API 문서로 확인됨)
latest_body_x = None
latest_body_y = None
latest_body_z = None
latest_body_updated_at = None   # /info로 body 값이 마지막으로 갱신된 시각 (신선도 확인용)

WAYPOINT_REACH_THRESHOLD = 3.0   # 이 거리 안으로 들어오면 다음 waypoint로 전환
DESTINATION_REACH_THRESHOLD = 3.0
ANGLE_STOP_THRESHOLD = 3.0       # 이 각도(도) 이내면 정렬됐다고 판단
ANGLE_ENTER_TURN = 25.0          # 직진 모드에서 이 각도 이상 벗어나야 조향 모드로 전환
ANGLE_EXIT_TURN = 8.0            # 조향 모드에서 이 각도 이내로 들어와야 다시 직진 모드로 전환
MAX_TURNING_FRAMES = 60          # 조향 모드가 이 프레임 수를 넘기면 강제로 직진시켜 원운동 탈출
                                  # (원래 20이었는데, 실제 회전각 데이터로 확인해본 결과
                                  #  목표가 거의 정반대 방향일 때 유턴하는 데 14~19프레임 정도
                                  #  걸리는 게 정상이라, 20으로는 유턴이 거의 끝나갈 때마다
                                  #  번번이 강제 중단시켜서 처음부터 다시 돌게 만드는 원인이었음.
                                  #  실제 회전각 데이터를 쓰는 지금은 노이즈로 인한 가짜 원운동
                                  #  걱정이 줄었으니 여유있게 늘림)
FORCED_STRAIGHT_DURATION = 12    # 원운동 탈출 시 최소 이만큼 프레임은 무조건 직진 유지
                                  # (한 프레임만 직진하면 방향이 거의 안 바뀌어서 곧바로 같은
                                  #  원운동에 다시 빠짐 - 실제 로그로 확인된 문제. 충분히 오래
                                  #  직진해야 heading_vec도 정확해지고 궤적 자체도 바뀜)
FORCED_STRAIGHT_ABORT_RATIO = 1.3  # 강제 직진 시작 시점 거리 대비, 이 배율 이상 더 멀어지면
                                    # (방향이 틀렸다는 뜻이므로) 즉시 중단하고 조향 모드로 복귀
                                    # * 각도 기준으로 체크하면 애초에 조향 모드에 들어올 때부터
                                    #   각도차가 크므로 시작하자마자 중단돼버리는 문제가 있어서
                                    #   "실제로 목표에 가까워지고 있는가"를 기준으로 바꿈
MIN_MOVE_FOR_HEADING = 0.5       # 이 거리 이상 움직였을 때만 heading_vec을 갱신 (미세 흔들림 무시)
HEADING_HISTORY_SIZE = 4         # 최근 몇 개의 유효 heading을 평균 낼지
PATH_DIVERGENCE_THRESHOLD = 15.0 # 목표 waypoint에서 이 거리 이상 벗어나면 경로를 강제로 재계산
ENABLE_DIVERGENCE_REPLAN = False  # False로 하면 "처음에 한 번 계획한 경로만 계속 따라가기" 모드
                                   # (조향이 불안정해서 경로를 크게 벗어나도 재계산하지 않고,
                                   #  그냥 지금 경로의 waypoint를 계속 목표로 삼아 조향만 시도함)

# Unity 쪽 A/D 회전 방향이 우리 코드의 가정과 반대일 수 있어서 넣어둔 스위치.
# 실제로 목적지 반대 방향으로 계속 발산하는 증상이 있으면 True로 바꿔서 테스트해볼 것.
INVERT_STEERING = True   # 이전 로그 분석 결과 True가 맞는 것으로 확인됨

MAX_FORWARD_SPEED = 1.0  # 테스트용 속도 제한. 1.0=최대. 0.5로 낮추면 모든 전진 명령의
                          # weight가 절반으로 줄어듦 (조향 로직 자체는 안 바뀜, 순수 속도만 제한)

# 경로 이미지를 저장할 폴더
PATH_IMAGE_DIR = "path_snapshots"
os.makedirs(PATH_IMAGE_DIR, exist_ok=True)


def save_path_image(planner_obj, path, start, goal, filename=None):
    """
    현재 장애물 + 계산된 경로를 matplotlib으로 그려서 PNG 파일로 저장.
    (astar_planner.AStarPlanner.plot()과 거의 같은 로직이지만,
     plt.show() 대신 파일로 저장하도록 바꾼 버전 - 서버에서는 화면이 없기 때문)
    """
    # AStarPlanner는 grid-built 여부를 `_grid_valid`로, DStarLitePlanner는 `_grid_built`로
    # 관리해서 속성 이름이 서로 다름 (동일한 의미: "grid_size_x*grid_size_z 배열이
    # 이미 생성되어 있는가"). 어느 플래너가 들어와도 동작하도록 둘 다 확인.
    grid_already_built = getattr(planner_obj, "_grid_valid", None)
    if grid_already_built is None:
        grid_already_built = getattr(planner_obj, "_grid_built", False)
    if not grid_already_built:
        planner_obj._build_grid()

    fig, ax = plt.subplots(figsize=(6, 6))

    # 장애물(마진 포함) 그리기
    for obs in planner_obj._obstacles:
        x_min = obs.x_min - planner_obj.obstacle_margin
        x_max = obs.x_max + planner_obj.obstacle_margin
        z_min = obs.z_min - planner_obj.obstacle_margin
        z_max = obs.z_max + planner_obj.obstacle_margin
        w = x_max - x_min
        h = z_max - z_min
        rect = plt.Rectangle((x_min, z_min), w, h, alpha=0.3, color="gray")
        ax.add_patch(rect)

    # 경로
    if path:
        xs = [p[0] for p in path]
        zs = [p[1] for p in path]
        ax.plot(xs, zs, marker="o", markersize=3, linewidth=1.5, color="blue")

    # 시작점 / 목적지 표시
    ax.plot(start[0], start[1], marker="^", markersize=10, color="green", label="start")
    ax.plot(goal[0], goal[1], marker="*", markersize=12, color="red", label="goal")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(planner_obj.grid_min_x, planner_obj.grid_max_x)
    ax.set_ylim(planner_obj.grid_min_z, planner_obj.grid_max_z)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    algo_name = "D* Lite" if USE_DSTAR_LITE else "A*"
    ax.set_title(f"{algo_name} path")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"path_{timestamp}.png"
    filepath = os.path.join(PATH_IMAGE_DIR, filename)

    fig.savefig(filepath)
    plt.close(fig)  # 서버가 계속 켜져 있으므로 figure를 닫아서 메모리 누수 방지

    # 계산 시간 로그 (D* Lite는 last_compute_time_ms / last_compute_type 을 제공함)
    compute_info = ""
    if hasattr(planner_obj, "last_compute_time_ms"):
        compute_info = (f", compute={planner_obj.last_compute_time_ms:.3f}ms"
                         f"({planner_obj.last_compute_type})")
    print(f"🖼️ Path image saved: {filepath}{compute_info}")
    return filepath


# ----------------------------------------------------------------------
# 유틸리티: 각도 계산 및 조향 명령 변환
# ----------------------------------------------------------------------
def _angle_deg(v):
    """벡터 (dx, dz) -> 월드 기준 각도(도, -180~180)"""
    return math.degrees(math.atan2(v[1], v[0]))


def _normalize_angle(angle):
    """각도를 -180~180 범위로 정규화"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def _build_command(move_ws, move_ad, fire=False):
    """조향/이동 파라미터를 combined_commands 포맷으로 변환"""
    ws_cmd, ws_w = move_ws
    ad_cmd, ad_w = move_ad
    return {
        "moveWS": {"command": ws_cmd, "weight": ws_w},
        "moveAD": {"command": ad_cmd, "weight": ad_w},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": fire,
    }


def _stop_command():
    return _build_command(("STOP", 1.0), ("", 0.0))


def reset_planning_state():
    """
    새 에피소드 시작(재시작) 시 서버 쪽 경로 계획/조향 상태를 전부 초기화.
    /init 뿐 아니라 실제로 재시작 버튼이 호출하는 /start 에서도 호출해야
    이전 목적지/경로/조향 상태가 새 에피소드까지 이어지지 않음.
    """
    global current_destination, current_path, path_index
    global last_position, last_valid_heading, heading_history
    global steering_mode, turning_frame_count, forced_straight_countdown, forced_straight_start_dist
    global last_angle_diff

    current_destination = None
    current_path = []
    path_index = 0
    last_position = None
    last_valid_heading = None
    heading_history = []
    steering_mode = "straight"
    turning_frame_count = 0
    forced_straight_countdown = 0
    forced_straight_start_dist = None
    last_angle_diff = None

    # D* Lite는 이전 탐색(g/rhs, km, start/goal 노드)을 내부에 들고 있으므로,
    # 에피소드가 재시작되면(탱크 위치가 완전히 바뀌므로) 이 상태를 반드시 리셋해야 함.
    # 안 그러면 새 에피소드의 첫 find_path()가 "이전 에피소드 위치에서 조금 이동한 것"으로
    # 착각해서 잘못된 증분 재계산을 시도할 수 있음. (장애물/그리드 자체는 유지)
    if hasattr(planner, "reset"):
        planner.reset()


def compute_move_command(current_pos, heading_vec, target_pos, has_real_heading=False):
    """
    현재 위치/진행방향과 목표 waypoint를 비교해서
    전진/조향 명령을 계산.

    heading_vec: 최근 이동 방향 추정 벡터 (dx, dz). None이면 정면(1,0) 가정.
    has_real_heading: /info로부터 받은 실제 회전각을 쓰고 있는지 여부.
        True면 전진 없이도 heading을 정확히 알 수 있으므로, 각도차가 클 때
        전진을 완전히 끊고 제자리 회전(pivot turn)만 하도록 허용함 - 이러면
        넓게 도는 유턴 대신 좁게 제자리에서 방향을 틀 수 있어서 장애물 충돌을 줄일 수 있음.
        False(위치 추정 fallback)면 예전처럼 최소 전진을 유지함 - 안 그러면
        위치가 안 바뀌어서 heading을 영원히 갱신 못 하는 '죽은 순환'에 빠질 수 있음.

    [조향 상태 기계 방식으로 변경]
    예전엔 각도차가 조금만 나도 매 프레임 즉시 A/D를 바꿔가며 반응했는데,
    이러면 "직진하며 정확한 heading을 잴 기회" 자체가 없어서 부정확한 추정치를
    쫓아 계속 방향을 바꾸다가 큰 원을 그리며 도는 현상이 생겼음(실제 로그로 확인됨).

    이제는 "직진(straight) / 조향(turning)" 두 상태를 히스테리시스로 오가도록 바꿈:
    - 직진 모드: 각도차가 ANGLE_ENTER_TURN(25도)을 넘기 전까지는 방향을 절대 안 건드리고
      계속 똑바로 감. 이 구간에서 heading_vec이 훨씬 정확하게 측정됨.
    - 조향 모드: 한번 크게 벗어났다고 판단되면, 각도차가 ANGLE_EXIT_TURN(8도) 이내로
      들어올 때까지는 계속 조향에 집중. 살짝 정렬됐다고 바로 다시 안 바꿈(스치듯 지나가는
      정렬을 신뢰하지 않음).
    """
    global steering_mode, turning_frame_count, forced_straight_countdown, forced_straight_start_dist

    dx = target_pos[0] - current_pos[0]
    dz = target_pos[1] - current_pos[1]
    dist = math.hypot(dx, dz)

    if heading_vec is None or (heading_vec[0] == 0 and heading_vec[1] == 0):
        heading_vec = (1.0, 0.0)

    target_angle = _angle_deg((dx, dz))
    heading_angle = _angle_deg(heading_vec)
    angle_diff = _normalize_angle(target_angle - heading_angle)

    global last_angle_diff

    # 강제 직진 구간이 아직 남아있으면, 각도 계산과 무관하게 무조건 직진 유지.
    # (원운동 탈출 직후 한 프레임만 직진하면 방향이 거의 안 바뀌어서 곧바로 같은
    #  원운동에 다시 빠지는 문제가 실제로 있었음 - 최소 지속시간을 보장해줘야 함)
    #
    # 단, 방향 확인 없이 무조건 N프레임을 직진시키면, 하필 반대 방향을 보고 있을 때
    # 오히려 목표에서 더 멀어지는 문제가 실제로 발생했음. 그래서 "시작 시점 거리보다
    # 실제로 더 멀어지고 있는지"를 매 프레임 확인해서, 악화되면 즉시 중단하고 조향 모드로 복귀.
    # (각도 기준으로 체크하면 애초에 조향 모드에 들어올 때부터 각도차가 크기 때문에
    #  시작하자마자 중단돼버리는 문제가 있어서 거리 기준으로 바꿈)
    if forced_straight_countdown > 0:
        if forced_straight_start_dist is not None and dist > forced_straight_start_dist * FORCED_STRAIGHT_ABORT_RATIO:
            forced_straight_countdown = 0
            forced_straight_start_dist = None
            steering_mode = "turning"
            turning_frame_count = 0
        else:
            forced_straight_countdown -= 1
            steering_mode = "straight"
            turning_frame_count = 0
            if forced_straight_countdown == 0:
                forced_straight_start_dist = None
    else:
        # 상태 전환 (히스테리시스)
        if steering_mode == "straight":
            if abs(angle_diff) > ANGLE_ENTER_TURN:
                steering_mode = "turning"
                turning_frame_count = 0
        else:  # "turning"
            # 오버슈트(목표를 지나쳐버림) 감지: 실제 회전각 데이터를 쓰기 시작한 뒤,
            # 탱크의 회전 속도가 한 프레임(서버 폴링 주기)당 15~20도씩 될 만큼 빨라서,
            # "각도차가 8도 이내로 들어오는 순간"을 매번 건너뛰고 계속 같은 방향으로
            # 돌아버리는 문제가 실제로 확인됨. 그래서 각도차 부호가 뒤집히면
            # (=목표를 이미 지나쳐서 반대편으로 넘어갔으면) 정확히 8도 이내가 아니어도
            # "충분히 정렬됐다"고 보고 즉시 조향을 멈춤.
            overshot = (
                last_angle_diff is not None
                and abs(last_angle_diff) > ANGLE_EXIT_TURN
                and (last_angle_diff > 0) != (angle_diff > 0)
            )
            if abs(angle_diff) < ANGLE_EXIT_TURN or overshot:
                steering_mode = "straight"
                turning_frame_count = 0
            else:
                turning_frame_count += 1
                if turning_frame_count > MAX_TURNING_FRAMES:
                    # 최대 조향(1.0)을 계속 유지하면 실제 차량 물리 특성상 고정 반지름의
                    # 원을 그리며 영원히 도는 상태(limit cycle)에 갇힐 수 있음 - 실제로
                    # 로그에서 waypoint 주변을 계속 맴돌며 각도차가 절대 안 줄어드는 현상으로 확인됨.
                    # 이 경우 지금 각도 계산과 무관하게 여러 프레임(FORCED_STRAIGHT_DURATION)
                    # 동안 강제로 직진시켜서 궤적 자체를 바꿔버림 (단, 위의 abort 조건으로 보호됨).
                    steering_mode = "straight"
                    turning_frame_count = 0
                    forced_straight_countdown = FORCED_STRAIGHT_DURATION
                    forced_straight_start_dist = dist

    last_angle_diff = angle_diff


    if steering_mode == "straight":
        # 직진 모드: 약간의 각도차는 무시하고 그냥 쭉 감 (조향 완전히 끔)
        move_ad = ("", 0.0)
        forward_weight = 1.0
        is_pure_pivot = False
    else:
        # 전진 여부를 먼저 결정 (제자리 회전인지 아닌지에 따라 조향 강도를 다르게 줄 것이므로)
        if has_real_heading:
            PIVOT_ONLY_ANGLE = 30.0  # 이 각도 이상 벗어나 있으면 전진 없이 제자리 회전만
            is_pure_pivot = abs(angle_diff) > PIVOT_ONLY_ANGLE
            if is_pure_pivot:
                forward_weight = 0.0
            else:
                forward_weight = 1.0 - abs(angle_diff) / PIVOT_ONLY_ANGLE
        else:
            MIN_FORWARD_WHILE_TURNING = 0.2
            forward_weight = max(MIN_FORWARD_WHILE_TURNING, 1.0 - abs(angle_diff) / 180.0)
            is_pure_pivot = False

        # 조향 모드: 정렬될 때까지 계속 회전
        TURN_SATURATION_ANGLE = 120.0
        turn_weight = min(1.0, abs(angle_diff) / TURN_SATURATION_ANGLE)

        # 순수 제자리 회전(전진 없음)일 때 회전 강도를 낮췄던 이유는, 예전에 관성 때문에
        # 위치가 계속 밀리면서 오버슈트 감지가 씹히는 문제가 있어서였음(0.5로 안전하게 제한).
        # 그런데 STOP 명령으로 실제 위치가 완전히 고정된다는 게 로그로 확인됐으니
        # (dist_to_wp가 회전 내내 그대로 유지됨), 그 제약을 완화해서 회전 속도를 올림.
        # 오버슈트 감지(부호 반전)는 회전 속도와 무관하게 계속 보호막 역할을 해줌.
        if is_pure_pivot:
            PIVOT_MAX_TURN_WEIGHT = 1.0
            turn_weight = min(turn_weight, PIVOT_MAX_TURN_WEIGHT)

        turn_right = angle_diff > ANGLE_STOP_THRESHOLD
        turn_left = angle_diff < -ANGLE_STOP_THRESHOLD

        if INVERT_STEERING:
            turn_right, turn_left = turn_left, turn_right

        if turn_right:
            move_ad = ("D", turn_weight)
        elif turn_left:
            move_ad = ("A", turn_weight)
        else:
            move_ad = ("", 0.0)

    # 테스트용 속도 제한 - forward_weight를 계산한 뒤 마지막에 한번 더 스케일링.
    # 조향 판단 로직 자체는 그대로 두고, 순수하게 "얼마나 세게 W를 누르는지"만 낮춤.
    forward_weight *= MAX_FORWARD_SPEED

    if is_pure_pivot:
        # ("W", 0.0)으로는 관성 때문에 실제로 안 멈추고 계속 미끄러지는 문제가
        # 실제 로그로 확인됨 (전진 weight 0인데도 위치가 수십 유닛씩 계속 이동).
        # API 문서상 moveWS에는 W/S와 별개로 STOP이라는 명시적 제동 명령이 있어서,
        # 그냥 "안 누르는" 것과 "적극적으로 멈추라"는 걸 구분해서 STOP을 사용.
        move_ws = ("STOP", 1.0)
    else:
        move_ws = ("W", forward_weight)

    return move_ws, move_ad, dist


# ----------------------------------------------------------------------
# 기존 엔드포인트
# ----------------------------------------------------------------------
@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()
    target_classes = {0: "tank", 1: "rock", 2: "car", 7: "truck", 15: "rock"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)


@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    left_path = "temp_left.jpg"
    right_path = "temp_right.jpg"

    try:
        left_image.save(left_path)
        right_image.save(right_path)
    except Exception as e:
        return jsonify({"result": "error", "message": str(e)}), 500

    return jsonify({"result": "success"})


@app.route('/info', methods=['POST'])
def info():
    global latest_body_x, latest_body_y, latest_body_z, latest_body_updated_at

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

<<<<<<< HEAD
    global all_info
    all_info = data # info 정보 전역변수에 저장

    #print("📨 /info data received:", data)
=======
    # 디버그: /info 자체가 호출되고 있는지, 어떤 키들이 들어있는지 무조건 확인하기 위함.
    # (/info가 아예 안 불리는 건지, 불리는데 playerBodyX가 없는 건지 구분하기 위함)
    print(f"📋 /info called, keys={list(data.keys())}")

    # /get_action에는 회전 정보가 없고, 이 /info 엔드포인트로만 실제 차체 회전각이 온다
    # (API 문서 확인 결과). playerBodyX/Y/Z 세 값을 모두 저장해두고 로그로 남겨서,
    # 실제로 조향할 때 어느 축이 뚜렷하게 바뀌는지 확인 후 get_action에서 사용할 값을 정한다.
    if "playerBodyX" in data:
        latest_body_x = data.get("playerBodyX")
        latest_body_y = data.get("playerBodyY")
        latest_body_z = data.get("playerBodyZ")
        latest_body_updated_at = time.time()
        print(f"🧭 /info body rotation: X={latest_body_x}, Y={latest_body_y}, Z={latest_body_z}")
>>>>>>> 263804ccd93f6321e12b19a5ac16b427cc88a4ed

    return jsonify({"status": "success", "control": ""})

<<<<<<< HEAD
def get_angle_diff(target, current):
    diff = (target - current + 180) % 360 - 180
    return diff
=======

# ----------------------------------------------------------------------
# A* 경로 계획 연동 엔드포인트
# ----------------------------------------------------------------------
@app.route('/set_destination', methods=['POST'])
def set_destination():
    global current_destination, current_path, path_index

    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        current_destination = (x, z)
        # 목적지가 새로 들어오면 이전 경로는 무효화 -> 다음 /get_action에서 재계산
        current_path = []
        path_index = 0
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400


@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    # force=True: Unity가 Content-Type: application/json 헤더 없이 요청을 보내는 경우
    # request.get_json()이 에러 없이 그냥 None을 반환해서 아래 400 분기로 조용히 빠지는
    # 문제가 있었음 (/info 에서 이미 한 번 겪었던 것과 동일한 문제).
    data = request.get_json(force=True, silent=True)
    if not data:
        # 여기서 아무 로그도 안 남기면 "요청이 아예 안 옴"과 "요청은 왔는데 파싱 실패"를
        # 구분할 수 없어서 디버깅이 막혔던 것이므로, 실패해도 반드시 흔적을 남김
        print(f"🪨 [DEBUG] /update_obstacle 호출됨 but data가 비어있음. "
              f"raw body={request.get_data(as_text=True)[:200]!r}, "
              f"content_type={request.content_type}")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = []
    for item in data.get("obstacles", []):
        try:
            obstacles.append(ObstacleRect.from_min_max(
                x_min=float(item["x_min"]),
                x_max=float(item["x_max"]),
                z_min=float(item["z_min"]),
                z_max=float(item["z_max"]),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    planner.set_obstacles(obstacles)

    # 장애물이 바뀌면 기존 경로는 더 이상 유효하지 않을 수 있으므로 재계산 유도
    global current_path, path_index
    current_path = []
    path_index = 0

    print(f"🪨 Obstacle Data received, {len(obstacles)} obstacles set")
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

>>>>>>> 263804ccd93f6321e12b19a5ac16b427cc88a4ed

@app.route('/get_action', methods=['POST'])
def get_action():
    global current_path, path_index, last_position, current_destination

    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0.0)
    pos_y = position.get("y", 0.0)
    pos_z = position.get("z", 0.0)

    current_pos = (pos_x, pos_z)

    # /get_action 페이로드 자체에는 회전 정보가 없음 (API 문서로 확인됨).
    # 대신 /info 엔드포인트가 주기적으로 보내주는 playerBodyX를 전역변수(latest_body_x)에
    # 저장해뒀다가 여기서 사용. 실제 로그로 확인한 결과 Y,Z는 거의 항상 0에 가깝고
    # X만 의미 있는 값(예: 11.5도)을 가지고 있어서, 이 프로젝트에서는 X가 실제 좌우
    # 선회(요, yaw)값인 것으로 확인됨 (Unity 표준 관례와 다름).
    real_heading_vec = None
    if latest_body_x is not None:
        math_angle_deg = _normalize_angle(90.0 - float(latest_body_x))
        math_angle_rad = math.radians(math_angle_deg)
        real_heading_vec = (math.cos(math_angle_rad), math.sin(math_angle_rad))

<<<<<<< HEAD
    degree = all_info['playerBodyX']
    now_speed = all_info['playerSpeed']

    # print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    # print(f"🎯 Turret received: x={turret_x}, y={turret_y}")
    if path:
        if idx >= len(path):
            return jsonify({"moveAD": {"command": "", "weight": 0}, "moveWS": {"command": "", "weight": 0}})
        
        target = path[idx]

        dx = target[0] - pos_x
        dz = target[1] - pos_z

        distance = math.hypot(dx, dz) # 대각선 거리 -> 직선 거리

        target_angle = math.degrees(math.atan2(dx, dz))
        if target_angle < 0:
            target_angle += 360

        angle_diff = get_angle_diff(target_angle, degree)
        abs_angle_diff = abs(angle_diff)


        MAX_SPEED = 20.0 
    
        if abs_angle_diff > 15:
            target_speed = 0.05  
        else:
            target_speed = max(6, MAX_SPEED * (1 - (abs_angle_diff / 15.0)))


        if now_speed > 5.0:
            BRAKING_DISTANCE = 30.0
        else:
            BRAKING_DISTANCE = max(5.0, math.sqrt(max(0.0, now_speed)) * 4.5)

        if distance < BRAKING_DISTANCE:
            distance_ratio = max(0.0, min(1.0, distance / BRAKING_DISTANCE))
            braking_curve = distance_ratio * distance_ratio * distance_ratio * distance_ratio
            distance_target_speed = max(1.5, MAX_SPEED * braking_curve)
            target_speed = min(target_speed, distance_target_speed)



        forward_power = 0.0
        brake_power = 0.0
        ws_command = "W"

        if now_speed < target_speed:
            ws_command = "W"
            speed_diff = target_speed - now_speed
            forward_power = min(1.0, max(0.4, speed_diff / 1.5))
        else:
            if now_speed <= 3.0:
                ws_command = "W"
                forward_power = 0.0  
                brake_power = 0.0
            else:
                ws_command = "S"
                if now_speed > 15.0 and distance < BRAKING_DISTANCE:
                    brake_power = 1.0
                else:
                    speed_gap = now_speed - target_speed
                    brake_power = min(1.0, max(0.4, (speed_gap * speed_gap) / 1.0))

                if now_speed < 8.0:
                    brake_power = min(0.35, brake_power)



        speed_ratio_squared = (now_speed / MAX_SPEED) ** 2
        STOPPING_DISTANCE = max(5.5, speed_ratio_squared * 65.0)

        if distance < STOPPING_DISTANCE and now_speed > 0.5:
            if ws_command == "W":
                forward_power = 0.0
                
            ws_command = "S"
            
            # 목적지에 가까워질수록(ratio가 1.0에 가까워짐), 속도가 빠를수록 강하게 제동
            stop_distance_ratio = (STOPPING_DISTANCE - distance) / STOPPING_DISTANCE
            stop_speed_ratio = min(1.0, now_speed / MAX_SPEED)
            
            # 승수를 3.5로 상향하여 65m 영역 진입과 동시에 풀 브레이크(1.0)가 조기에 걸리도록 유도
            stopping_intensity = min(5.0, max(0.5, stop_distance_ratio * stop_speed_ratio * 15.0))
            brake_power = max(brake_power, stopping_intensity)



        speed_factor = max(0.35, 1.0 - (now_speed / 90.0))
        turn_power = min(0.5 * speed_factor, abs_angle_diff / 20.0)
        
        command = {}
        if angle_diff > 1.0:    
            command["moveAD"] = {"command": "D", "weight": turn_power}
        elif angle_diff < -1.0:
            command["moveAD"] = {"command": "A", "weight": turn_power}
        else:
            command["moveAD"] = {"command": "", "weight": 0}

        if ws_command == "W":
            command["moveWS"] = {"command": "W", "weight": forward_power}
        else:
            command["moveWS"] = {"command": "S", "weight": brake_power}



        angle_mitigation = max(0.15, 1.0 - (abs_angle_diff / 60.0))
        arrival_threshold = max(1.5, (now_speed * 0.2) * angle_mitigation) 
        
        if distance < arrival_threshold:
            idx += 1
    else: # path가 없을 경우
        command = {
            "moveWS": {"command": "STOP", "weight": 0.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }

    # print("🔁 Sent Combined Action:", command)
=======
    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    if latest_body_x is not None:
        staleness = time.time() - latest_body_updated_at if latest_body_updated_at else None
        stale_str = f"{staleness:.2f}s ago" if staleness is not None else "unknown"
        print(f"🧭 Using /info body rotation: X={latest_body_x}, Y={latest_body_y}, Z={latest_body_z} "
              f"-> math_angle(from X)={_normalize_angle(90.0 - float(latest_body_x)):.1f} "
              f"(updated {stale_str})")

    # 목적지가 아직 없으면 정지
    if current_destination is None:
        return jsonify(_stop_command())

    # 경로가 없으면(최초 호출이거나 장애물/목적지가 갱신됐으면) 새로 계산
    if not current_path:
        current_path = timed_find_path(current_pos, current_destination)
        path_index = 0
        if not current_path:
            print("⚠️ No path found")
            return jsonify(_stop_command())

        # 경로가 새로 만들어진 시점에 이미지로 저장 (매 프레임이 아니라 재계산될 때만)
        try:
            save_path_image(planner, current_path, current_pos, current_destination)
        except Exception as e:
            # stderr로 새서 로그에 안 보일 수 있으니, 반드시 stdout(print)으로도 남김
            import traceback
            print(f"🖼️ [ERROR] 경로 이미지 저장 실패: {e}")
            traceback.print_exc()

    # 목적지 도달 여부 확인
    dist_to_dest = math.hypot(
        current_destination[0] - pos_x,
        current_destination[1] - pos_z,
    )
    if dist_to_dest < DESTINATION_REACH_THRESHOLD:
        print("✅ Destination reached")
        current_path = []
        path_index = 0
        # current_destination도 같이 비워야 함. 안 그러면 다음 /get_action 호출 때
        # current_path가 비어있으니 또 find_path()를 재계산하고, 도착 판정도 다시 나서
        # 매 프레임마다 "이미지 저장 + 도착 로그"가 무한 반복되는 낭비가 있었음
        # (실제 로그에서 확인됨 - 도착 후 수십 번 연속으로 같은 로그가 찍힘)
        current_destination = None
        return jsonify(_stop_command())

    # 현재 waypoint 도달했으면 다음으로 전환
    while path_index < len(current_path):
        wp = current_path[path_index]
        d = math.hypot(wp[0] - pos_x, wp[1] - pos_z)
        if d < WAYPOINT_REACH_THRESHOLD:
            path_index += 1
        else:
            break

    if path_index >= len(current_path):
        # 경로 소진 -> 재계산
        current_path = timed_find_path(current_pos, current_destination)
        path_index = 0
        if not current_path:
            return jsonify(_stop_command())
        try:
            save_path_image(planner, current_path, current_pos, current_destination)
        except Exception as e:
            # stderr로 새서 로그에 안 보일 수 있으니, 반드시 stdout(print)으로도 남김
            import traceback
            print(f"🖼️ [ERROR] 경로 이미지 저장 실패: {e}")
            traceback.print_exc()

    target_waypoint = current_path[path_index]

    # 경로에서 너무 멀리 벗어났으면(조향이 잘못돼서 발산하는 등) 낡은 waypoint를
    # 붙잡고 있지 말고 현재 위치 기준으로 경로를 강제로 다시 계산.
    # ENABLE_DIVERGENCE_REPLAN=False면 이 안전장치를 끄고, 처음 계획한 경로의
    # waypoint를 계속 목표로 삼아 조향만 계속 시도함 ("한 번 계획 -> 그대로 추종" 모드).
    if ENABLE_DIVERGENCE_REPLAN:
        dist_to_target_wp = math.hypot(target_waypoint[0] - pos_x, target_waypoint[1] - pos_z)
        if dist_to_target_wp > PATH_DIVERGENCE_THRESHOLD:
            print(f"⚠️ Diverged from path (dist_to_wp={dist_to_target_wp:.2f}) - replanning")
            current_path = timed_find_path(current_pos, current_destination)
            path_index = 0
            if not current_path:
                return jsonify(_stop_command())
            try:
                save_path_image(planner, current_path, current_pos, current_destination)
            except Exception as e:
                # stderr로 새서 로그에 안 보일 수 있으니, 반드시 stdout(print)으로도 남김
                import traceback
                print(f"🖼️ [ERROR] 경로 이미지 저장 실패: {e}")
                traceback.print_exc()
            target_waypoint = current_path[path_index]

    # 진행 방향 추정 (이전 위치 대비 이동 벡터)
    # 미세한 물리 흔들림(0.01~0.03 수준의 위치 변화)만으로는 heading을 갱신하지 않고,
    # 실제로 어느 정도 움직였을 때만("MIN_MOVE_FOR_HEADING" 이상) 새 heading으로 신뢰.
    # 그렇지 않으면 노이즈 수준의 벡터 방향이 그대로 조향 계산에 들어가서
    # 매 프레임 엉뚱한 방향으로 튀는 원인이 됨.
    global last_valid_heading, heading_history
    if real_heading_vec is not None:
        # Unity가 실제 회전각(body.x)을 보내주면 이걸 그대로 신뢰한다.
        # 위치 변화로 추정하던 기존 방식(노이즈 + 지연 있음)은 이제 여기서 안 씀 -
        # 그동안 겪었던 원운동/역주행 문제들의 근본 원인이 바로 이 추정 오차였음.
        heading_vec = real_heading_vec
        last_valid_heading = real_heading_vec
    else:
        # body.x가 없는 경우를 대비한 예전 방식(위치 변화 기반 추정) - 안전망으로 유지
        if last_position is not None:
            raw_dx = pos_x - last_position[0]
            raw_dz = pos_z - last_position[1]
            moved_dist = math.hypot(raw_dx, raw_dz)
            if moved_dist >= MIN_MOVE_FOR_HEADING:
                heading_history.append((raw_dx, raw_dz))
                if len(heading_history) > HEADING_HISTORY_SIZE:
                    heading_history.pop(0)
                avg_dx = sum(v[0] for v in heading_history) / len(heading_history)
                avg_dz = sum(v[1] for v in heading_history) / len(heading_history)
                last_valid_heading = (avg_dx, avg_dz)
        heading_vec = last_valid_heading
    last_position = current_pos

    move_ws, move_ad, dist = compute_move_command(
        current_pos, heading_vec, target_waypoint,
        has_real_heading=(real_heading_vec is not None),
    )
    command = _build_command(move_ws, move_ad, fire=False)

    # 디버그: 실제로 어떤 각도 계산이 나오고 있는지 확인용 로그
    dbg_target_angle = _angle_deg((target_waypoint[0] - pos_x, target_waypoint[1] - pos_z))
    dbg_heading_angle = _angle_deg(heading_vec) if heading_vec else None
    print(f"🧭 heading_vec={heading_vec}, target_angle={dbg_target_angle:.1f}, "
          f"heading_angle={dbg_heading_angle}, waypoint={target_waypoint}, dist_to_wp={dist:.2f}, "
          f"steering_mode={steering_mode}, turning_frame_count={turning_frame_count}, "
          f"forced_straight_countdown={forced_straight_countdown}")

    print(f"🔁 Sent Move Command: {command} (waypoint {path_index}/{len(current_path)})")
>>>>>>> 263804ccd93f6321e12b19a5ac16b427cc88a4ed
    return jsonify(command)


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


<<<<<<< HEAD
    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")

        global dest
        dest = (x, y, z) # 목적지 저장

        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400


def update_obstacles_from_payload(payload: dict):
    """
    오브젝트 정보 planner class에 입력
    """
    obs_list = []
    for item in payload.get("obstacles", []):
        obs = ObstacleRect.from_min_max(
            x_min=item["x_min"],
            x_max=item["x_max"],
            z_min=item["z_min"],
            z_max=item["z_max"],
        )
        obs_list.append(obs)
    planner.set_obstacles(obs_list)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    update_obstacles_from_payload(data)

    # 최초 탐색이 아닌 경우
    if path_flag:
        print('경로 재탐색')
        global path, idx
        idx = 0
        path = planner.find_path((all_info['playerPos']['x'], all_info['playerPos']['z']), (dest[0], dest[2]))
        planner.plot(path, show_grid=True, title="A* Demo (300x300)", fname='path')

    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
=======
@app.route('/collision', methods=['POST'])
>>>>>>> 263804ccd93f6321e12b19a5ac16b427cc88a4ed
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})


@app.route('/init', methods=['GET'])
def init():
<<<<<<< HEAD
    start_point = (60, 10, 27.23)
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": start_point[0],  #Blue Start Position
        "blStartY": start_point[1],
        "blStartZ": start_point[2],
        "rdStartX": 59, #Red Start Position
=======
    # Unity에서 재시작(에피소드 초기화) 시 이 엔드포인트가 다시 호출되는데,
    # 서버 쪽에 남아있는 경로 계획 상태를 여기서 같이 초기화하지 않으면
    # 새 에피소드에서도 이전 목적지/경로/조향 상태를 그대로 들고 있어서
    # "재시작이 제대로 안 먹는 것처럼" 보이는 원인이 됨.
    reset_planning_state()

    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59,
>>>>>>> 263804ccd93f6321e12b19a5ac16b427cc88a4ed
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detectMode": False,
        "logMode": True,   # False였으면 Unity가 /info(회전각 포함)를 아예 안 보낼 수 있어서 켬
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": True,   # logMode만으론 안 됐을 수 있어서 같이 켜서 테스트
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit": True
    }
<<<<<<< HEAD

    global path, idx, path_flag
    planner.reset_planner() # 처음부터 path finding을 위한 초기화
    path = planner.find_path((start_point[0], start_point[2]), (dest[0], dest[2]))
    planner.plot(path, show_grid=True, title="A* Demo (300x300)", fname='path')

    idx = 0 # path point idx 초기화
    path_flag = True 

=======
    print("🛠️ Initialization config sent via /init (server state reset):", config)
>>>>>>> 263804ccd93f6321e12b19a5ac16b427cc88a4ed
    return jsonify(config)


@app.route('/start', methods=['GET'])
def start():
    # 실제로 재시작 버튼이 호출하는 게 /init이 아니라 /start인 것으로 확인됨.
    # 여기서도 서버 쪽 경로 계획 상태를 초기화해야 재시작이 실제로 반영됨.
    reset_planning_state()
    print("🚀 /start command received (server state reset)")
    return jsonify({"control": ""})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)