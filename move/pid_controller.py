"""
pid_controller.py
=================

Tank Challenge용 주행 제어 모듈.


1. 속도 PID 제어
2. 조향 PID 제어
3. /info 데이터에서 현재 속도 계산
4. 목적지까지 남은 거리에 따른 목표 속도 계산
5. D* Lite 경로의 Look-ahead point 선택
6. 경로의 앞쪽 코너 탐색
7. 코너 진입 속도 제한
8. 차체 방향이 경로와 어긋난 경우 전진 출력 제한
9. 서버가 반환할 moveWS / moveAD 명령 생성
10. /get_action 제어 주기(dt) 관리
11. 목적지 변경 감지 및 PID 상태 초기화
12. 목적지 도착 latch 및 최종 제동 상태 관리

"""

# 거리, 각도, 속도 계산에 sqrt, hypot, atan2, radians 등이 필요하다.
import math

# /info 위치 변화량으로 속도를 계산할 때 실제 시간 간격 dt가 필요하다.
import time


# ============================================================
# 1. 직선 주행용 속도 PID + 목적지 정지 제어 파라미터
# ============================================================

# 전차가 직선에서 목표로 하는 최고 속도 [km/h].
# 속도 PID가 아무리 큰 가속 명령을 만들어도 목표속도는 기본적으로 이 값을 넘지 않는다.
MAX_SPEED_KMH = 60.0

# 물리식 v^2 = 2ad에서 km/h 대신 m/s를 사용하기 위해 변환한 최고속도.
# 60 km/h / 3.6 = 약 16.67 m/s.
MAX_SPEED_MPS = MAX_SPEED_KMH / 3.6

# 계획상 사용할 평균 감속도 [m/s^2].
# 실제 S 입력 weight가 아니라 "이 정도 감속할 수 있다"고 제어기가 가정하는 값이다.
# 값이 작을수록 필요한 제동거리가 길어져 더 일찍 감속한다.
# 값이 클수록 늦게 감속한다.
PLANNED_BRAKE_DECEL_MPS2 = 2.0

# 목적지/제동 계산에 추가하는 안전 여유거리 [m].
# 계산된 제동거리보다 이 거리만큼 더 일찍 감속하도록 하기 위한 값이다.
BRAKE_MARGIN_M = 5.0

# 목적지에서 이 거리 이내로 들어오면 도착 영역으로 판단하는 기준 [m].
STOP_DISTANCE_M = 1.0

# 실제 속도가 이 값 이하이면 사실상 정지 상태로 볼 수 있는 기준 [km/h].
STOP_SPEED_KMH = 1.0

# 최고속도 근처에서 과도한 가속을 방지하기 위한 속도 기준 [km/h].
# 서버의 longitudinal 제어 로직에서 사용한다.
SPEED_LIMIT_START_KMH = 58.0

# 목적지 최종 제동 시 S 명령을 유지할 최대 시간 [sec].
# 시뮬레이터에서 S는 브레이크이면서 동시에 후진 명령이므로
# 너무 오래 보내면 정지 후 후진할 수 있어 시간을 제한한다.
FINAL_BRAKE_DURATION_SEC = 0.8

# 목적지 최종 제동 시 사용할 S 명령의 weight.
# 0~1 범위이며 값이 클수록 강한 제동 입력이다.
FINAL_BRAKE_WEIGHT = 0.60

# 속도 PID의 P gain.
# 현재 목표속도와 실제속도의 오차에 즉각 비례해서 반응한다.
SPEED_KP = 0.12

# 속도 PID의 I gain.
# 오랫동안 남아 있는 속도 오차를 누적하여 보상한다.
SPEED_KI = 0.004

# 속도 PID의 D gain.
# 속도 오차가 얼마나 빠르게 변하는지 보고 급격한 변화/오버슈트를 완화한다.
SPEED_KD = 0.02


# ============================================================
# 2. D* Lite 경로 추종용 조향 제어 파라미터
# ============================================================

# 현재 위치에서 D* Lite 경로를 따라 기본적으로 몇 m 앞을 바라볼지 결정한다.
LOOKAHEAD_BASE_M = 3.0

# 현재 속도[km/h]가 증가할 때 Look-ahead 거리를 얼마나 증가시킬지 결정한다.
# 실제 계산식:
# lookahead = LOOKAHEAD_BASE_M + LOOKAHEAD_SPEED_GAIN * current_speed_kmh
LOOKAHEAD_SPEED_GAIN = 0.08

# Look-ahead가 너무 짧아져 경로의 바로 앞 점만 추종하는 것을 막는 최소값 [m].
LOOKAHEAD_MIN_M = 3.0

# Look-ahead가 너무 길어져 코너 안쪽을 크게 잘라먹는 것을 막는 최대값 [m].
LOOKAHEAD_MAX_M = 7.0

# 조향 PID의 P gain.
# 목표 heading과 현재 차체 yaw의 각도 오차에 비례해 A/D 조향량을 결정한다.
STEER_KP = 0.025

# 조향 PID의 I gain.
# 현재는 0이므로 조향에서는 적분항을 사실상 사용하지 않는다.
STEER_KI = 0.0

# 조향 PID의 D gain.
# 방향 오차가 빠르게 변할 때 조향 출력을 완화하여 흔들림을 줄인다.
STEER_KD = 0.003

# 방향 오차가 이 각도 이하이면 조향하지 않는다 [deg].
# 작은 노이즈 때문에 A/D가 계속 번갈아 입력되는 것을 방지한다.
STEER_DEADBAND_DEG = 1.5

# A/D 조향 weight의 최대값.
# PID 출력이 너무 커져도 조향 명령은 최대 0.85까지만 사용한다.
STEER_MAX_WEIGHT = 0.85


# ============================================================
# 3. 다가오는 코너 선행 감속 파라미터
# ============================================================

# 현재 위치에서 경로 앞쪽 몇 m까지 코너를 탐색할지 결정한다.
CORNER_PREVIEW_DISTANCE_M = 50.0

# 경로 방향 변화가 이 각도 이상일 때만 의미 있는 코너로 판단한다 [deg].
# D* Lite 경로의 작은 꺾임/격자 노이즈를 모두 코너로 처리하지 않기 위한 기준이다.
CORNER_ANGLE_THRESHOLD_DEG = 20.0

# 코너 각도를 계산할 때 후보점 바로 앞/뒤 한 칸이 아니라
# 약 몇 m 떨어진 방향을 비교할지 결정한다.
# D* Lite의 계단식 경로로 인한 각도 노이즈를 줄인다.
CORNER_DIRECTION_SAMPLE_M = 3.0

# 이 각도 이상이면 급격한 코너(Sharp)로 분류한다.
SHARP_CORNER_ANGLE_DEG = 70.0

# 이 각도 이상이고 Sharp 미만이면 중간 코너(Medium)로 분류한다.
MEDIUM_CORNER_ANGLE_DEG = 40.0

# 코너까지 8m 이내이면 Near 영역으로 본다.
CORNER_NEAR_DISTANCE_M = 8.0

# 코너까지 18m 이내이면 Mid 영역으로 본다.
CORNER_MID_DISTANCE_M = 18.0

# 코너까지 35m 이내이면 Far 영역으로 본다.
CORNER_FAR_DISTANCE_M = 35.0

# Sharp 코너가 Near 영역에 있을 때 허용할 최대 목표속도 [km/h].
SHARP_CORNER_SPEED_NEAR_KMH = 8.0

# Sharp 코너가 Mid 영역에 있을 때 허용할 최대 목표속도 [km/h].
SHARP_CORNER_SPEED_MID_KMH = 18.0

# Sharp 코너가 Far 영역에 있을 때 허용할 최대 목표속도 [km/h].
SHARP_CORNER_SPEED_FAR_KMH = 35.0

# Medium 코너가 Near 영역에 있을 때 허용할 최대 목표속도 [km/h].
MEDIUM_CORNER_SPEED_NEAR_KMH = 12.0

# Medium 코너가 Mid 영역에 있을 때 허용할 최대 목표속도 [km/h].
MEDIUM_CORNER_SPEED_MID_KMH = 23.0

# Medium 코너가 Far 영역에 있을 때 허용할 최대 목표속도 [km/h].
MEDIUM_CORNER_SPEED_FAR_KMH = 40.0

# 20~40도 수준의 완만한 코너가 매우 가까울 때 적용할 제한속도 [km/h].
GENTLE_CORNER_SPEED_NEAR_KMH = 30.0


# ============================================================
# 4. 공통 보조 함수
# ============================================================

def clamp(value, minimum, maximum):
    """
    value가 지정 범위를 벗어나지 않도록 제한한다.

    예:
        clamp(1.4, 0.0, 1.0) -> 1.0
        clamp(-0.2, 0.0, 1.0) -> 0.0
    """

    # min(value, maximum)으로 상한을 제한한 뒤,
    # max(minimum, ...)으로 하한도 제한한다.
    return max(minimum, min(maximum, value))


# ============================================================
# 5. PID Controller
# ============================================================

class PIDController:
    """
    P + I + D 출력을 계산하는 범용 PID Controller.

    속도 제어와 조향 제어에서 같은 클래스를 재사용한다.
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        output_min=-1.0,
        output_max=1.0,
        integral_min=-10.0,
        integral_max=10.0,
    ):
        # 비례항 gain.
        self.kp = float(kp)

        # 적분항 gain.
        self.ki = float(ki)

        # 미분항 gain.
        self.kd = float(kd)

        # PID 최종 출력의 최소값.
        self.output_min = float(output_min)

        # PID 최종 출력의 최대값.
        self.output_max = float(output_max)

        # 적분 누적값의 최소 제한.
        # integral wind-up을 막기 위해 사용한다.
        self.integral_min = float(integral_min)

        # 적분 누적값의 최대 제한.
        self.integral_max = float(integral_max)

        # 현재까지 누적된 error * dt.
        self.integral = 0.0

        # 이전 제어 주기의 error.
        # derivative = (현재오차 - 이전오차) / dt 계산에 필요하다.
        self.previous_error = None

    def reset(self):
        """
        새로운 목적지를 설정하거나 제어를 초기화할 때 PID 내부 상태를 비운다.
        """

        # 누적된 I항을 제거한다.
        self.integral = 0.0

        # 이전 오차를 제거하여 다음 update의 D항을 0부터 시작하게 한다.
        self.previous_error = None

    def update(self, error, dt):
        """
        현재 error와 제어 주기 dt를 받아 PID 출력을 계산한다.
        """

        # dt가 0에 가까우면 derivative가 폭발하므로 최소 0.001초를 보장한다.
        dt = max(float(dt), 1e-3)

        # I항용 누적 오차.
        # 너무 크게 누적되지 않도록 integral_min~integral_max 범위로 제한한다.
        self.integral = clamp(
            self.integral + error * dt,
            self.integral_min,
            self.integral_max,
        )

        # 첫 호출에서는 previous_error가 없으므로 D항을 0으로 한다.
        # 이후부터는 오차 변화율을 계산한다.
        derivative = (
            0.0
            if self.previous_error is None
            else (error - self.previous_error) / dt
        )

        # PID 기본식:
        # output = Kp*e + Ki*∫e dt + Kd*de/dt
        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        # 다음 호출에서 derivative 계산에 사용하기 위해 현재 error를 저장한다.
        self.previous_error = error

        # 서버에 보낼 weight 범위를 넘어가지 않도록 최종 PID 출력을 제한한다.
        return clamp(
            output,
            self.output_min,
            self.output_max,
        )


# 속도용 PID 인스턴스.
# 출력 기본 범위는 -1~1이며,
# +출력은 W, -출력은 S로 변환한다.
speed_pid = PIDController(
    SPEED_KP,
    SPEED_KI,
    SPEED_KD,
)

# 조향용 PID 인스턴스.
# 좌/우 회전이 모두 필요하므로 출력은 -STEER_MAX_WEIGHT ~ +STEER_MAX_WEIGHT.
steering_pid = PIDController(
    STEER_KP,
    STEER_KI,
    STEER_KD,
    output_min=-STEER_MAX_WEIGHT,
    output_max=STEER_MAX_WEIGHT,

    # 조향 I항은 현재 ki=0이지만,
    # 나중에 I항을 사용할 경우를 대비해 누적범위를 별도로 설정한다.
    integral_min=-30.0,
    integral_max=30.0,
)


# ============================================================
# 6. /info 기반 속도 상태
# ============================================================

# 현재 필터링된 최신 속도 [km/h].
info_speed_kmh = None

# 이전 /info 호출에서 받은 전차 위치 [x, z].
# explicit speed가 없을 경우 위치 변화량으로 속도를 계산하는 데 사용한다.
info_previous_position = None

# 이전 /info를 처리한 시간.
info_previous_time = None

# 속도 EMA(Exponential Moving Average) 필터 계수.
# 0.35이면 새 측정값 35%, 이전 필터값 65%를 섞는다.
INFO_SPEED_EMA_ALPHA = 0.35


# ============================================================
# 6-1. /get_action 주행 제어 상태
# ============================================================

# 직전 /get_action 제어가 실행된 시각 [sec].
#
# time.monotonic() 값을 저장한다.
# 다음 제어 주기에서
#
#     dt = now - last_control_time
#
# 을 계산하기 위해 필요하다.
#
# PID의 I항은 error * dt,
# D항은 (error - previous_error) / dt 를 사용하므로
# 실제 제어 호출 간격을 기억해야 한다.
#
# None:
#   아직 첫 제어가 실행되지 않았거나
#   목적지 변경/초기화 때문에 시간 기준을 새로 잡아야 하는 상태.
last_control_time = None


# 목적지 도착 상태를 한 번 확정했는지 저장하는 latch 변수.
#
# False:
#   아직 목적지 도착으로 판단하지 않은 상태.
#
# True:
#   STOP_DISTANCE_M 안에 한 번 들어온 상태.
#
# 한 번 True가 되면 전차가 관성으로 목적지를 조금 지나가더라도
# 다시 일반 PID 주행으로 돌아가 W가 입력되는 것을 막는다.
arrival_latched = False


# 마지막으로 PID가 추종하던 목적지의 (x, z) 좌표.
#
# 예:
#   (150.0, 280.0)
#
# 현재 목적지와 이 값이 다르면 새로운 목적지로 판단하고
# speed_pid, steering_pid, 제어시간, 도착상태, 최종제동 상태를 초기화한다.
#
# None:
#   아직 PID가 어떤 목적지도 추종하지 않았거나
#   전체 제어 상태가 초기화된 상태.
last_pid_destination = None


# 목적지에 도착한 뒤 최종 S 제동을 처음 시작한 시각 [sec].
#
# 시뮬레이터에서 S는 단순 브레이크가 아니라 후진 명령도 겸하기 때문에
# 계속 S를 보내면 정지 후 뒤로 움직일 수 있다.
#
# 따라서 최초 제동 시각을 저장하고
# FINAL_BRAKE_DURATION_SEC 동안만 S를 허용한다.
#
# None:
#   아직 최종 제동이 시작되지 않은 상태.
final_brake_start_time = None


def extract_speed_from_info(data):
    """
    /info JSON에서 시뮬레이터가 직접 제공하는 속도값을 찾는다.

    반환:
        (속도[km/h], 사용한 key)

    현재 코드에서는 시뮬레이터 속도값을 m/s라고 가정하고 *3.6 한다.
    """

    # 시뮬레이터 버전/키 이름 차이에 대응하기 위해 여러 후보 key를 순서대로 검사한다.
    keys = (
        "PlayerSpeed",
        "playerSpeed",
        "speed",
        "velocity",
    )

    # 후보 key를 하나씩 확인한다.
    for key in keys:

        # 해당 key가 존재하고 값도 None이 아닌 경우에만 처리한다.
        if key in data and data[key] is not None:
            try:
                # abs(): 후진 속도가 음수로 전달되더라도 속력 크기만 사용.
                # float(): JSON 숫자/문자열을 실수형으로 통일.
                # *3.6: m/s -> km/h 변환.
                return abs(float(data[key])) * 3.6, key

            # 숫자로 변환할 수 없는 값이면 다음 key를 검사한다.
            except (TypeError, ValueError):
                pass

    # 사용할 수 있는 속도값을 찾지 못한 경우.
    return None, None


def update_info_speed(data, player_position):
    """
    /info가 들어올 때 현재 속도 상태를 갱신한다.

    우선순위:
    1. /info에 명시적 속도 필드가 있으면 그것을 사용.
    2. 없으면 현재 위치와 이전 위치의 차이 / dt로 속도를 추정.

    player_position은 [x, z] 형태를 기대한다.
    """

    # 이 함수 호출 사이에도 상태를 유지해야 하므로 module global 값을 사용한다.
    global info_speed_kmh
    global info_previous_position
    global info_previous_time

    # 현재 시간을 monotonic clock으로 얻는다.
    # 시스템 시간이 변경되어도 시간차 계산이 안정적이다.
    now = time.monotonic()

    # /info에서 직접 속도 필드를 먼저 찾는다.
    explicit_speed_kmh, speed_key = extract_speed_from_info(
        data
    )

    # 직접 속도가 있으면 우선 사용한다.
    measured_speed_kmh = explicit_speed_kmh

    # 로그에서 속도가 어느 데이터로 계산됐는지 확인하기 위한 source.
    speed_source = speed_key

    # 직접 속도값이 없고,
    # 이전 위치와 이전 시간이 모두 존재할 때만 위치 기반 속도를 계산한다.
    if (
        measured_speed_kmh is None
        and info_previous_position is not None
        and info_previous_time is not None
    ):
        # 두 /info 처리 시점 사이의 시간차 [sec].
        dt = now - info_previous_time

        # 너무 짧은 dt는 노이즈가 커지고,
        # 너무 긴 dt는 실제 순간속도를 잘 표현하지 못하므로 범위를 제한한다.
        if 0.01 <= dt <= 1.0:

            # X축 이동량.
            dx = (
                float(player_position[0])
                - float(info_previous_position[0])
            )

            # Z축 이동량.
            dz = (
                float(player_position[1])
                - float(info_previous_position[1])
            )

            # 평면 이동거리 / 시간 = m/s,
            # 여기에 *3.6하여 km/h로 변환한다.
            measured_speed_kmh = (
                math.hypot(dx, dz)
                / dt
                * 3.6
            )

            # 디버깅 로그에서 위치 기반 계산임을 알 수 있게 표시한다.
            speed_source = "playerPos/dt"

    # 다음 호출의 속도 계산을 위해 현재 위치를 저장한다.
    info_previous_position = [
        float(player_position[0]),
        float(player_position[1]),
    ]

    # 다음 호출의 dt 계산을 위해 현재 시간을 저장한다.
    info_previous_time = now

    # 이번 호출에서 사용할 수 있는 속도값을 얻은 경우만 필터를 갱신한다.
    if measured_speed_kmh is not None:

        # 첫 측정이라 이전 EMA 값이 없으면 측정값을 그대로 초기값으로 사용한다.
        if info_speed_kmh is None:
            info_speed_kmh = measured_speed_kmh

        else:
            # EMA:
            # new_filtered
            # = alpha * new_measurement
            # + (1-alpha) * previous_filtered
            #
            # 순간적인 통신/측정 노이즈를 완화한다.
            info_speed_kmh = (
                INFO_SPEED_EMA_ALPHA
                * measured_speed_kmh
                + (
                    1.0
                    - INFO_SPEED_EMA_ALPHA
                )
                * info_speed_kmh
            )

    # 현재 필터링된 속도와 속도 출처를 함께 반환한다.
    return info_speed_kmh, speed_source


def read_player_speed_kmh():
    """
    다른 서버 코드가 현재 PID 모듈에 저장된 최신 속도를 읽을 때 사용한다.
    """

    return info_speed_kmh


# ============================================================
# 6-2. /get_action 제어 상태 관리 함수
# ============================================================

def reset_control_state(reset_destination=True):
    """
    PID와 /get_action에서 사용하는 주행 제어 상태를 초기화한다.

    Parameters
    ----------
    reset_destination : bool
        True:
            last_pid_destination까지 None으로 초기화한다.
            /init, episode 재시작, 전체 주행 초기화에서 사용한다.

        False:
            현재 목적지 정보는 유지하고
            PID/시간/도착/최종제동 상태만 초기화한다.

    초기화되는 항목
    ----------------
    speed_pid
        속도 PID의 integral, previous_error 초기화.

    steering_pid
        조향 PID의 integral, previous_error 초기화.

    last_control_time
        다음 get_control_dt() 호출을 첫 제어 주기로 만든다.

    arrival_latched
        이전 목적지의 도착 판정을 제거한다.

    final_brake_start_time
        이전 목적지의 최종 제동 timer를 제거한다.

    last_pid_destination
        reset_destination=True일 때만 제거한다.

    왜 필요한가
    -----------
    이전 목적지나 이전 episode에서 남은 PID/제동 상태가
    새로운 주행에 섞이는 것을 막기 위해 사용한다.
    """
    global last_control_time
    global arrival_latched
    global last_pid_destination
    global final_brake_start_time

    # 속도 PID 내부 누적 상태 초기화.
    speed_pid.reset()

    # 조향 PID 내부 누적 상태 초기화.
    steering_pid.reset()

    # 다음 제어에서 dt를 새로 시작하도록 시간 기준 제거.
    last_control_time = None

    # 이전 목적지의 도착 판정 제거.
    arrival_latched = False

    # 이전 목적지의 최종 S 제동 timer 제거.
    final_brake_start_time = None

    # 전체 초기화일 때만 마지막 목적지 기억도 제거.
    if reset_destination:
        last_pid_destination = None


def get_control_dt():
    """
    현재 /get_action 제어 주기의 dt를 계산한다.

    Returns
    -------
    now : float
        time.monotonic()으로 얻은 현재 제어 시각 [sec].

    dt : float
        직전 제어와 현재 제어 사이의 시간 간격 [sec].

    계산 방식
    ---------
    첫 호출:
        last_control_time이 None이므로 0.05 sec를 사용한다.

    이후 호출:
        now - last_control_time을 실제 dt로 사용한다.

    최종적으로 0.01 ~ 0.25 sec 범위로 제한한다.

    왜 필요한가
    -----------
    PID의 적분항과 미분항은 시간 간격 dt가 필요하다.

        I = integral + error * dt
        D = (error - previous_error) / dt

    통신 지연 때문에 dt가 지나치게 커지거나,
    호출 간격이 너무 짧아 dt가 거의 0이 되는 경우
    PID 출력이 불안정해질 수 있으므로 범위를 제한한다.
    """
    global last_control_time

    # 현재 제어 시각.
    now = time.monotonic()

    # 첫 호출은 안정적인 기본 주기 0.05 sec 사용.
    if last_control_time is None:
        dt = 0.05

    else:
        # 실제 제어 호출 간격을 구하고 비정상적인 값을 제한한다.
        dt = clamp(
            now - last_control_time,
            0.01,
            0.25,
        )

    # 다음 제어 주기에서 사용할 현재 시각 저장.
    last_control_time = now

    return now, dt


def check_destination_change(destination):
    """
    현재 목적지가 이전 PID 제어 목적지와 달라졌는지 확인한다.

    Parameters
    ----------
    destination : sequence | None
        현재 목적지의 [x, z] 또는 (x, z).

        destination[0]:
            목적지 x 좌표.

        destination[1]:
            목적지 z 좌표.

    Returns
    -------
    bool
        True:
            목적지가 변경되어 PID/제어 상태를 초기화한 경우.

        False:
            이전과 같은 목적지를 계속 추종 중인 경우.

    동작
    ----
    현재 목적지를 소수점 셋째 자리까지 반올림하여
    last_pid_destination과 비교한다.

    목적지가 변경되면:
        1. speed_pid.reset()
        2. steering_pid.reset()
        3. last_control_time = None
        4. arrival_latched = False
        5. final_brake_start_time = None
        6. last_pid_destination = 새 목적지

    왜 필요한가
    -----------
    이전 목적지를 따라가면서 누적된 PID의 I/D 상태와
    도착/제동 상태를 새 목적지에 그대로 사용하면
    첫 제어 출력이 튀거나 출발하지 못할 수 있기 때문이다.
    """
    global last_control_time
    global arrival_latched
    global last_pid_destination
    global final_brake_start_time

    # 목적지가 없으면 비교할 수 없으므로
    # 마지막 목적지 상태를 제거하고 제어 상태를 초기화한다.
    if destination is None:
        reset_control_state(
            reset_destination=True
        )
        return False

    # 미세한 실수 오차 때문에 같은 목적지를 다른 값으로
    # 판단하지 않도록 소수점 셋째 자리까지 반올림한다.
    destination_signature = (
        round(float(destination[0]), 3),
        round(float(destination[1]), 3),
    )

    # 이전 목적지와 같으면 아무것도 초기화하지 않는다.
    if destination_signature == last_pid_destination:
        return False

    # 새로운 목적지이면 PID 내부 상태 초기화.
    speed_pid.reset()
    steering_pid.reset()

    # 새로운 목적지 기준으로 제어 시간과 도착/제동 상태 초기화.
    last_control_time = None
    arrival_latched = False
    final_brake_start_time = None

    # 앞으로 비교할 수 있도록 새 목적지 저장.
    last_pid_destination = destination_signature

    return True


def update_arrival_state(
    distance_to_goal,
    current_speed_kmh,
    now=None,
):
    """
    목적지 도착 latch와 최종 S 제동 상태를 처리한다.

    Parameters
    ----------
    distance_to_goal : float
        현재 전차 위치에서 최종 목적지까지의 거리 [m].

    current_speed_kmh : float
        현재 전차 속도 [km/h].

    now : float | None
        현재 제어 시각.
        일반적으로 get_control_dt()에서 반환된 now를 그대로 넘긴다.
        None이면 함수 내부에서 time.monotonic()을 호출한다.

    Returns
    -------
    None
        아직 목적지 도착 상태가 아니다.
        서버는 일반 PID 주행을 계속하면 된다.

    dict
        목적지 도착 상태일 때 사용할 command.

        속도가 남아 있고 최종 제동 허용시간 안:
            S 제동 command 반환.

        충분히 느려졌거나 제동시간 종료:
            모든 입력을 해제한 stop command 반환.

    동작 순서
    ---------
    1. distance_to_goal <= STOP_DISTANCE_M 이면 도착 판정.
    2. arrival_latched = True로 고정.
    3. 최초 도착 시 final_brake_start_time 저장.
    4. FINAL_BRAKE_DURATION_SEC 동안만 S 제동.
    5. 이후에는 S를 끄고 정지 command 반환.

    왜 필요한가
    -----------
    전차가 목적지를 살짝 지나쳤다고 다시 W를 주는 것을 막고,
    S를 너무 오래 보내 정지 후 후진하는 것도 막기 위해 필요하다.
    """
    global arrival_latched
    global final_brake_start_time

    # 제어 시각이 외부에서 전달되지 않았으면 현재 시각 사용.
    if now is None:
        now = time.monotonic()

    distance_to_goal = float(distance_to_goal)
    current_speed_kmh = float(current_speed_kmh)

    # 아직 도착한 적이 없고 목적지 반경 밖이면 일반 주행 계속.
    if (
        not arrival_latched
        and distance_to_goal > STOP_DISTANCE_M
    ):
        return None

    # 목적지 반경에 한 번 들어오면 도착 상태를 고정한다.
    arrival_latched = True

    # 최초 도착 순간에만 최종 제동 시작 시각 기록.
    if final_brake_start_time is None:
        final_brake_start_time = now

    # 최종 제동을 시작한 뒤 지난 시간.
    final_brake_elapsed = (
        now - final_brake_start_time
    )

    # 아직 충분히 느려지지 않았고,
    # 허용된 최종 제동 시간 안이면 S 제동.
    if (
        current_speed_kmh > STOP_SPEED_KMH
        and final_brake_elapsed < FINAL_BRAKE_DURATION_SEC
    ):
        return make_longitudinal_command(
            -FINAL_BRAKE_WEIGHT
        )

    # 충분히 느려졌거나 제동 제한시간이 지나면
    # S를 더 보내지 않고 모든 입력을 해제한다.
    return make_stop_command()


# ============================================================
# 7. 목적지까지 남은 거리에 따른 목표 속도
# ============================================================

def calculate_target_speed_kmh(distance_m):
    """
    목적지까지 남은 거리에서 정지할 수 있는 최대 허용속도를 계산한다.

    사용식:
        v^2 = 2*a*d
        v = sqrt(2*a*d)

    멀리 있으면 MAX_SPEED_KMH까지 허용하고,
    목적지에 가까워질수록 목표속도를 연속적으로 낮춘다.
    """

    # STOP_DISTANCE_M 안쪽은 이미 도착 영역으로 취급하기 때문에
    # 실제 감속 계산에 사용할 수 있는 거리에서 빼준다.
    usable_distance = max(
        0.0,
        float(distance_m) - STOP_DISTANCE_M,
    )

    # 현재 남은 거리에서 정지하기 위해 허용 가능한 속도[m/s].
    braking_speed_mps = math.sqrt(
        2.0
        * PLANNED_BRAKE_DECEL_MPS2
        * usable_distance
    )

    # 물리식이 매우 큰 속도를 허용하더라도 최고속도는 넘지 않게 한다.
    target_speed_mps = min(
        MAX_SPEED_MPS,
        braking_speed_mps,
    )

    # 서버의 속도 제어 로직이 km/h 기준이므로 다시 km/h로 변환한다.
    return target_speed_mps * 3.6


# ============================================================
# 8. 차체 방향 / 각도 계산
# ============================================================

def normalize_angle_deg(angle):
    """
    임의의 각도를 -180~+180 deg로 변환한다.

    예:
        350 deg -> -10 deg
        190 deg -> -170 deg

    이렇게 해야 왼쪽/오른쪽 중 어느 방향으로 얼마나 돌아야 하는지
    가장 짧은 회전각을 계산할 수 있다.
    """

    return (
        float(angle) + 180.0
    ) % 360.0 - 180.0


def read_player_body_yaw_deg(info):
    """
    서버 전역변수 latest_info에 직접 의존하지 않도록
    /info JSON을 인자로 받아 차체 yaw를 추출한다.

    반환값:
        0~360 deg
    """

    # 현재 시뮬레이터에서 주로 사용하는 key.
    value = info.get("playerBodyX")

    # 대소문자가 다른 버전에도 대응한다.
    if value is None:
        value = info.get("PlayerBodyX")

    # 둘 다 없으면 yaw를 계산할 수 없다.
    if value is None:
        return None

    try:
        # 360으로 나눈 나머지를 사용해 0~360 범위로 정규화한다.
        return float(value) % 360.0

    # 숫자로 변환할 수 없는 값이면 사용할 수 없다고 판단한다.
    except (TypeError, ValueError):
        return None


# ============================================================
# 9. D* Lite 경로에서 Look-ahead point 선택
# ============================================================

def select_lookahead_point(
    path,
    current_position,
    lookahead_distance,
):
    """
    현재 위치에서 가장 가까운 D* Lite path point를 찾고,
    그 지점부터 경로를 따라 lookahead_distance만큼 앞의 점을 반환한다.

    반환:
        target_point, target_index
    """

    # 경로가 없으면 목표점도 선택할 수 없다.
    if not path:
        return None, None

    # 현재 위치를 실수형 x, z로 분리한다.
    cx, cz = map(
        float,
        current_position,
    )

    # 경로 전체 point 중 현재 위치와 유클리드 거리가 가장 가까운 index를 찾는다.
    nearest_index = min(
        range(len(path)),
        key=lambda i: math.hypot(
            float(path[i][0]) - cx,
            float(path[i][1]) - cz,
        ),
    )

    # 아직 앞으로 이동하지 않았으므로 target은 nearest point에서 시작한다.
    target_index = nearest_index

    # nearest_index 이후 경로 길이를 누적하기 위한 변수.
    accumulated = 0.0

    # 현재 가장 가까운 경로점부터 path 끝까지 앞쪽으로 탐색한다.
    for i in range(
        nearest_index,
        len(path) - 1,
    ):
        # 현재 path point.
        x1, z1 = map(
            float,
            path[i],
        )

        # 바로 다음 path point.
        x2, z2 = map(
            float,
            path[i + 1],
        )

        # 두 점 사이 실제 길이를 누적한다.
        accumulated += math.hypot(
            x2 - x1,
            z2 - z1,
        )

        # 현재까지 도달한 가장 앞쪽 index.
        target_index = i + 1

        # 누적 경로 길이가 목표 Look-ahead 거리 이상이면 탐색 종료.
        if accumulated >= lookahead_distance:
            break

    # 선택한 path point와 index를 함께 반환한다.
    return (
        path[target_index],
        target_index,
    )


def _path_distance_between(
    path,
    start_index,
    end_index,
):
    """
    path의 start_index부터 end_index까지 실제 경로 길이를 계산한다.

    함수명 앞의 '_'는 외부 공개 API보다는
    모듈 내부 보조 함수라는 의미로 사용한 것이다.
    """

    # 경로가 없거나 구간이 역방향/0길이면 거리 0.
    if (
        not path
        or end_index <= start_index
    ):
        return 0.0

    # 누적 경로 거리.
    total = 0.0

    # end_index가 path 범위를 넘어가는 것을 방지한다.
    end_index = min(
        end_index,
        len(path) - 1,
    )

    # start -> end 사이의 각 path segment를 순회한다.
    for i in range(
        start_index,
        end_index,
    ):
        x1, z1 = map(
            float,
            path[i],
        )

        x2, z2 = map(
            float,
            path[i + 1],
        )

        # 각 segment 길이를 누적한다.
        total += math.hypot(
            x2 - x1,
            z2 - z1,
        )

    return total


# ============================================================
# 10. 앞으로 다가올 코너 탐색
# ============================================================

def find_upcoming_corner(
    path,
    current_position,
    preview_distance=CORNER_PREVIEW_DISTANCE_M,
    corner_angle_threshold=CORNER_ANGLE_THRESHOLD_DEG,
    direction_sample_distance=CORNER_DIRECTION_SAMPLE_M,
):
    """
    현재 위치 앞쪽 preview_distance 범위에서
    첫 번째 의미 있는 코너를 찾는다.

    반환 예:
        {
            "distance": 20.5,
            "angle": 78.0,
            "index": 120,
            "point": (100, 150),
        }
    """

    # 코너를 정의하려면 최소 3개 이상의 path point가 필요하다.
    if (
        path is None
        or len(path) < 3
    ):
        return None

    # 현재 위치.
    px, pz = map(
        float,
        current_position,
    )

    # 현재 전차 위치에 가장 가까운 path index를 구한다.
    nearest_index = min(
        range(len(path)),
        key=lambda i: math.hypot(
            float(path[i][0]) - px,
            float(path[i][1]) - pz,
        ),
    )

    # 전차가 정확히 path point 위에 있지 않을 수 있으므로
    # 현재 위치 -> 가장 가까운 path point 거리도 코너거리 계산에 포함한다.
    current_to_nearest = math.hypot(
        float(path[nearest_index][0]) - px,
        float(path[nearest_index][1]) - pz,
    )

    # 현재 위치 다음부터 path 끝 직전까지 각 점을 코너 후보로 검사한다.
    for candidate_index in range(
        nearest_index + 1,
        len(path) - 1,
    ):
        # 현재 전차 위치에서 후보 코너까지 path를 따라간 실제 거리.
        distance_to_corner = (
            current_to_nearest
            + _path_distance_between(
                path,
                nearest_index,
                candidate_index,
            )
        )

        # preview 범위를 넘어갔다면 더 먼 점은 볼 필요가 없으므로 종료한다.
        if distance_to_corner > preview_distance:
            break

        # 후보점 기준 뒤쪽 방향을 잡기 위한 index.
        back_index = candidate_index

        # 후보점에서 뒤로 얼마나 이동했는지 누적거리.
        accumulated = 0.0

        # 후보점에서 뒤쪽으로 direction_sample_distance만큼 이동한다.
        while back_index > nearest_index:
            x1, z1 = map(
                float,
                path[back_index],
            )
            x0, z0 = map(
                float,
                path[back_index - 1],
            )

            accumulated += math.hypot(
                x1 - x0,
                z1 - z0,
            )

            back_index -= 1

            if accumulated >= direction_sample_distance:
                break

        # 후보점 기준 앞쪽 방향을 잡기 위한 index.
        front_index = candidate_index

        # 앞쪽 누적거리 초기화.
        accumulated = 0.0

        # 후보점에서 앞쪽으로 direction_sample_distance만큼 이동한다.
        while front_index < len(path) - 1:
            x0, z0 = map(
                float,
                path[front_index],
            )
            x1, z1 = map(
                float,
                path[front_index + 1],
            )

            accumulated += math.hypot(
                x1 - x0,
                z1 - z0,
            )

            front_index += 1

            if accumulated >= direction_sample_distance:
                break

        # 앞/뒤 샘플이 실제로 후보점에서 떨어지지 않았다면
        # 방향 비교가 불가능하므로 해당 후보를 건너뛴다.
        if (
            back_index == candidate_index
            or front_index == candidate_index
        ):
            continue

        # 뒤쪽 샘플 point.
        bx, bz = map(
            float,
            path[back_index],
        )

        # 코너 후보 point.
        cx, cz = map(
            float,
            path[candidate_index],
        )

        # 앞쪽 샘플 point.
        fx, fz = map(
            float,
            path[front_index],
        )

        # 코너 진입 전 방향.
        # +Z = 0도, +X = +90도 기준.
        heading_before = math.degrees(
            math.atan2(
                cx - bx,
                cz - bz,
            )
        )

        # 코너 통과 후 방향.
        heading_after = math.degrees(
            math.atan2(
                fx - cx,
                fz - cz,
            )
        )

        # 두 방향 차이를 -180~180 범위로 정규화한 뒤 절댓값을 취해
        # 실제 코너의 꺾임 각도 크기를 얻는다.
        turn_angle = abs(
            normalize_angle_deg(
                heading_after
                - heading_before
            )
        )

        # 설정한 threshold 이상의 꺾임이면 의미 있는 코너로 판정한다.
        if turn_angle >= corner_angle_threshold:
            return {
                # 현재 위치에서 코너까지 path를 따라간 거리.
                "distance": distance_to_corner,

                # 코너 꺾임 각도.
                "angle": turn_angle,

                # path 배열 안의 코너 index.
                "index": candidate_index,

                # 코너 world/grid 좌표.
                "point": (cx, cz),
            }

    # preview 범위 안에서 코너를 찾지 못함.
    return None


# ============================================================
# 11. 출발/방향 정렬 시 전진 가속 제한
# ============================================================

def apply_alignment_speed_limit(
    command,
    heading_error_deg,
    current_speed_kmh,
):
    """
    차체가 경로 방향과 크게 어긋난 상태에서
    W=1.0으로 바로 가속하여 벽에 충돌하는 것을 막는다.

    주의:
    이 함수는 코너 자체의 목표속도를 계산하는 함수가 아니라,
    차체 정렬 상태가 좋지 않을 때 W 출력을 보조적으로 제한하는 함수다.
    """

    # command 자체가 없으면 그대로 반환.
    if not command:
        return command

    # command 안의 전/후진 명령 부분을 읽는다.
    ws = command.get(
        "moveWS",
        {},
    )

    # W 전진 명령이 아니면 S 제동 등을 수정하지 않는다.
    if ws.get("command") != "W":
        return command

    # 현재 목표 방향과 차체 방향의 절대 오차 [deg].
    angle = abs(
        float(heading_error_deg)
    )

    # 현재 속도 [km/h].
    speed = float(
        current_speed_kmh
    )

    # PID가 원래 요청한 W weight.
    original_weight = float(
        ws.get(
            "weight",
            0.0,
        )
    )

    # 50도 이상 크게 틀어져 있으면 전진을 끄고 조향을 우선한다.
    if angle >= 50.0:
        command["moveWS"] = {
            "command": "",
            "weight": 0.0,
        }
        return command

    # 30~50도면 최대 W=0.20으로 제한한다.
    if angle >= 30.0:
        command["moveWS"] = {
            "command": "W",
            "weight": min(
                original_weight,
                0.20,
            ),
        }
        return command

    # 15~30도면 최대 W=0.45로 제한한다.
    if angle >= 15.0:
        command["moveWS"] = {
            "command": "W",
            "weight": min(
                original_weight,
                0.45,
            ),
        }
        return command

    # 7~15도는 저속일 때만 W를 0.70 이하로 제한한다.
    # 이미 20km/h 이상 움직이고 있다면 heading error만으로
    # 계속 가속을 막지 않도록 한다.
    if (
        angle >= 7.0
        and speed < 20.0
    ):
        command["moveWS"] = {
            "command": "W",
            "weight": min(
                original_weight,
                0.70,
            ),
        }
        return command

    # 7도 미만이면 PID가 만든 W 출력을 그대로 사용한다.
    return command


# ============================================================
# 12. 코너 종류/거리별 목표속도 제한
# ============================================================

def calculate_corner_speed_limit(
    corner,
    normal_speed_kmh=MAX_SPEED_KMH,
):
    """
    find_upcoming_corner()가 찾은 코너의 각도와 거리로
    현재 허용할 목표속도를 결정한다.

    현재 버전은 연속식이 아니라
    Sharp/Medium/Gentle + Near/Mid/Far 규칙 기반 방식이다.
    """

    # 앞쪽에 코너가 없으면 일반 최고속도를 허용한다.
    if corner is None:
        return float(
            normal_speed_kmh
        )

    # 현재 위치에서 코너까지 거리.
    distance = float(
        corner["distance"]
    )

    # 코너의 방향 변화각.
    angle = float(
        corner["angle"]
    )

    # 70도 이상 Sharp 코너.
    if angle >= SHARP_CORNER_ANGLE_DEG:

        # 8m 이내면 가장 강한 속도 제한.
        if distance <= CORNER_NEAR_DISTANCE_M:
            return SHARP_CORNER_SPEED_NEAR_KMH

        # 18m 이내면 중간 수준 제한.
        if distance <= CORNER_MID_DISTANCE_M:
            return SHARP_CORNER_SPEED_MID_KMH

        # 35m 이내면 선행 감속.
        if distance <= CORNER_FAR_DISTANCE_M:
            return SHARP_CORNER_SPEED_FAR_KMH

    # 40~70도 Medium 코너.
    elif angle >= MEDIUM_CORNER_ANGLE_DEG:

        if distance <= CORNER_NEAR_DISTANCE_M:
            return MEDIUM_CORNER_SPEED_NEAR_KMH

        if distance <= CORNER_MID_DISTANCE_M:
            return MEDIUM_CORNER_SPEED_MID_KMH

        if distance <= CORNER_FAR_DISTANCE_M:
            return MEDIUM_CORNER_SPEED_FAR_KMH

    # 20~40도 Gentle 코너.
    elif angle >= CORNER_ANGLE_THRESHOLD_DEG:

        # 완만한 코너는 현재 코드에서 5m 이내일 때만 제한한다.
        if distance <= 5.0:
            return GENTLE_CORNER_SPEED_NEAR_KMH

    # 위 조건에 해당하지 않으면 일반 목표속도를 그대로 사용한다.
    return float(
        normal_speed_kmh
    )


# ============================================================
# 13. D* Lite 경로 추종 조향 명령 생성
# ============================================================

def calculate_steering_command(
    current_position,
    body_yaw_deg,
    path,
    current_speed_kmh,
    dt,
):
    """
    현재 위치와 D* Lite path를 받아
    Look-ahead point를 향하도록 A/D 조향 명령을 계산한다.

    반환:
        steering_command, steering_info
    """

    # 속도가 높을수록 경로를 조금 더 멀리 바라본다.
    # clamp로 최소/최대 Look-ahead를 보장한다.
    lookahead_distance = clamp(
        LOOKAHEAD_BASE_M
        + LOOKAHEAD_SPEED_GAIN
        * current_speed_kmh,
        LOOKAHEAD_MIN_M,
        LOOKAHEAD_MAX_M,
    )

    # 실제 D* Lite 경로 상에서 lookahead_distance만큼 앞쪽 point를 선택한다.
    target_point, target_index = (
        select_lookahead_point(
            path,
            current_position,
            lookahead_distance,
        )
    )

    # 경로가 없어 target을 선택할 수 없으면 조향 입력을 해제한다.
    if target_point is None:

        # 이전 조향 오차가 다음 경로에 영향을 주지 않도록 PID 상태 초기화.
        steering_pid.reset()

        return {
            "command": "",
            "weight": 0.0,
        }, None

    # 현재 위치.
    cx, cz = map(
        float,
        current_position,
    )

    # 목표 Look-ahead point.
    tx, tz = map(
        float,
        target_point,
    )

    # 목표점까지 X 방향 차이.
    dx = tx - cx

    # 목표점까지 Z 방향 차이.
    dz = tz - cz

    # 목표점을 바라보기 위한 절대 heading.
    # atan2(dx, dz)를 사용하여 +Z=0°, +X=+90° 기준으로 계산한다.
    target_heading_deg = (
        math.degrees(
            math.atan2(
                dx,
                dz,
            )
        )
        % 360.0
    )

    # 목표 heading - 현재 yaw.
    # normalize하여 -180~180 범위의 가장 짧은 회전오차를 얻는다.
    heading_error_deg = (
        normalize_angle_deg(
            target_heading_deg
            - float(body_yaw_deg)
        )
    )

    # 오차가 deadband 안에 있으면 직진으로 보고 조향하지 않는다.
    if (
        abs(heading_error_deg)
        <= STEER_DEADBAND_DEG
    ):
        steering_pid.reset()
        steer_output = 0.0

    else:
        # 방향 오차를 steering PID에 넣어 A/D weight를 계산한다.
        steer_output = (
            steering_pid.update(
                heading_error_deg,
                dt,
            )
        )

    # 양수 출력은 오른쪽 D.
    if steer_output > 0.0:
        ad_command = "D"

        # 혹시 PID 출력이 최대값을 넘더라도 조향 상한을 다시 보장한다.
        ad_weight = min(
            abs(steer_output),
            STEER_MAX_WEIGHT,
        )

    # 음수 출력은 왼쪽 A.
    elif steer_output < 0.0:
        ad_command = "A"

        ad_weight = min(
            abs(steer_output),
            STEER_MAX_WEIGHT,
        )

    # 0이면 조향 없음.
    else:
        ad_command = ""
        ad_weight = 0.0

    # 서버 로그/디버깅에서 사용할 부가정보.
    info = {
        "target_point": (
            tx,
            tz,
        ),
        "target_index": target_index,
        "lookahead_m": lookahead_distance,
        "target_heading_deg": (
            target_heading_deg
        ),
        "body_yaw_deg": (
            float(body_yaw_deg)
        ),
        "heading_error_deg": (
            heading_error_deg
        ),
        "steer_output": steer_output,
    }

    # 실제 서버 command 형식에 맞춘 A/D 명령과
    # 디버깅 정보를 함께 반환한다.
    return {
        "command": ad_command,
        "weight": round(
            ad_weight,
            4,
        ),
    }, info


# ============================================================
# 14. 서버 반환 command 생성
# ============================================================

def make_stop_command():
    """
    모든 이동/조향 입력을 해제한 command를 만든다.

    'STOP'이라는 별도 명령을 보내는 것이 아니라
    command="" / weight=0 형태로 입력을 해제한다.
    """

    return {
        # 전진/후진 입력 해제.
        "moveWS": {
            "command": "",
            "weight": 0.0,
        },

        # 좌/우 조향 입력 해제.
        "moveAD": {
            "command": "",
            "weight": 0.0,
        },

        # 포탑 좌/우 입력 해제.
        "turretQE": {
            "command": "",
            "weight": 0.0,
        },

        # 포탑 상/하 입력 해제.
        "turretRF": {
            "command": "",
            "weight": 0.0,
        },

        # 발사하지 않음.
        "fire": False,
    }


def make_longitudinal_command(
    pid_output,
):
    """
    속도 PID의 -1~+1 출력을 서버 moveWS 명령으로 변환한다.

    pid_output > 0  -> W
    pid_output < 0  -> S
    pid_output ≈ 0  -> 입력 해제
    """

    # 너무 작은 PID 출력은 제어 노이즈로 보고 무시한다.
    deadband = 0.02

    # 양수 PID 출력이면 전진.
    if pid_output > deadband:
        ws_command = "W"

        # 서버 weight 범위인 0~1로 제한한다.
        ws_weight = clamp(
            pid_output,
            0.0,
            1.0,
        )

    # 음수 PID 출력이면 S.
    # 시뮬레이터에서는 감속/후진 방향 입력이다.
    elif pid_output < -deadband:
        ws_command = "S"

        # weight는 양수로 보내야 하므로 절댓값 사용.
        ws_weight = clamp(
            abs(pid_output),
            0.0,
            1.0,
        )

    # deadband 안이면 가속/제동 입력 없음.
    else:
        ws_command = ""
        ws_weight = 0.0

    # 서버가 요구하는 command JSON 형식으로 반환한다.
    return {
        "moveWS": {
            "command": ws_command,
            "weight": round(
                ws_weight,
                4,
            ),
        },

        # 이 함수는 longitudinal만 담당하므로
        # 조향은 빈 상태로 생성한다.
        # 서버에서 calculate_steering_command() 결과를 나중에 덮어쓴다.
        "moveAD": {
            "command": "",
            "weight": 0.0,
        },

        # 포탑 제어는 이 모듈의 대상이 아니므로 비활성.
        "turretQE": {
            "command": "",
            "weight": 0.0,
        },

        "turretRF": {
            "command": "",
            "weight": 0.0,
        },

        "fire": False,
    }


# ============================================================
# 15. 외부 서버에서 호출할 초기화 함수
# ============================================================

def reset_pid_controllers():
    """
    목적지가 바뀌거나 episode가 초기화될 때 호출한다.

    이전 목적지에서 누적된 PID의 I/D 상태가
    새 목적지 제어에 영향을 주지 않도록 한다.
    """

    # 속도 PID 초기화.
    speed_pid.reset()

    # 조향 PID 초기화.
    steering_pid.reset()


def reset_speed_state():
    """
    /info 기반 속도 추정 상태를 초기화한다.

    시뮬레이터 재시작/episode 초기화 시 이전 위치와 시간이
    새 episode의 속도 계산에 섞이는 것을 방지한다.
    """

    global info_speed_kmh
    global info_previous_position
    global info_previous_time

    # 현재 필터링 속도 삭제.
    info_speed_kmh = None

    # 이전 위치 삭제.
    info_previous_position = None

    # 이전 측정 시간 삭제.
    info_previous_time = None



def reset_all_controller_state():
    """
    PID 제어와 /info 속도 추정 상태를 모두 초기화한다.

    사용 시점
    ---------
    - /init
    - 시뮬레이터 episode 재시작
    - 주행을 처음부터 다시 시작할 때

    초기화 항목
    -----------
    speed_pid
    steering_pid
    last_control_time
    arrival_latched
    last_pid_destination
    final_brake_start_time
    info_speed_kmh
    info_previous_position
    info_previous_time

    왜 필요한가
    -----------
    새 episode에서 이전 주행의 PID 오차,
    목적지 도착 상태, 제동 timer, 속도 추정 기준이
    남아 있지 않도록 한 번에 정리하기 위한 함수다.
    """

    # PID + /get_action 제어 상태 전체 초기화.
    reset_control_state(
        reset_destination=True
    )

    # /info 기반 속도 추정 상태 초기화.
    reset_speed_state()
