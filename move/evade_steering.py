# -*- coding: utf-8 -*-
"""
evade_steering.py — 적 전차 차체 선회(evade) 기동

Tank Challenge · Team 3 (Rotex) · 사격통제 파트
enemy-controller.py (5100 번 서버) 에서 **차체를 돌리는 부분만** 뽑은 것.

═══════════════════════════════════════════════════════════════
 무엇이 들어 있고 무엇이 없나
═══════════════════════════════════════════════════════════════

 들어 있다   차체 기동 — 목표점 계산, 조향, 안전 울타리 4종
 없다        포탑 선회(_slew) · 조준 · 사격 판정
             /info · /get_action 같은 Flask 엔드포인트
             _S 를 채우는 _update_self() · _poll_ally()

 원본 코드 그대로다. 주석 한 줄 고치지 않았고, 함수마다
 [원본 enemy-controller.py NNN줄] 로 출처를 적어 두었다.

═══════════════════════════════════════════════════════════════
 명령이 만들어지는 순서
═══════════════════════════════════════════════════════════════

     get_action()                    (원본에 있음, 이 파일에는 없음)
         |
         +-- _b_evade()              어디로 갈지 정한다
         |       +-- _ring_to()          아군 중심 원 위의 목표점
         |               +-- _steer_to()     방위 오차 -> (A/D, weight)
         |
         +-- _fence(cmd)             위험하면 그 명령을 덮어쓴다
                 |
                 +-- ① _avoid()          충돌 회피   (최우선)
                 |      +-- _tcpa()           최근접 예측
                 |      +-- _corridor_gap()   아군 진행선까지 거리
                 |      +-- _retreat_target() 비켜설 자리
                 |             +-- _ally_box()   아군 발자국 상자
                 |
                 +-- ② 교전지역 울타리    구역 밖이면 되돌린다
                 +-- ③ 맵 경계 울타리     최후의 보루
                 +-- ④ 이탈 울타리        FAR 이상 벌어지면 복귀

 반환 형식 — 시뮬레이터 /get_action 응답

     {"moveWS":   {"command": "W"|"S"|"STOP", "weight": 0~1},
      "moveAD":   {"command": "A"|"D"|"",     "weight": 0~1},
      "turretQE": {...}, "turretRF": {...}, "fire": bool}

 차체는 moveWS / moveAD 만 쓴다. turret 두 개와 fire 는 항상 빈 값이다.

═══════════════════════════════════════════════════════════════
 evade 가 어떤 기동인가
═══════════════════════════════════════════════════════════════

 아군을 중심으로 한 원 위를 돌되, WEAVE_PERIOD(8 초)마다 선회 방향을
 **구형파로** 뒤집는다. 반경도 아주 천천히 흔든다(±15 %).

 각도를 사인파로 쓸어 넘기는 방식은 2026-08-26 에 실패했다.
 목표점이 전차보다 20 배 빨리 돌아 제자리에서 떨기만 했고, 그 떨림이
 아군 추적기에 '거대한 선회율' 로 잡혀 60발 0명중이 나왔다.
 실패 기록은 _b_evade 주석에 그대로 남겨 두었다.

═══════════════════════════════════════════════════════════════
 이 파일만으로는 서버가 안 돌아간다
═══════════════════════════════════════════════════════════════

 원본에 아래가 있어야 한다.

   _S              공유 상태 딕셔너리
   _update_self()  내 위치 · 속도 · 차체 방위 갱신
   _poll_ally()    아군 좌표를 5000 번에서 폴링 (+ 아군 속도 추정)
   get_action()    위 함수들을 순서대로 부르는 곳

 다만 기동만 떼어 돌려 볼 수는 있다. 파일 맨 아래에 _S 를 흉내 낸
 최소 상태와 demo() 를 붙여 두었다. 그 부분만 원본에 없는 코드다.

     python evade_steering.py

 아군이 순회로를 도는 동안 적이 어떻게 움직이는지 좌표가 찍힌다.

═══════════════════════════════════════════════════════════════
 최근에 바뀐 것
═══════════════════════════════════════════════════════════════

 2026-09-10  충돌 회피를 _fence 맨 앞으로. 그 전에는 교전지역 울타리가
             먼저라 아군이 그 경로 위에 있어도 밀고 들어갔다
             (47번 판 실측: 두 전차 거리 17.6 m)
 2026-09-10  EVADE_SPEED 3.4 -> 6.5. 실측 enemy_speed 로 정정
 2026-09-09  교전지역 울타리 추가, KEEP_RANGE 25 -> 30
 2026-08-26  _ring_to 목표점 추종으로 전면 재작성 (그 전 궤도는 발산)
 2026-08-25  안전 울타리 추가 (그 전에는 맵 밖으로 떨어졌다)
═══════════════════════════════════════════════════════════════
"""

import math
import time


SPEED = 0.254          # 기본 전진 weight (0~1). 낮출수록 느리다
                      #
                      # 유일한 실측:  SPEED 0.30  ->  3.42 m/s   (8/25 charge)
                      # 선형 환산하면 0.13 은 약 1.5 m/s 인데 아직 미검증이다.
                      #
                      # 주의 — 8/26 첫 판의 0.30 m/s 는 이 값의 실측이 아니다.
                      #   그 판은 BEHAVIOR 가 static 이라 SPEED 를 쓰지도 않았다.
                      #   (/set?b=circle 을 안 걸었다)
                      #   그때 적이 26.8 m 를 움직인 것은 시뮬레이터 내장
                      #   추적 AI 가 밀고 우리 STOP 이 막아서 생긴 잔여 이동이다.
                      #
                      # ★ circle 로 한 판 돌린 뒤 실제 속도를 재고 여기를 맞춘다.
TURN = 0.45           # 기본 조향 weight
KEEP_RANGE = 30.0     # circle / strafe 가 유지하려는 거리 [m]
                      #
                      # 2026-08-25  45 -> 25
                      #   45 m 궤도는 아군이 (110,253) 에 있을 때
                      #   북쪽 끝이 z=298 이 되어 맵(300)을 벗어난다.
                      #   실제로 적이 z=300.0 벽에 붙어 못 나왔다.
                      #
                      #   25 m 로 줄이면 궤도가 x 85~135, z 228~278
                      #   로 맵 안에 완전히 들어가고, 궤도 위 최대
                      #   경사도 28° -> 15° 로 낮아진다.
                      #   사선은 18방향 전부 트인다.
                      #
                      #   덤: 사격 모듈이 스스로 권장하는 거리가
                      #   24~26 m 다 (state.fire.suggest). 비행시간이
                      #   0.45 s 로 짧아 예측오차가 크게 준다
                      #   (오차는 비행시간의 제곱에 비례).
                      #
                      # ── 2026-09-09  25.0 -> 32.0 ────────────────
                      #   44번 판 집계에서 897틱 중 253틱(28%)이
                      #   '최소사거리미달' 이었다. 적이 아군에게
                      #   너무 붙어서 탄도해가 아예 안 나온 것이다.
                      #
                      #   최소 사거리는 고저차에 아주 민감하다.
                      #       dy +0.0 m -> 20.5 m
                      #       dy +0.6 m -> 24.6 m   <- 44번 판 로그값
                      #       dy +1.0 m -> 27.2 m
                      #       dy +1.5 m -> 30.2 m
                      #   25 m 로 잡으면 평지에서도 여유가 0.4 m 다.
                      #   32 m 면 dy +1.5 m 까지 버틴다.
                      #
                      #   아군 쪽 상한(tof_max 0.70 s = 약 44 m)과
                      #   MIN_ENGAGE 25 m 사이라 아군 사격도 안 막는다.
                      #   44번 판 아군 사격 거리는 23.9~34.5 m 였다.
WEAVE_PERIOD = 8.0    # evade 의 좌우 반전 주기 [s]
                      #   2026-08-26  3.0 -> 8.0
                      #   3 초로는 전차가 반전을 끝내지 못하고 제자리에서
                      #   떨기만 했다. 선회 18 °/s 기준 180° 반전에
                      #   10 초가 필요하다. 8 초면 144° — 실제로 뒤집힌다.
                      #   strafe(20 초)의 2.5 배 빈도라 충분히 어렵다.


# 이보다 가까우면 전진을 멈춘다.
#   우리 최소 사거리가 20.5 m 다 (near_tol 3.0 감안 하한 17.5 m).
#   기존 18.0 은 여유가 1.5 m 뿐이라, 고저차가 조금만 생겨도
#   최소 사거리가 커져 사격 불가가 된다.
#   (8/12 실측: dy +3 m 이면 최소 사거리가 38 m 로 뛴다)
#   30 m 로 두면 어떤 고저차에서도 안전하게 교전할 수 있다.
STOP_RANGE = 36.0
MAP = 300.0


# [원본 enemy-controller.py 426줄]
def _norm180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# [원본 enemy-controller.py 430줄]
def _bearing(a, b) -> float:
    """a 에서 b 를 보는 방위 [deg]. 0 = +Z(북), 시계방향."""
    return (math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) + 360.0) % 360.0


# [원본 enemy-controller.py 435줄]
def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


# [원본 enemy-controller.py 439줄]
def _cmd(ws="", w_ws=0.0, ad="", w_ad=0.0, fire=False):
    # 2026-08-25  STOP 일 때는 조향을 반드시 비운다.
    #
    #   주행팀 pid_controller.py 는 멈출 때 항상 이렇게 보낸다:
    #       "moveWS": {"command": "STOP", "weight": 1.0}
    #       "moveAD": {"command": "",     "weight": 0.0}
    #
    #   우리는 STOP 과 D 를 같이 보내고 있었다 (로그: 명령=STOP/D).
    #   "멈춰라 + 동시에 돌아라" 는 모순된 명령이라 물리가 튄다.
    #   실제로 적 속도가 21 m/s(76 km/h)까지 치솟고 월드 밖으로
    #   떨어졌다 (enemy y = -4832 m).
    if ws == "STOP":
        ad, w_ad = "", 0.0
    return {
        "moveWS": {"command": ws, "weight": round(w_ws, 2)},
        "moveAD": {"command": ad, "weight": round(w_ad, 2)},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": bool(fire),
    }


# [원본 enemy-controller.py 461줄]
def _no_data():
    """
    내 위치나 아군 위치를 모를 때의 명령.

    2026-08-25 수정: 예전에는 여기서 _cmd("W", SPEED) 로 '일단 직진' 했다.
    그런데 아군 좌표가 아예 안 들어와서 이 분기에 영원히 머물렀고,
    적 전차가 방향도 안 보고 맵 끝(z=0)까지 직진해 버렸다.
    모르면 멈추는 게 맞다. 멈춰 있으면 이상하다는 걸 바로 알 수 있다.
    """
    return _cmd("STOP", 1.0)


# [원본 enemy-controller.py 473줄]
def _steer_to(target_brg: float):
    """
    목표 방위로 돌기 위한 (조향키, weight).
    오차가 작으면 조향하지 않는다 — 지그재그를 막는다.
    """
    err = _norm180(target_brg - _S["body"])
    if abs(err) < 4.0:
        return "", 0.0
    w = min(1.0, max(0.15, abs(err) / 45.0)) * TURN / 0.45
    return ("D" if err > 0 else "A"), min(1.0, w)


# [원본 enemy-controller.py 485줄]
def _drive_toward(point, speed=None, stop_range=STOP_RANGE):
    """point 를 향해 전진. 너무 가까우면 멈춘다."""
    me = _S["my"]
    if me is None:
        return _no_data()
    d = _dist(me, point)
    ad, w_ad = _steer_to(_bearing(me, point))
    if d <= stop_range:
        return _cmd("STOP", 1.0, ad, w_ad)
    return _cmd("W", speed if speed is not None else SPEED, ad, w_ad)


# ══════════════════════════════════════════════════════════
#  행동
# ══════════════════════════════════════════════════════════


# [원본 enemy-controller.py 512줄]
def _ring_to(lead_deg, radius):
    """
    아군을 중심으로 한 반경 radius 원 위에서, 지금 내 각도보다
    lead_deg 만큼 앞선 점으로 간다.

    2026-08-26  circle / strafe / evade 의 공통 뼈대.

      예전 방식('접선 방향으로 가라')은 열린 제어라 궤도가 계속 부풀었다.
      실측에서 반경 45 m 를 요구했는데 79 m 까지 벌어졌고 맵 벽에 처박혔다.
      목표점을 직접 계산하고 맵 안으로 가두면 그런 일이 원천 차단된다.

      lead_deg 부호가 선회 방향이다. 크기가 클수록 접선에 가깝게 돌고,
      작으면 반경 오차를 빨리 되잡는다.
    """
    me, ally = _S["my"], _S["ally"]

    # ── 최소 사거리 하한 ──────────────────────────────────────────
    #
    #   lead_deg 가 크면 목표점이 접선 쪽으로 멀어져 전차가 안쪽으로
    #   질러가고, 그만큼 반경이 줄어든다. 모의에서 evade 가 11.3 m,
    #   strafe 가 16.6 m 까지 붙었다.
    #   우리 최소 사거리가 20.5 m 라 그 안으로 들어오면 아예 못 쏜다.
    #   표적이 사거리 밖으로 도망가는 것과 똑같이 측정이 망가진다.
    #
    #   그래서 너무 붙으면 lead 를 줄이고(=바깥을 곧장 향하게)
    #   목표 반경을 키워 밀어낸다.
    d = _dist(me, ally)

    # ── 반경 오차 되먹임 ──────────────────────────────────────────
    #
    #   lead 가 크면 전차가 거의 접선 방향으로 달리므로 반경을 되잡는
    #   힘이 약하다. 방향을 뒤집는 순간마다 조금씩 밀려나고, 그게 쌓여
    #   모의에서 13 ~ 48 m 까지 벌어졌다. 목표는 25 m 다.
    #
    #   반경이 어긋난 만큼 lead 를 줄인다.
    #     오차 0 m   -> lead 그대로 (순수 선회)
    #     오차 12 m  -> lead 15 % (거의 곧장 목표 반경으로)
    #   되잡으면 저절로 다시 선회로 돌아간다.
    lead_deg *= max(0.10, 1.0 - abs(d - radius) / 6.0)

    # 양쪽 하드 한계 — 어느 쪽으로 벗어나도 측정이 망가진다.
    #   안쪽: 최소 사거리 20.5 m 아래면 아예 못 쏜다.
    #   바깥: 비과시간이 tof_max 0.70 s 를 넘으면 게이트가 막는다
    #         (39 m 부근). 도망간 것과 같다.
    if d < MIN_KEEP:
        lead_deg *= 0.25
        radius = max(radius, KEEP_RANGE * 1.35)
    elif d > MAX_KEEP:
        lead_deg *= 0.25
        radius = min(radius, KEEP_RANGE * 0.90)

    a = math.radians(_bearing(ally, me) + lead_deg)
    tx = ally[0] + radius * math.sin(a)
    tz = ally[1] + radius * math.cos(a)
    tx = max(EDGE, min(MAP - EDGE, tx))      # 맵 안으로 강제
    tz = max(EDGE, min(MAP - EDGE, tz))
    ad, w_ad = _steer_to(_bearing(me, (tx, tz)))
    return _cmd("W", SPEED, ad, w_ad)


# [원본 enemy-controller.py 572줄]
def _b_evade():
    """
    가장 어려운 표적 — 좌우로 계속 흔든다.

    2026-08-26 재작성. 예전에는 '아군 쪽으로 가되 방위에 흔들림을 더한다'
    였는데, 적이 이미 25 m 궤도 위에 스폰하므로 '접근' 자체가 의미가 없다.
    이제는 궤도를 돌되 선회 방향이 주기적으로 뒤집히게 한다.

    왜 어려운가
      선회 방향이 뒤집힐 때마다 표적의 진행 방향이 반대로 꺾인다.
      우리 예측기는 '지금 속도와 가속도가 유지된다'고 가정하므로
      꺾이는 순간 예측이 가장 크게 빗나간다.
      비과시간 0.49 s 동안 표적이 방향을 바꾸면 리드가 통째로 틀린다.

    반경도 함께 흔들어 거리 예측까지 흔든다.
    """
    me, ally = _S["my"], _S["ally"]
    if me is None or ally is None:
        return _no_data()

    # ── 2026-08-26  전면 재작성. 첫 시도는 완전히 실패했다 ────────────
    #
    #   실패한 방식
    #       lead = 55° * sin(2*pi*t/3)      각도를 사인파로 쓸어 넘김
    #
    #   왜 실패했나
    #       목표점이 반경 25 m 원 위를 최대 115 °/s 로 돈다.
    #       그 선속도가 50 m/s 다. 적 전차는 2.5 m/s 다. 20 배 빠르다.
    #       전차는 목표점을 절대 못 쫓고 제자리에서 좌우로 떨기만 한다.
    #
    #       그 떨림이 우리 추적기에 '거대한 선회율' 로 잡히고,
    #       CTRV 예측이 원을 그리며 말려들어가 조준점이 엉뚱한 데로 갔다.
    #           조준점까지 거리  15.61 m  (200 초 내내 고정)
    #           실제 적 거리     42.80 m
    #           탄착 오차        28 ~ 67 m
    #           60발 0명중
    #
    #       측정하려던 것(회피 기동)이 아니라 '고장난 표적' 을 만든 셈이다.
    #
    #   고친 방식 — strafe 와 같은 '방향 뒤집기', 대신 주기를 짧게
    #       lead 를 사인파로 쓸지 않고 부호만 뒤집는다(구형파).
    #       목표점은 순간 반대편으로 '점프' 할 뿐 쓸고 다니지 않으므로
    #       전차는 그냥 새 목표를 향해 돌면 된다. strafe 가 잘 됐던 이유다.
    #
    #       주기 8 초 = strafe(20 초)의 2.5 배 빈도.
    #       전차 선회 18 °/s 로 8 초면 144° 를 돌 수 있어 실제로 반전한다.
    #   반전할 때마다 궤도 바깥으로 밀려나므로 기준 반경을 조금 키운다.
    #   실측 선회율(drift_deg 로 역산) 28~38 °/s 범위에서 모의한 결과
    #   KEEP x 1.12 = 28 m 가 최소 사거리 20.5 m 를 가장 잘 지켰다.
    ph = math.sin(2.0 * math.pi * _S["t"] / WEAVE_PERIOD)
    lead = 26.0 if ph >= 0.0 else -26.0          # 구형파 — 쓸지 않고 점프
    # 반경도 흔들어 거리 예측까지 흔든다. 단 아주 천천히.
    rad = KEEP_RANGE * 1.12 * (1.0 + 0.15 * math.sin(
        2.0 * math.pi * _S["t"] / (WEAVE_PERIOD * 2.1)))
    return _ring_to(lead, rad)


# ══════════════════════════════════════════════════════════
#  안전 울타리 — 2026-08-25 추가
# ══════════════════════════════════════════════════════════
#
#  왜 필요한가
#      circle / strafe / evade 에는 맵 경계 검사가 전혀 없었다.
#      (MAP=300 상수는 _b_linear 만 썼다)
#      선회하다 남쪽 끝 z=0 으로 나가 월드 밖으로 떨어졌다.
#
#          enemy = [87.86, -4832.8, 0.0]     지면 아래 4.8 km
#          enemy_speed = 21.16 m/s
#
#      떨어진 뒤에도 좌표는 계속 오므로 아군은 허공을 조준했고,
#      보정값(bias)까지 오염됐다.
#
#  두 겹으로 막는다
#      ① 맵 가장자리 EDGE 안으로 들어오면 맵 중심으로 되돌린다.
#      ② 아군에게서 FAR 이상 멀어지면 아군 쪽으로 되돌린다.
#         (측정 구간을 벗어나 헤매는 것도 막는다)
# 이보다 가까워지면 궤도를 바깥으로 밀어낸다 [m].
#   우리 최소 사거리가 20.5 m 다. 그 안으로 들어오면 아예 못 쏘므로
#   표적이 도망간 것과 똑같이 측정이 망가진다. 여유를 둬 23 m.
MIN_KEEP = 27.0


# 이보다 멀어지면 궤도를 안쪽으로 당긴다 [m].
#   비과시간이 tof_max 0.70 s 를 넘으면 게이트가 사격을 막는다.
#   0.70 s x 56.36 m/s = 39.5 m 이므로 36 m 에서 되잡는다.
MAX_KEEP = 42.0


EDGE = 12.0     # 맵 경계에서 이만큼은 떨어져 있는다 [m]
                #   2026-08-25  30 -> 12
                #   30 은 너무 넓어서 북쪽 교전구역 자체를 금지구역으로
                #   만들었다. 실제 사고는 z=300 (벽) 에서 났으므로
                #   12 m 면 충분하고, 궤도를 넉넉히 담을 수 있다.
FAR = 55.0      # 아군에게서 이보다 멀어지지 않는다 [m]


# ══════════════════════════════════════════════════════════
#  교전지역  (2026-09-09 추가)
# ══════════════════════════════════════════════════════════
#
#  왜 필요한가
#      기존 울타리는 '맵 전체(300x300)' 와 '아군까지 FAR' 두 가지뿐이라
#      "여기서만 싸운다" 는 개념이 없었다.
#
#      9/08 evade 5 m/s 시험(05번)에서 적이
#          118 s  (108, 174)   49 m
#          122 s  (118, 159)   65 m   <- FAR 55 m 를 넘김
#          210 s  (161,  26)  205 m   맵 남동쪽 구석
#      까지 흘러가 7 발 만에 판이 죽었다. 명중률은 7발 7명중 100 % 였다.
#      못 맞춘 게 아니라 표본을 못 모은 것이다.
#
#  좌표는 어떻게 얻었나
#      2026-09-09. tracking mode 를 끄고 아군 전차를 키보드로 몰아
#      네 귀퉁이에 세운 뒤 3D 상황도의 '아군 기갑' 마커를 읽었다.
#          (75, 287)   (75, 259)   (131, 253)   (142, 290)
#      그 외접 직사각형이 아래 값이다.
#
#  지형
#      네 지점 표고가 8 ~ 12 m 로 기복이 4 m 다.
#      데모맵 전체는 7.9 ~ 23.6 m (기복 15.7 m) 이므로 이 구역은
#      훨씬 평탄하다. 고저차 dy 가 최소 사거리를 흔드는 문제가 작다.
#      (dy +3 m 면 최소 사거리가 20.5 -> 38 m 로 뛴다. 8/12 실측)
ZONE_X0, ZONE_X1 = 75.0, 142.0
ZONE_Z0, ZONE_Z1 = 253.0, 290.0


# 울타리를 푸는 깊이 [m].  나가면 바로 걸리고, 이만큼 들어와야 풀린다.
#   같은 선에서 걸고 풀면 경계에 붙어 덜덜 떤다 (히스테리시스).
#   5 m/s 로 달리면 한 틱(0.13 s)에 0.65 m 를 가고, 방향을 되돌리는 데
#   선회 시간이 더 걸리므로 5 m 는 있어야 한다.
ZONE_MARGIN = 5.0


ZONE_X0 = max(ZONE_X0, EDGE)
ZONE_X1 = min(ZONE_X1, MAP - EDGE)
ZONE_Z0 = max(ZONE_Z0, EDGE)
ZONE_Z1 = min(ZONE_Z1, MAP - EDGE)


# [원본 enemy-controller.py 788줄]
def _zone_center():
    return ((ZONE_X0 + ZONE_X1) * 0.5, (ZONE_Z0 + ZONE_Z1) * 0.5)


# [원본 enemy-controller.py 792줄]
def _in_zone(p, margin=0.0):
    """점이 교전지역 안인가. margin 을 주면 그만큼 안쪽만 참으로 본다."""
    return (ZONE_X0 + margin <= p[0] <= ZONE_X1 - margin
            and ZONE_Z0 + margin <= p[1] <= ZONE_Z1 - margin)


# ══════════════════════════════════════════════════════════
#  충돌 회피  (2026-09-10 다시 넣음)
# ══════════════════════════════════════════════════════════
#
#  왜 다시 넣나
#      8/26 에 "표본이 다 모이면 다시 넣는 것을 권한다" 며 뺐다가
#      9/09 에 넣었는데, 그 뒤 파일을 여러 번 덮어쓰는 과정에서
#      통째로 사라졌다. 46·47번 판에서 두 전차가 계속 붙은 이유다.
#      47번 판 /state 실측: 두 전차 거리 17.6 m, 1919틱 중 849틱(44%)이
#      '최소사거리미달' — 사격 자체가 성립하지 않는 거리였다.
#
#  설계 세 겹
#      ① 절대금지선 (AVOID_HARD)
#         이 안에 들어오면 다른 울타리를 전부 무시하고 물러난다.
#         교전지역을 잠깐 벗어나도 좋다. 충돌보다는 낫다.
#      ② 회피선 (AVOID_RANGE) / 해제선 (AVOID_CLEAR)
#         히스테리시스. 같은 선에서 걸고 풀면 경계에서 덜덜 떤다.
#      ③ 최근접 예측 (TCPA)
#         거리만 보면 늦는다. 마주 달리면 상대속도가 10 m/s 라
#         한 틱(0.21 s)에 2 m 씩 좁혀진다. 앞으로 PREDICT_S 안에
#         절대금지선을 깰 것 같으면 지금 피한다.
#
#  물러날 곳을 고르는 법
#      아군 반대 방향이 1순위지만, 그쪽이 교전지역 밖이면 소용없다.
#      아군 반대편을 기준으로 좌우로 훑어, 교전지역 안이면서 아군에게서
#      가장 먼 점을 고른다.

AVOID_HARD = 16.0     # 절대금지선 [m]
                      #   전차 길이 7.5 m 의 두 배 남짓.
                      #   상대속도 10 m/s 라면 1.6 초 여유다.
AVOID_RANGE = 24.0    # 이 안으로 들어오면 회피 시작 [m]
AVOID_CLEAR = 30.0    # 이만큼 벌어져야 회피 해제 [m]
                      #   KEEP_RANGE(30) 와 같다. 회피가 풀리는 자리가
                      #   곧 궤도 반지름이라 바로 교전으로 이어진다.
PREDICT_S = 2.0       # 최근접을 이 시간까지 내다본다 [s]

# ── 2026-09-10  경로 회랑 ────────────────────────────────
#
#   속도가 문제다. 적은 약 2.9 m/s (SPEED 0.254), 긴급해도 3.4 m/s 다.
#   아군은 5~7 m/s 로 달린다. 적이 더 느리다.
#
#   느린 쪽이 24 m 에서 반응하면 늦는다. 아군이 24 m 를 오는 데
#   3.4~4.8 초, 적은 그 사이에 방향을 틀고(수 초) 옆으로 5 m 를
#   빠져나가야 한다. 모의에서 실제로 못 빠져나갔다.
#
#   그래서 '점' 이 아니라 '선' 을 피한다.
#   아군은 tour.py 가 준 경유점을 향해 직선으로 달린다. 그 진행
#   방향으로 PATH_LOOK 초만큼 그은 선이 앞으로 지나갈 자리다.
#   그 선에서 CORRIDOR_M 안에 있으면, 아직 멀더라도 미리 비킨다.
#   느린 쪽이 빠른 쪽을 피하는 유일한 방법은 '먼저' 비키는 것이다.
PATH_LOOK = 6.0       # 아군 진행 방향을 이만큼 앞까지 본다 [s]
CORRIDOR_M = 20.0     # 그 선에서 이만큼은 떨어져 있는다 [m]

# ── 2026-09-10  아군 순회 구역 ────────────────────────────
#
#   아군은 tour.py 가 준 사각 순회로를 계속 돈다
#   (92,263)-(128,263)-(128,278)-(92,278).
#   그 사각형 '안' 에는 안전한 자리가 없다. 지금 비켜도 한 바퀴 뒤에
#   다시 걸린다. 모의에서 적이 (103~114, 267~274) 를 맴돌며 계속
#   최소거리 2 m 까지 붙었다.
#
#   적은 그 순회로를 모른다. 대신 아군이 지나간 자리를 모아 상자를
#   만든다. 회피할 자리를 고를 때 그 상자에서 먼 곳을 크게 우대한다.
#   교전지역은 x 75~142 / z 253~288 이고 순회로는 그보다 작으므로
#   상자 밖에도 설 자리가 있다.
HOME_KEEP_S = 90.0    # 아군 발자국을 이만큼 기억한다 [s]
HOME_PAD = 8.0        # 상자를 이만큼 넓혀 본다 [m]


# [원본 enemy-controller.py 867줄]
def _ally_box():
    """아군이 최근에 지나간 자리를 감싸는 상자. 자료가 모자라면 None."""
    hist = _S["ally_hist"]
    if len(hist) < 8:
        return None
    xs = [p[0] for p in hist]
    zs = [p[1] for p in hist]
    return (min(xs) - HOME_PAD, max(xs) + HOME_PAD,
            min(zs) - HOME_PAD, max(zs) + HOME_PAD)


# [원본 enemy-controller.py 878줄]
def _box_gap(p, box):
    """상자 밖이면 상자까지의 거리, 안이면 음수(가장 가까운 변까지)."""
    x0, x1, z0, z1 = box
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dz = max(z0 - p[1], 0.0, p[1] - z1)
    if dx > 0.0 or dz > 0.0:
        return math.hypot(dx, dz)
    return -min(p[0] - x0, x1 - p[0], p[1] - z0, z1 - p[1])


# [원본 enemy-controller.py 888줄]
def _corridor_gap(me, ally, ally_v):
    """아군이 앞으로 지나갈 선까지의 거리 [m]. 정지해 있으면 점까지 거리."""
    sp = math.hypot(ally_v[0], ally_v[1])
    if sp < 0.5:
        return _dist(me, ally)
    ex, ez = ally[0] + ally_v[0] * PATH_LOOK, ally[1] + ally_v[1] * PATH_LOOK
    ax, az = ally
    dx, dz = ex - ax, ez - az
    L2 = dx * dx + dz * dz
    t = ((me[0] - ax) * dx + (me[1] - az) * dz) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(me[0] - (ax + dx * t), me[1] - (az + dz * t))


# [원본 enemy-controller.py 902줄]
def _tcpa(me, my_v, ally, ally_v, horizon):
    """
    앞으로 horizon 초 안의 최근접 (시각, 거리) 를 돌려준다.

    상대운동을 직선으로 본다. 두 전차 모두 조향 중이면 정확하지 않지만,
    '지금 좁혀지고 있는가' 를 판정하는 데에는 충분하다.
    """
    rx, rz = ally[0] - me[0], ally[1] - me[1]
    vx, vz = ally_v[0] - my_v[0], ally_v[1] - my_v[1]
    vv = vx * vx + vz * vz
    if vv < 1e-6:
        return 0.0, math.hypot(rx, rz)
    t = -(rx * vx + rz * vz) / vv
    t = max(0.0, min(horizon, t))
    return t, math.hypot(rx + vx * t, rz + vz * t)


# 적의 실제 최고 속도 [m/s].  회피 방향을 고를 때 쓴다.
#
#   2026-09-10 정정.  처음에 3.4 로 잡았다. 근거는 이 파일 SPEED 주석의
#   "8/25 실측 SPEED 0.30 -> 3.42 m/s" 였는데, 그 값을 그대로 믿으면 안
#   된다는 것이 사격 기록으로 드러났다.
#
#   shots CSV 의 enemy_speed 열은 시뮬레이터가 알려 준 적의 실제 속도다.
#   9개 판을 모아 보면 SPEED = 0.254 로 도는데도
#
#       판별 최대   5.37 ~ 6.80 m/s      (전체 최대 6.80)
#       판별 평균   3.57 ~ 4.57 m/s
#
#   같은 자료에서 아군 최대는 7.76 m/s 다. 적이 느린 것이 아니라
#   거의 비슷하다. 3.4 로 두면 회피 방향을 고를 때 '이만큼밖에 못 간다'
#   고 과소평가해서 엉뚱한 쪽을 고른다.
EVADE_SPEED = 6.5


# [원본 enemy-controller.py 937줄]
def _retreat_target(me, ally, want):
    """
    비켜설 자리를 고른다.

    ── 왜 '반대 방향' 이 답이 아닌가 ────────────────────────────
    적의 최고 속도는 약 3.4 m/s 인데 아군은 5~7 m/s 로 달린다.
    정반대로 도망치면 속도차만큼 계속 좁혀진다. 절대 못 벗어난다.
    느린 쪽이 빠른 쪽을 피하는 방법은 '뒤로' 가 아니라 '옆으로' 다.

    ── 왜 매 틱 다시 고르면 안 되는가 ──────────────────────────
    아군이 경유점을 돌 때마다 진행 방향이 확 바뀐다. 그때마다 다시
    고르면 적은 좌우로 흔들리기만 하고 제자리를 못 뜬다.
    모의에서 실제로 그랬다 — 순회로 한가운데를 맴돌며 2 m 까지 붙었다.
    그래서 한 번 고르면 도착하거나 위험해질 때까지 붙잡는다.

    ── 무엇을 좋은 자리로 보는가 ──────────────────────────────
    ① 아군 발자국 상자 밖일 것 — 한 바퀴 뒤에 또 걸리지 않는다
    ② 앞으로 PATH_LOOK 초 동안 아군과 멀 것
    ③ 교전지역 안일 것 (밖도 허용하되 감점)
    """
    now = time.time()
    al_v = _S["ally_vel"] or [0.0, 0.0]
    box = _ally_box()

    # 잡아 둔 자리가 아직 쓸 만하면 그대로 쓴다.
    lock = _S["safe_tgt"]
    if lock is not None:
        far_enough = _dist(ally, lock) > AVOID_CLEAR * 0.8
        arrived = _dist(me, lock) < 4.0
        stale = (now - _S["safe_t"]) > 20.0
        if far_enough and not arrived and not stale:
            return lock
        _S["safe_tgt"] = None

    def score(px, pz):
        v = 0.0
        # ① 아군 발자국 상자에서 얼마나 떨어졌나 (가장 크게 본다)
        if box is not None:
            v += 2.0 * max(-12.0, min(20.0, _box_gap((px, pz), box)))
        # ② 앞으로 아군과 얼마나 멀리 있게 되나
        worst = 1e9
        for k in range(1, 13):
            t = PATH_LOOK * k / 12.0
            axx, azz = ally[0] + al_v[0] * t, ally[1] + al_v[1] * t
            worst = min(worst, math.hypot(px - axx, pz - azz))
        v += min(worst, 45.0)
        # ③ 교전지역 / 맵 경계
        if not (ZONE_X0 + 3.0 <= px <= ZONE_X1 - 3.0
                and ZONE_Z0 + 3.0 <= pz <= ZONE_Z1 - 3.0):
            # 2026-09-10  -25 -> -60.
            #   -25 로는 '구역을 나가면 아군에게서 아주 멀어진다' 는
            #   점수가 이겨서, 적이 교전지역을 통째로 벗어나 z 233 까지
            #   내려가 앉았다. 충돌은 안 나지만 교전이 성립하지 않는다.
            v -= 60.0
        if not (EDGE <= px <= MAP - EDGE and EDGE <= pz <= MAP - EDGE):
            v -= 80.0
        # 너무 멀면 가는 데 오래 걸린다 — 가까운 쪽을 살짝 우대
        v -= 0.15 * _dist(me, (px, pz))
        return v

    # 교전지역 안을 먼저 훑는다. 점수만으로 두면 '구역 밖으로 멀리
    # 도망가는' 쪽이 이겨서 적이 판을 떠나 버린다 (모의에서 구역이탈
    # 65 % 까지 나왔다). 안에 설 자리가 있으면 무조건 안에서 고른다.
    def search(inside_only):
        bb, bs = None, -1e9
        for deg in range(0, 360, 10):
            a = math.radians(deg)
            for r in (want * 0.6, want, want * 1.5):
                px = me[0] + r * math.sin(a)
                pz = me[1] + r * math.cos(a)
                if inside_only and not (
                        ZONE_X0 + 3.0 <= px <= ZONE_X1 - 3.0
                        and ZONE_Z0 + 3.0 <= pz <= ZONE_Z1 - 3.0):
                    continue
                sc = score(px, pz)
                if sc > bs:
                    bb, bs = [px, pz], sc
        return bb, bs

    best, best_s = search(True)
    if best is None:
        best, best_s = search(False)
    if best is None:
        best = [me[0], me[1]]
    _S["safe_tgt"] = best
    _S["safe_t"] = now
    return best


# [원본 enemy-controller.py 1026줄]
def _avoid(me, ally):
    """
    충돌 회피 명령. 필요 없으면 None.

    다른 어떤 울타리보다 먼저 본다. 여기서 명령이 나오면 그대로 나간다.
    """
    d = _dist(me, ally)

    # 판 전체 최소 거리를 기록해 둔다 (판정 근거)
    if _S["min_gap"] is None or d < _S["min_gap"]:
        _S["min_gap"] = round(d, 2)
    if d < AVOID_HARD:
        _S["gap_hard"] += 1
    if d < AVOID_RANGE:
        _S["gap_warn"] += 1

    avoiding = str(_S.get("fence") or "").startswith("충돌")

    # 최근접 예측
    my_v = _S["my_vel"] or [0.0, 0.0]
    al_v = _S["ally_vel"] or [0.0, 0.0]
    t_c, d_c = _tcpa(me, my_v, ally, al_v, PREDICT_S)
    closing = (d_c < AVOID_HARD and t_c > 0.0)

    # 아군이 앞으로 지나갈 선까지의 거리
    cg = _corridor_gap(me, ally, al_v)
    in_corridor = (cg < CORRIDOR_M)

    need = ((d < AVOID_RANGE) or closing or in_corridor
            or (avoiding and (d < AVOID_CLEAR or cg < CORRIDOR_M + 6.0)))
    if not need:
        if avoiding:
            print(f"[적] 충돌 회피 해제 — 아군까지 {d:.1f} m, 경로에서 {cg:.1f} m")
            _S["fence"] = ""
        return None

    tgt = _retreat_target(me, ally, AVOID_CLEAR)
    out = _bearing(me, tgt)
    err = abs(_norm180(out - _S["body"]))
    ad, w = _steer_to(out)

    if not avoiding:
        why = (f"거리 {d:.1f} m" if d < AVOID_RANGE
               else f"경로에서 {cg:.1f} m" if in_corridor
               else f"{t_c:.1f}초 뒤 {d_c:.1f} m 로 근접")
        print(f"[적] ★ 충돌 회피 — {why}. "
              f"({tgt[0]:.0f}, {tgt[1]:.0f}) 로 비킨다.")

    # ── 속도 ────────────────────────────────────────────────
    #   회피 중에는 항상 최고 속도다. 적이 아군보다 느리므로
    #   평소 속도(SPEED 0.254)로 비키면 제때 못 빠져나간다.
    #   후진도 0.7 이 아니라 1.0 으로 올렸다.
    #
    #   차체가 비킬 쪽을 보면 전진, 아니면 후진. 후진은 방위 추정이
    #   틀려도 반드시 차체 뒤로 가므로 벽에 눌려도 빠져나온다.
    if err <= 90.0:
        _S["fence"] = "충돌-긴급전진" if d < AVOID_HARD else "충돌-전진"
        return _cmd("W", 1.0, ad, w)
    _S["fence"] = "충돌-긴급후진" if d < AVOID_HARD else "충돌-후진"
    return _cmd("S", 1.0, ad, w)


# [원본 enemy-controller.py 1088줄]
def _fence(cmd):
    """행동이 만든 명령을 검사해, 위험하면 되돌리는 명령으로 바꾼다."""
    me = _S["my"]
    if me is None:
        return cmd
    x, z = me
    now = time.time()

    # ── 2026-09-10  ★ 충돌 회피가 가장 먼저다 ────────────────────
    #
    #   교전지역 울타리보다 먼저 본다. 47번 판에서 적이 구역 밖
    #   (107.8, 252.0) 에서 중심 (108, 270) 으로 되돌아가는 중이었는데,
    #   아군이 (97.1, 265.9) 로 그 경로 위에 있었다. 구역 울타리는
    #   아군을 보지 않으므로 그대로 밀고 들어간다.
    #
    #   충돌을 피하려고 잠깐 구역을 벗어나는 것은 허용한다.
    #   구역은 교전 편의를 위한 선이고, 충돌은 판 자체를 망친다.
    _ally = _S["ally"]
    if _ally is not None:
        _av = _avoid(me, _ally)
        if _av is not None:
            return _av

    # ── 2026-09-09  교전지역 울타리 ─────────────────────────────
    #
    #  맵 경계 울타리보다 '먼저' 본다.
    #      교전지역은 위에서 EDGE 로 잘라 맵 안쪽에 완전히 들어가 있다.
    #      따라서 교전지역을 지키면 맵 경계는 저절로 지켜진다.
    #      맵 경계 울타리는 그대로 두되 이제 최후의 보루 역할만 한다.
    #
    #  되돌리는 방법은 맵 경계 울타리와 같다.
    #      차체가 구역 중심 쪽을 보면(오차 90도 이내) 전진,
    #      등지고 있으면 후진. 후진은 방위 추정이 틀려도 반드시
    #      차체 뒤쪽으로 가므로 벽에 눌려 방위가 굳어도 빠져나온다.
    #      (8/25 · 8/26 에 z=300 벽에서 두 판 연속 죽은 원인이 그것이다)
    #
    #  히스테리시스
    #      나가는 순간 걸고, ZONE_MARGIN 만큼 들어와야 푼다.
    #      같은 선에서 걸고 풀면 경계에 붙어 덜덜 떤다.
    _zoning = str(_S.get("fence") or "").startswith("교전지역")
    if (not _in_zone((x, z))) or (_zoning and not _in_zone((x, z), ZONE_MARGIN)):
        _tgt = _zone_center()
        _out = _bearing(me, _tgt)
        _err = abs(_norm180(_out - _S["body"]))
        _ad, _w = _steer_to(_out)
        if not _zoning:
            print(f"[적] ★ 교전지역 이탈 — 위치 ({x:.0f}, {z:.0f}), "
                  f"구역 x {ZONE_X0:.0f}~{ZONE_X1:.0f} / z {ZONE_Z0:.0f}~{ZONE_Z1:.0f}. "
                  f"중심 ({_tgt[0]:.0f}, {_tgt[1]:.0f}) 으로 되돌린다.")
        if _err <= 90.0:
            _S["fence"] = "교전지역-전진"
            return _cmd("W", SPEED, _ad, _w)
        _S["fence"] = "교전지역-후진"
        return _cmd("S", 0.7, _ad, _w)
    if _zoning:
        print(f"[적] 교전지역 복귀 완료 — ({x:.0f}, {z:.0f})")
        _S["fence"] = ""

    # ── 2026-08-25  끼임 감지 ────────────────────────────
    #
    #   적이 z=300.0 (북쪽 벽) 에 정확히 붙어 못 빠져나왔다.
    #   울타리는 켜져 있었고 'W + D' 를 계속 보냈는데도 소용없었다.
    #
    #   이유: 차체 방위를 '위치 변화'로 재기 때문이다.
    #   벽에 정면으로 눌린 채 옆으로 미끄러지면 실제 차체는 북쪽(0°)
    #   을 보는데 이동은 동쪽이라 추정 방위가 90.0° 로 굳는다.
    #   (실측 body 가 정확히 90.0° 였다)
    #   틀린 방위로 조향하니 영원히 제자리다.
    #
    #   해결: 앞으로 못 가면 방위를 따지지 말고 그냥 후진한다.
    #   후진은 방위를 몰라도 벽에서 떨어지게 해준다.
    near_edge = (x < EDGE or x > MAP - EDGE or z < EDGE or z > MAP - EDGE)

    # ── 2026-08-26  경계 탈출을 전면 재작성 ──────────────────────
    #
    #  틀렸던 것 ①  끼임 판정
    #      "4 초 동안 3 m 도 못 움직이면 끼임" 으로 봤다.
    #      벽에 눌린 채 옆으로 미끄러지면 3 m 를 넘기므로
    #      '움직이니까 안 끼었다' 고 판단해 버린다.
    #      실측: 북쪽 벽(z=300)에 붙은 채 x 가 124 -> 189 로 65 m 밀렸는데
    #            한 번도 안 걸렸다. 291 초, 약 41 발 분량을 날렸다.
    #
    #  틀렸던 것 ②  탈출 방법
    #      "끼면 무조건 후진" 으로 했다. 그런데 차체가 이미 맵 안쪽을
    #      보고 있으면 후진은 오히려 벽으로 되돌아가는 방향이다.
    #      모의에서 벽을 벗어나다 말고 다시 붙었다.
    #
    #  바른 방법 — 차체가 어디를 보는지로 정한다
    #      맵 중심 방향과 차체 방위의 차이가
    #          90° 이내  ->  안쪽을 본다. 그냥 전진하면 빠져나온다.
    #          90° 초과  ->  벽을 본다. 전진하면 계속 박는다. 후진한다.
    #      후진하면 차체는 방위 +180° 로 움직이므로 반드시 안쪽으로 간다.
    #      시간 조건이 필요 없다. 경계에 닿는 순간 바로 옳은 쪽으로 움직인다.
    if near_edge:
        out = _bearing(me, (MAP * 0.5, MAP * 0.5))     # 맵 중심 = 무조건 안쪽
        err = abs(_norm180(out - _S["body"]))
        ad, w = _steer_to(out)                          # 어느 쪽이든 계속 돌린다
        if err <= 90.0:
            _S["fence"] = "경계-전진"
            return _cmd("W", SPEED, ad, w)
        _S["fence"] = "경계-후진"
        return _cmd("S", 0.7, ad, w)

    # ② 아군과의 거리
    ally = _S["ally"]
    if ally is not None and _dist(me, ally) > FAR:
        # 2026-09-08  이탈은 드물게 일어나므로 시작·종료를 반드시 남긴다.
        if _S.get("fence") != "이탈":
            print(f"[적] ★ 이탈 울타리 작동 — 아군까지 {_dist(me, ally):.0f} m "
                  f"(한계 {FAR:.0f} m). 아군 쪽으로 되돌린다.")
        ad, w = _steer_to(_bearing(me, ally))
        _S["fence"] = "이탈"
        return _cmd("W", SPEED, ad, w)
    if _S.get("fence") == "이탈":
        print("[적] 이탈 울타리 해제 — 궤도로 복귀.")

    _S["fence"] = ""
    return cmd


# ══════════════════════════════════════════════════════════
#  엔드포인트
# ══════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════
#  여기서부터는 원본에 없다 — 이 파일만 돌려 보기 위한 최소 장치
# ══════════════════════════════════════════════════════════
#
#   원본에서는 _S 가 서버 전역 상태이고, _update_self() 와 _poll_ally()
#   가 채운다. 여기서는 같은 열쇠만 가진 딕셔너리를 만들어
#   기동 부분만 떼어 돌려 본다.

_S = {
    "t": 0.0,          # 시뮬레이터 시각 [s]
    "my": None,        # 적(=이 코드가 조종하는 전차) 위치 [x, z]
    "ally": None,      # 아군 위치 [x, z]
    "body": 0.0,       # 적 차체 방위 [deg]
    "n": 0,
    "wp": 0,
    "prev": None, "prev_t": 0.0,
    "body_ok": True, "body_true": True,
    "ally_t": 0.0, "ally_err": "",
    "fence": "",       # 어떤 울타리가 개입했는가
    "edge_t": 0.0,
    "ally_vel": None,  # 아군 속도 추정 [vx, vz]   (_poll_ally 가 채운다)
    "my_vel": None,    # 내 속도 추정   [vx, vz]   (_update_self 가 채운다)
    "ally_raw": None, "ally_raw_t": 0.0,
    "ally_hist": [],   # 아군 발자국  [(x, z, t), ...]
    "safe_tgt": None, "safe_t": 0.0,
    "min_gap": None, "gap_hard": 0, "gap_warn": 0,
    "poll_n": 0, "fresh_n": 0, "fresh_dt": 0.0,
}


def demo(ally_speed=5.0, ticks=900, dt=0.212,
         enemy_start=(120.0, 270.0), tank_speed=6.5, turn_rate=60.0):
    """
    아군이 tour.py 순회로를 도는 동안 적이 어떻게 움직이는지 찍는다.

    주의 — 이건 모의다. 실제 시뮬레이터의 가감속·지형·충돌은 없다.
    weight 를 속도에 그대로 곱하는 1차 모형이다.
    tank_speed 6.5 m/s 는 실측 enemy_speed 최대값(9판 6.80)에서 잡았다.
    """
    WPS = [(88.0, 264.0), (108.0, 264.0), (108.0, 274.0), (88.0, 274.0)]
    ex, ez = enemy_start
    ax, az = WPS[0]
    body = 0.0
    wi = 1
    prev = [ex, ez]
    mind = 1e9

    print("아군 순회로 %s" % " -> ".join("(%.0f,%.0f)" % w for w in WPS))
    print("적 시작 (%.0f, %.0f)   아군 %.1f m/s   %d 틱 x %.3f s\n"
          % (ex, ez, ally_speed, ticks, dt))
    print("%6s %8s %16s %16s  %s" % ("t[s]", "거리", "적", "아군", "울타리"))

    for k in range(ticks):
        # 아군은 경유점으로 직진한다. 적을 보지 않는다.
        tx, tz = WPS[wi]
        d = math.hypot(tx - ax, tz - az)
        if d < 2.0:
            wi = (wi + 1) % len(WPS)
            tx, tz = WPS[wi]
            d = math.hypot(tx - ax, tz - az)
        avx, avz = (tx - ax) / d * ally_speed, (tz - az) / d * ally_speed

        _S["t"] = k * dt
        _S["my"] = [ex, ez]
        _S["ally"] = [ax, az]
        _S["body"] = body
        _S["ally_vel"] = [avx, avz]
        _S["my_vel"] = [(ex - prev[0]) / dt, (ez - prev[1]) / dt]
        prev = [ex, ez]
        _S["ally_raw"] = [ax, az]
        _S["ally_raw_t"] = time.time()
        h = _S["ally_hist"]
        if not h or math.hypot(h[-1][0] - ax, h[-1][1] - az) > 1.5:
            h.append((ax, az, time.time()))
            del h[:-400]

        cmd = _fence(_b_evade())

        # 명령 -> 이동 (1차 모형)
        ws, wv = cmd["moveWS"]["command"], cmd["moveWS"]["weight"]
        step = tank_speed * wv * dt
        if ws == "W":
            ex += step * math.sin(math.radians(body))
            ez += step * math.cos(math.radians(body))
        elif ws == "S":
            ex -= step * math.sin(math.radians(body))
            ez -= step * math.cos(math.radians(body))
        ad, aw = cmd["moveAD"]["command"], cmd["moveAD"]["weight"]
        if ad == "D":
            body = (body + turn_rate * aw * dt) % 360.0
        elif ad == "A":
            body = (body - turn_rate * aw * dt) % 360.0

        ax += avx * dt
        az += avz * dt
        gap = math.hypot(ax - ex, az - ez)
        mind = min(mind, gap)

        if k % 60 == 0:
            print("%6.1f %7.1fm (%6.1f,%6.1f) (%6.1f,%6.1f)  %s"
                  % (_S["t"], gap, ex, ez, ax, az, _S["fence"] or "-"))

    print()
    print("최소 거리 %.2f m   금지선(%.0f m) 아래 %d틱   회피선(%.0f m) 아래 %d틱"
          % (mind, AVOID_HARD, _S["gap_hard"], AVOID_RANGE, _S["gap_warn"]))


if __name__ == "__main__":
    demo()
