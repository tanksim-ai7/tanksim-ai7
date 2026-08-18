# -*- coding: utf-8 -*-
# ── 버전 ────────────────────────────────────────────────
#   파일   auto_aim_bot_v6.py
#   버전   v10   (2026-08-07)
#   역할   봇 본체 + 대시보드. CFG 가 여기 있다.
#   변경 이력은 같은 폴더의  변경이력.md  를 볼 것.
#   ※ 파일명은 바꾸지 않는다 (import 가 이름으로 걸려 있다).
#      버전 구분은 이 배너 + 날짜 폴더(260806/260807/…) 로 한다.
# ────────────────────────────────────────────────────────
"""
auto_aim_bot_v6.py - 정지 사격 기반 고명중률 교전 봇

목표: 정지표적 / 이동표적 명중률 90% 이상

v5 대비 핵심 변경 (자세한 내용은 CHANGES.xlsx)
  B1  정지 사격 상태기계     MOVE -> HALT -> AIM -> FIRE -> (재장전 중) MOVE
                             포탑이 비안정화라 차체가 움직이면 발사 순간
                             포탑 월드각이 틀어진다. v5 는 allow_moving_fire=True
                             로 이동 중에도 쐈다. 재장전이 6.6 s 나 되므로
                             정지 사격으로 바꿔도 화력 손실이 거의 없다.
  B2  발사 틱 포탑 정지      fire 와 회전 명령을 동시에 내지 않는다.
  B3  교전 거리 자동 선택    고정 밴드(45~70 m) 폐지. 매 틱 P(hit) 이 최대가 되는
                             거리를 계산해 그쪽으로 기동한다.
                             정지표적 -> 26~31 m (비행시간 최소)
                             20 m/s 횡단표적 -> 45~60 m (포탑 추적 여유 확보)
  B4  자기 보정 되먹임       착탄점을 BiasEstimator 로 되먹여 잔여 모형오차 제거
  B5  사격 기록 정합 개선    한 번에 한 발만 비행하므로 시간창 우선으로 짝짓는다
  B6  런타임 튜닝            /set?band_near=..&p_hit_min=..&behavior=..
  B7  조건별 통계            정지/이동, 거리대별 명중률을 나눠 집계
  B8  추적 불가 판정         포탑 40 deg/s 로 따라갈 수 없는 표적은 쏘지 않고
                             거리를 벌린다 (NOTRACK)
  B9  인내 로직              오래 못 쏘면 임계 P(hit) 을 서서히 낮춰
                             '완벽한 기회를 기다리다 한 발도 못 쏘는' 것을 막는다

오프라인 검증 (offline_sim.py, 판당 220 s x 25 판)
    정지표적  99.6%   등속직선 100%   회피(사행) 94.0%   돌진 98.3%
    순찰(전속) 90.0%  선회(20 m/s 전속 궤도) 58.0%  <- 물리 한계
    (v5 동일 조건: 정지 96%, 회피 53%, 선회 25%)

실행:  python auto_aim_bot_v6.py    ->  http://localhost:5000
같은 폴더에 fire_control.py, enemy_bot.py 필요.

시뮬레이터 설정
    Mode = Simulation,  Request Port = 5000
    Enemy Request Port = 5100,  Use Enemy Server 체크
    Terrain = Simple Flat  (탄도 상수가 평지 기준으로 측정됨)
"""
import math
import os
import logging
import threading
import time
from flask import Flask, request, jsonify, Response

from enemy_bot import EnemyBot
from fire_control import (Ballistics, FireControl, TurretParams, TargetSize,
                          TargetTracker, MotionLimits, BiasEstimator,
                          hit_probability, target_extents, desired_range,
                          required_yaw_rate, optimal_range,
                          impact_aspect, aspect_zone, bearing, ang_diff, dist2d)

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = Flask(__name__)

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
START_BLUE = (150.0, 10.0, 60.0)
START_RED = (150.0, 10.0, 150.0)     # 초기 거리 90 m - 접근 시간 단축

MAP_MARGIN = 35.0
BODY_DEADBAND = 3.0
YAW_RATE = 40.0            # 차체 선회 (P1 실측)

# 제어 주기 초기값. 봇이 매 틱 실측해서 덮어쓰므로 '시동값'일 뿐이지만,
# 첫 수십 틱의 조준 게인을 정하므로 실제와 가까운 값이 좋다.
#   2026-08-05 실사격 로그 실측: 중앙값 0.1329 s (Master PC, Interval 0.1)
#   기존 0.41 은 다른 PC 값이었다. 3.1 배 차이.
CTRL_DT = 0.133

# B13: 틱 단위 진단 로그. /save 시 ticks_v6_<tag>.csv 로 함께 저장된다.
TICK_LOG = True

# B16: 저장 파일 이름에 붙는 태그. 대시보드에서 바꾼다.
CURRENT_TAG = "run1"

CFG = {
    # ── 교전 거리 (B3, B8) ────────────────────────────────
    # 거리는 고정이 아니라 '표적 횡단 속도'에 맞춰 자동으로 정해진다.
    #   가까울수록  비행시간이 짧아 예측오차가 작다 (오차 ∝ t^2)
    #   멀수록      필요한 포탑 각속도가 작아 추적 여유가 생긴다
    # 포탑은 40 deg/s 뿐이라, 20 m/s 로 횡단하는 표적은 28.6 m 안쪽에서는
    # 물리적으로 추적 자체가 불가능하다. (v5/v6 초안이 선회표적에 0발 쏜 원인)
    "band_near": 26.0,     # 탐색 하한 (최소 사거리 21.9 m + 여유)
    "band_far": 72.0,      # 탐색 상한
    "band_half": 9.0,      # 최적 거리 ± 이 폭을 교전 밴드로 삼는다
    "track_duty": 0.45,    # (구식 desired_range 용) 포탑 용량 중 추적 배분
    "track_duty_max": 0.80,  # 이 비율을 넘으면 사격 금지 (NOTRACK)
    # 사격 임계 명중확률. 재장전이 6.6 s 라 헛방이 비싸므로 높게 잡는다.
    # 오프라인 스윕 결과 0.90 에서 회피표적 94% (0.50 이면 92%)
    "p_hit_min": 0.90,
    # 이 시간 넘게 못 쏘면 임계값을 p_hit_floor 까지 서서히 낮춘다 [s]
    "patience": 18.0,
    "p_hit_floor": 0.28,
    # 정지 판정 속도 [m/s]. 이보다 느리면 '멈췄다'로 본다.
    "halt_speed": 0.20,
    # 정지 대기 최대 시간 [s]. 초과하면 정지한 것으로 간주 (교착 방지)
    "halt_timeout": 2.5,
    # 조준이 끝나지 않을 때 재기동으로 돌아가는 시간 [s]
    "aim_timeout": 5.0,
    # 후방 침투 기동
    "flank": True,
    # 데드밴드 여유 계수 (작을수록 엄격 = 더 정확, 더 오래 조준)
    # 오프라인 스윕: 0.50 -> 85%, 0.35 -> 91%, 0.30 -> 93.5% (회피표적)
    "db_safety": 0.30,
    # ── 앙각 데드밴드 상한 [deg]  (B14, 2026-08-05 실사격 로그 반영) ──
    # 왜 0.60 -> 0.15 인가
    #   실사격 34발의 틱 로그에서 앙각 데드밴드가 평균 0.565 deg 로
    #   상한(0.60)에 붙어 있었다. dR/dtheta 가 6.27 m/deg 이므로
    #   이것만으로 사거리 오차 ±3.50 m 를 허용한다.
    #   실측 종방향 표준편차가 2.84 m 였는데, 데드밴드 안에서 균등분포로
    #   가정했을 때의 표준편차가 2.02 m 다. 즉 종방향 산포의 절반이
    #   '예측 오차'가 아니라 '조준을 덜 하고 쏜 것'이었다.
    #
    #   그동안 0.60 을 못 낮춘 이유는 제어 주기를 0.41 s 로 가정했기 때문이다.
    #   그러면 앙각 최소 이동이 5 x 0.02 x 0.41 = 0.041 deg 라 여유가 없었다.
    #   실측 0.133 s 에서는 0.0133 deg 로 3 배 촘촘하다. 조일 수 있다.
    #
    #   dt=0.1329 오프라인 스윕 (정지 / 회피 / 선회, 괄호는 사격수)
    #     0.60  100%(156)  96.2%(160)  99.2%(388)
    #     0.20  100%(156)  97.6%(123)  99.6%(241)
    #   * 0.15  100%(156)  97.6%(126)  99.2%(239)
    #     0.10  100%(156)  98.3%(120)  100%(196)
    #     0.05  100%(156)  97.5%(119)  100%( 20)  <- 사격 기회 붕괴
    #   0.10~0.20 이 평지, 0.15 를 채택 (0.05 는 조준이 안 끝나 못 쏜다)
    #
    #   ! 이 값은 제어 주기에 의존한다. 다른 PC 에서 dt 가 0.4 s 대로
    #     측정되면 0.4 정도로 되돌려야 한다. 대시보드의 실측 dt 를 볼 것.
    "pitch_db_max": 0.15,
    # ── 비행시간 상한 [s]  (C11, 2026-08-06 실사격 근거) ──
    # 이동표적에서 비행시간이 0.5 s 를 넘으면 명중률이 급락한다.
    #   ~0.35 s 90.9% / ~0.50 s 87.0% / ~0.65 s 60.9% / ~1.50 s 25.0%
    # 오차가 비행시간에 비례해 커지므로 확률이 아니라 구조적 불리함이다.
    # None 이면 게이트를 끈다. 정지표적에는 적용되지 않는다.
    #
    # 실측 73발에 사후 필터를 걸어본 결과 (통과율 / 명중률):
    #   없음  100% / 65.8%      0.60s  67% / 81.6%
    #   0.70s  85% / 72.6%      0.55s  58% / 83.3%
    #   0.65s  78% / 77.2%    * 0.50s  48% / 88.6%   <- 채택
    #                           0.40s  21% / 93.3%
    # 0.50 s 는 거리로 약 35 m. 교전밴드(25~35 m)와 일치한다.
    # '사격 손실'로 보이는 52% 는 실제로는 손실이 아니다 —
    # 봇이 먼 거리에서 헛방을 쏘는 대신 접근해서 쏘게 된다.
    "tof_max": 0.50,
    # 발사 전 포탑 정지 틱 수
    "settle_ticks": 0,
    # 자기 보정 ON/OFF (지면 착탄만 학습 - B10 참조)
    "bias": True,
    # C9 의도적 '길게 조준' 세기. 0 = 끔, 1 = 종방향 창의 정중앙을 겨냥.
    # 탄이 완만히 하강하므로 길게 조준하면 차체 윗부분을 때린다.
    # 짧으면 표적 앞 지면에 박혀 무조건 빗나간다.
    #
    # ── v7 변경: 0.8 -> 0.25  (B11, 2026-08-05 오프라인 스윕) ──
    # 0.8 은 '정지·저속 표적'만 보고 정한 값이었다. 이 값이면 조준점이
    # 표적보다 평균 +6.7 m 뒤에 놓인다. 정지표적은 그래도 맞지만,
    # 20 m/s 로 선회하는 표적에서는 여기에 예측 오차가 더해져
    # 종방향 허용창(긴 쪽)을 넘겨 탄이 차체 위를 지나간다.
    # 실패한 발의 횡방향 오차는 ±0.4 m 뿐이고 전부 종방향 +5~10 m 였다.
    #
    #  lon_gain   정지    등속   회피   돌진   순찰   선회    전체
    #    0.80     99.6   100.0  98.7  99.8  98.9  73.5   98.1
    #    0.35     100    100    97.2  99.5  100   92.8   98.7
    #  * 0.25     99.6   100.0  98.4  99.6  100   96.2   98.9
    #    0.15     100    100    96.0  99.5  100   100    99.1  (사격수 급감)
    # 0.15 는 선회 100% 지만 사격 기회가 절반으로 줄어 교전 효율이 나쁘다.
    # 0.25 가 '전 시나리오 90% 초과 + 사격수 유지' 를 동시에 만족한다.
    #
    # ── C14 (2026-08-06 저녁): 0.25 -> 0.15 로 다시 내린다 ────────
    #
    # 위 표는 오프라인 스윕이고, 오프라인의 명중 판정은 lon_long 기하
    # 모델(창 21~25 m)을 그대로 쓴다. 즉 '길게 조준해도 맞는다'를 전제로
    # 만든 표라 lon_gain 을 과대평가한다. 실사격이 그것을 부정했다.
    #
    # 실사격 근거 (mm8 + base_60 + c13 중 C13 조건 충족 87발)
    #   lon_shift 를 빼고 '표적 중심 기준 실제 종방향 착탄 위치'로 환산:
    #
    #     명중 82발   mu +1.76 m   sd 1.26   범위 -1.50 ~ +4.86
    #     실패  5발   mu +8.85 m   sd 2.68   범위 +4.63 ~ +10.88
    #
    #   +4.7 m 에서 깨끗하게 갈린다. 겹치는 구간이 없다.
    #   즉 긴 쪽 실제 한계는 21~25 m 가 아니라 약 +5 m 다.
    #   (C12 에서 4.0 m 로 잡은 건 방향은 맞았고 값이 좁았다. 다만 그때는
    #    횡방향 오차가 압도해 판정 자체가 불가능한 상태였다. C13 으로
    #    횡을 잡고 나서야 이 경계가 보였다.)
    #
    # 지금 조준점은 창 안에서 한쪽으로 치우쳐 있다.
    #     짧은 쪽 한계 약 -3.2 m  |  긴 쪽 한계 약 +4.7 m  -> 중앙 +0.75 m
    #     현재 명중 중심 +1.76 m  ->  긴 쪽까지 2.3 sigma 뿐
    # 2.3 sigma 면 정규분포에서 1% 지만 실측 산포는 꼬리가 두껍다.
    # 남은 실패 5발이 전부 이 꼬리다.
    #
    # 0.15 로 내리면 shift 2.7 -> 1.6 m, 중심이 +0.7 m 로 옮겨가
    # 긴 쪽까지 3.2 sigma, 짧은 쪽까지 3.1 sigma 로 대칭이 된다.
    #
    # '사격수 급감' 우려는 오프라인 관측인데, 그 원인이던 P(hit) 게이트가
    # 실사격에서는 87발 전부 1.00 이라 사실상 작동하지 않는다.
    # 실측에서 사격 수가 줄면 되돌린다.  /set?lon_gain=0.25
    "lon_gain": 0.15,

    # ── C12: 상면 관통 창(lon_long)의 상한 [m] — 시험했고 기각했다 ──
    #
    # None = 끔 (C9 기하 모델 그대로). 기능은 남겨 두되 쓰지 않는다.
    #
    # ▷ 세웠던 가설 (v8 51발 분석)
    #     모델이 믿은 lon_long      21.5 ~ 26.0 m
    #     실제 명중한 최대 종오차          +3.43 m
    #     실패 3발의 종오차       +9.65 / +11.56 / +11.75 m
    #   "C9 의 상면 관통 창이 7 배 낙관이고, 그 탓에 lon_shift 가 매 발
    #    +2.7 m 를 만들어 탄이 차체 위를 지나간다."
    #
    # ▷ 실측 결과 (v9, lon_long_max=4.0, 35발) — 가설이 틀렸다
    #
    #     조건        v8 (창 25 m)      v9 (창 4 m)
    #     이동-이동    28/32  87.5%      11/22  50.0%
    #     이동-정지    16/19  84.2%       7/10  70.0%
    #
    #   적 속도로 층화해도 (교란 제거) 결과가 같다.
    #     적속 0~5    13/13 100%  ->  2/4  50%
    #     적속 5~9     6/8   75%  ->  2/9  22%
    #     적속 9+      9/11  82%  ->  7/9  78%
    #   아군 속도 10+ m/s 에서 73% -> 36%.  Fisher p ~ 0.003.
    #
    # ▷ 가설이 어디서 틀렸나 (두 군데)
    #
    #   1) lon_shift 는 의도대로 2.64 -> 0.13 m 로 사라졌는데
    #      '길게 빠지는 실패'는 그대로 남았다 (+8.89/+7.64/+6.00 m).
    #      2.7 m 를 빼도 9 m 오차는 9 m 다. 애초에 크기가 안 맞았다.
    #      그 9~12 m 는 lon_shift 가 아니라 다른 원인이다.
    #      (유력: 차체 위를 스친 탄이 한참 더 날아가 지면에 박힌 것.
    #       그 경우 range_err 은 '조준 오차'가 아니라 '지나간 거리'다.)
    #
    #   2) v9 에만 새 실패 유형이 7발 생겼다.
    #        착탄점이 조준점 기준 표적 상자 '안'인데 빗나감
    #        v8: 0발  ->  v9: 7발
    #      조준점 자체가 틀린 곳이었다는 뜻 = 예측 오차다.
    #      길게 조준하던 2.6 m 가 예측 오차를 흡수하고 있었고,
    #      그것을 없애자 드러났다.
    #
    # ▷ 남은 교훈
    #   C9 의 비대칭 창은 '왜 맞는지'의 설명은 부정확할지 몰라도
    #   결과적으로 예측 오차의 완충 역할을 한다. 실측이 그렇게 말한다.
    #   창을 좁히려면 예측 오차부터 줄이고 나서 해야 한다.
    #
    #   재시도하려면 6.0 / 8.0 / 12.0 을 각 40발로 재보면 된다.
    #   (대시보드 /set?lon_long_max=6 로 코드 수정 없이 바꿀 수 있다)
    "lon_long_max": None,

    # ── C13: 차체 선회 중 사격 금지. |body_rate| 상한 [deg/s] ────
    #
    # 근거 (2026-08-06 실사격 116발 = move_move_v8 51 + base_60 65)
    #
    #   |body_rate|      발수   명중률   |횡오차| 평균
    #      0~ 5 deg/s     42   100.0%      0.70 m
    #     25~30            3   100.0%      0.77 m
    #     30~90           71    77.5%      1.87 m
    #
    #   사후 게이트  45/45 = 100.0%  (95% 하한 92.1%)
    #     이동-이동  21/21   이동-정지  20/20
    #
    #   아군 속도로 층화해도 회전만 낮으면 100% (8/8 at 9+ m/s).
    #   즉 '빠르면 못 맞힌다'가 아니라 '돌면 못 맞힌다'.
    #
    # 실패 16발 중 11발(69%)이 횡 반치수 초과였다. 종방향이 아니라
    # 횡방향이 문제였고, 그 원인이 차체 선회다.
    # (종오차 +10 m 로 보이던 것들은 옆으로 스쳐 지나간 탄이
    #  10 m 더 날아가 지면에 박힌 거리였다. 조준 오차가 아니다.)
    #
    # 15 deg/s: 차체는 직진(0~5) 아니면 최대선회(40~45)뿐이라
    #           5~30 어디에 두어도 결과가 같다. 가운데를 잡았다.
    #
    # 통과율이 39% 라 사격 수가 줄어든다. 급감하면
    #   /set?body_rate_max=25  로 완화하거나
    #   기동 계층에 '조준 중엔 직진' 을 넣는다.
    "body_rate_max": 15.0,

    # 이동 중 사격 허용 (True 로 두면 v5 동작에 가까워짐)
    "moving_fire": False,
}

# ── 적 전차 조종 ─────────────────────────────────────────
ENEMY_PORT = 5100
ENEMY_BEHAVIOR = "evade"   # static / linear / strafe / circle / evade / charge
ENEMY_KEEP_R = 60.0        # 적이 유지하려는 거리 [m]
ENEMY_SPEED_CAP = 1.0      # 적 주행 weight 상한 (난이도 조절)
ENEMY_MASK = False
ENEMY_TRACKING = True
# ══════════════════════════════════════════════════════════


def neutral():
    return {"moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False}


# ══════════════════════════════════════════════════════════
# 1. 텔레메트리
# ══════════════════════════════════════════════════════════
class Telemetry:
    def __init__(self):
        self.raw = {}
        self.t = None
        self.my = self.enemy = None
        self.body_x = self.turret_x = self.turret_y = 0.0
        self.body_y = self.body_z = 0.0        # B22: 차체 기울기 (앞뒤 / 좌우)
        self.enemy_body_x = 0.0
        self.my_hp = self.enemy_hp = None
        self.my_speed = self.enemy_speed = 0.0

    def update(self, d):
        if not d:
            return
        self.raw.update(d)
        r = self.raw
        self.t = r.get("time")
        p, e = r.get("playerPos"), r.get("enemyPos")
        if isinstance(p, dict):
            self.my = (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))
        if isinstance(e, dict):
            self.enemy = (float(e.get("x", 0)), float(e.get("y", 0)), float(e.get("z", 0)))
        self.body_x = float(r.get("playerBodyX", self.body_x) or 0.0)
        # ── B22 (2026-08-12): 차체 기울기 ────────────────────────
        #   시뮬레이터가 보내주는데 지금까지 한 번도 안 읽었다.
        #   평지에서는 둘 다 항상 0 이라 드러날 수가 없었다.
        #   Forest 에서 포탑을 돌릴수록 앙각 오차가 커지는 현상
        #   (0.83° -> 3.85° -> 7.13°) 의 원인 후보다.
        #   ** 지금은 읽어서 로그로만 남긴다. 조준에는 쓰지 않는다. **
        #   8/13 수정 — 시뮬레이터는 각도를 0~360 으로 보낸다.
        #   8/12 에는 그대로 넣어서 로그가 348.67° 같은 값이 됐다.
        #   ang_diff(a, 0) 이 곧 norm180(a) 다. 부호 있는 -180~+180 으로.
        self.body_y = ang_diff(float(r.get("playerBodyY", self.body_y) or 0.0), 0.0)
        self.body_z = ang_diff(float(r.get("playerBodyZ", self.body_z) or 0.0), 0.0)
        self.turret_x = float(r.get("playerTurretX", self.turret_x) or 0.0)
        self.turret_y = float(r.get("playerTurretY", self.turret_y) or 0.0)
        self.enemy_body_x = float(r.get("enemyBodyX", self.enemy_body_x) or 0.0)
        self.my_hp = r.get("playerHealth", self.my_hp)
        self.enemy_hp = r.get("enemyHealth", self.enemy_hp)
        self.my_speed = float(r.get("playerSpeed", 0.0) or 0.0)
        self.enemy_speed = float(r.get("enemySpeed", 0.0) or 0.0)

    @property
    def ready(self):
        return self.my is not None and self.enemy is not None

    @property
    def dist(self):
        if not self.ready:
            return None
        return math.hypot(self.enemy[0] - self.my[0], self.enemy[2] - self.my[2])

    @property
    def my_aspect(self):
        """적 기준 우리 방향의 상대각. |a|>120 이면 우리가 적의 후방에 있다."""
        if not self.ready:
            return None
        b = bearing(self.enemy, self.my)
        return ((b - self.enemy_body_x + 180.0) % 360.0) - 180.0


# ══════════════════════════════════════════════════════════
# 2. 기동 (재장전 시간에만 움직인다)
# ══════════════════════════════════════════════════════════
class Maneuver:
    FLANK_ENTER = 110.0
    FLANK_EXIT = 145.0

    def __init__(self, bal: Ballistics):
        self.bal = bal
        self.state = "IDLE"
        self.orbit_dir = 1
        self.body_rate = 0.0
        self.goal = None
        self.dt = CTRL_DT
        self.opt_range = CFG["band_near"]
        self.opt_p = 0.0
        self._flanking = True
        self._stuck_t = 0.0
        self._unstick_until = 0.0

    def band(self, v_cross: float = 0.0, fc=None, trk=None):
        """예상 명중확률이 최대가 되는 거리를 찾아 그 주변을 교전 밴드로 삼는다 (B8)"""
        rmin, rmax = self.bal.min_range(), self.bal.max_range()[0]
        s = fc.last_solution if fc else None
        hl = s.half_lat if (s and s.valid) else 2.6
        hn = s.half_lon if (s and s.valid) else 2.6
        opt, self.opt_p = optimal_range(
            self.bal, trk, hl, hn, v_cross,
            fc.t if fc else TurretParams(),
            duty_max=CFG["track_duty_max"],
            lo=CFG["band_near"], hi=CFG["band_far"])
        self.opt_range = opt
        half = CFG["band_half"]
        near = max(opt - half, rmin * 1.20)
        far = min(opt + half, rmax * 0.92)
        if far <= near + 6.0:
            far = near + 10.0
        return near, far

    def stop_cmd(self):
        self.body_rate = 0.0
        return ({"command": "STOP", "weight": 1.0},
                {"command": "", "weight": 0.0}, 0.0)

    def compute(self, tm: Telemetry, v_cross: float = 0.0, fc=None, trk=None):
        if not tm.ready:
            self.state = "IDLE"
            return self.stop_cmd()

        mx, _, mz = tm.my
        d = tm.dist
        near, far = self.band(v_cross, fc, trk)
        to_enemy = bearing(tm.my, tm.enemy)
        aspect = tm.my_aspect or 0.0

        if self._flanking and abs(aspect) >= self.FLANK_EXIT:
            self._flanking = False
        elif not self._flanking and abs(aspect) <= self.FLANK_ENTER:
            self._flanking = True

        if tm.t is not None:
            if self.state in ("FLANK", "ORBIT") and tm.my_speed < 0.4:
                self._stuck_t += self.dt
            else:
                self._stuck_t = 0.0
            if self._stuck_t > 2.5 and tm.t > self._unstick_until:
                self._unstick_until = tm.t + 3.0
                self.orbit_dir *= -1
                self._stuck_t = 0.0
        unstick = tm.t is not None and tm.t < self._unstick_until

        out = not (MAP_MARGIN < mx < 300 - MAP_MARGIN and
                   MAP_MARGIN < mz < 300 - MAP_MARGIN)

        if unstick:
            self.state = "UNSTICK"
            tgt_yaw = to_enemy + 90.0 * self.orbit_dir
            drive, w = "W", 1.0
        elif out:
            self.state = "RECENTER"
            self.orbit_dir *= -1
            tgt_yaw = bearing(tm.my, (150.0, 0.0, 150.0))
            drive, w = "W", 1.0
        elif d > far:
            self.state = "APPROACH"
            tgt_yaw = to_enemy
            drive, w = "W", 1.0
        elif d < near:
            self.state = "WITHDRAW"
            tgt_yaw = to_enemy
            drive, w = "S", 0.7
        elif CFG["flank"] and self._flanking:
            self.state = "FLANK"
            r = (near + far) * 0.5
            cur_ang = bearing(tm.enemy, tm.my)
            rear_ang = tm.enemy_body_x + 180.0
            delta = ang_diff(rear_ang, cur_ang)
            self.orbit_dir = 1 if delta > 0 else -1
            step = max(-55.0, min(55.0, delta))
            ga = math.radians(cur_ang + step)
            gx = tm.enemy[0] + math.sin(ga) * r
            gz = tm.enemy[2] + math.cos(ga) * r
            tgt_yaw = bearing(tm.my, (gx, 0.0, gz))
            drive, w = "W", 0.7
        else:
            self.state = "ORBIT"
            r = (near + far) * 0.5
            cur_ang = bearing(tm.enemy, tm.my)
            ga = math.radians(cur_ang + 45.0 * self.orbit_dir)
            gx = tm.enemy[0] + math.sin(ga) * r
            gz = tm.enemy[2] + math.cos(ga) * r
            tgt_yaw = bearing(tm.my, (gx, 0.0, gz))
            drive, w = "W", 0.7

        err = ang_diff(tgt_yaw, tm.body_x)
        ad = {"command": "", "weight": 0.0}
        rate = 0.0
        if abs(err) > BODY_DEADBAND:
            wt = min(1.0, max(0.1, abs(err) / (YAW_RATE * self.dt)))
            ad = {"command": "D" if err > 0 else "A", "weight": round(wt, 2)}
            rate = YAW_RATE * wt * (1.0 if err > 0 else -1.0)

        if abs(err) < 45.0:
            ws = {"command": drive, "weight": w}
        elif abs(err) < 110.0:
            ws = {"command": drive, "weight": round(w * 0.45, 2)}
        else:
            ws = {"command": "STOP", "weight": 1.0}
        self.body_rate = rate
        return ws, ad, rate


# ══════════════════════════════════════════════════════════
# 3. 사격 기록 + 자기 보정 되먹임
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# 3-a. 사거리 스윕  (v7 신규, B15)
# ══════════════════════════════════════════════════════════
#
# 무엇을 하는가
#     "서로 정지한 상태에서 사거리별로 쏴 보고 제원을 뽑는다."
#     지정한 사거리 목록을 순서대로 돌면서, 각 사거리에서
#     전차를 그 거리로 몰고 가 정지시킨 뒤 N 발 쏘고 다음으로 넘어간다.
#
# 왜 필요한가
#     평소 교전에서는 봇이 P(hit) 이 최대가 되는 거리를 스스로 고른다.
#     그래서 로그의 사거리 분포가 좁은 구간(26~40 m)에 몰린다.
#     실제로 2026-08-05 실사격 34발 중 55 m 초과가 1 발뿐이었다.
#     "사거리 100 m 에서 명중률이 얼마인가"를 답하려면
#     거리를 '골라서' 쏘게 강제해야 한다.
#
#     여기서 나오는 데이터가 있어야
#       · 사거리별 명중률 곡선
#       · 사거리별 계통 편향 (탄도 모델 잔차가 거리에 따라 어떻게 변하는가)
#       · 사거리별 산포 (조준 분해능이 어디서 한계에 걸리는가)
#     를 근거로 말할 수 있다.
class RangeSweep:
    def __init__(self):
        self.active = False
        self.ranges = []
        self.idx = 0
        self.shots_per = 3
        self.band_half = 2.0        # 목표 사거리 ± 이 폭을 교전 밴드로
        self._base_shots = 0        # 현재 사거리 시작 시점의 누적 사격수
        self.result = []            # [(사거리, 사격, 명중)]
        self._hits_at_start = 0

    def start(self, lo, hi, step, shots, band_half=2.0):
        # 시험이 끝나면 되돌릴 값들을 보관한다
        self._saved = {k: CFG[k] for k in ("band_near", "band_far",
                                           "band_half", "flank")}
        self.ranges = []
        r = float(lo)
        while r <= float(hi) + 1e-9:
            self.ranges.append(round(r, 1))
            r += float(step)
        self.idx = 0
        self.shots_per = int(shots)
        self.band_half = float(band_half)
        self.result = []
        self.active = bool(self.ranges)
        return self.active

    def stop(self):
        self.active = False
        for k, v in getattr(self, "_saved", {}).items():
            CFG[k] = v

    @property
    def target(self):
        if not self.active or self.idx >= len(self.ranges):
            return None
        return self.ranges[self.idx]

    def begin_range(self, fired, hits):
        self._base_shots = fired
        self._hits_at_start = hits

    def step_if_done(self, fired, hits):
        """이 사거리에서 목표 발수를 채웠으면 다음으로 넘어간다."""
        if not self.active:
            return False
        n = fired - self._base_shots
        if n < self.shots_per:
            return False
        self.result.append((self.ranges[self.idx], n, hits - self._hits_at_start))
        self.idx += 1
        if self.idx >= len(self.ranges):
            self.stop()          # CFG 원복까지 함께
            return True
        self.begin_range(fired, hits)
        return True

    def progress(self):
        if not self.ranges:
            return "미실행"
        if not self.active:
            return f"완료 ({len(self.result)}/{len(self.ranges)} 구간)"
        return (f"{self.idx + 1}/{len(self.ranges)} 구간 · 목표 {self.target:.0f} m "
                f"· {self.shots_per}발")


sweep = RangeSweep()


# ══════════════════════════════════════════════════════════
# 3-b. 틱 로그  (v7 신규, B13)
# ══════════════════════════════════════════════════════════
#
# 왜 필요한가
#     기존 사격 로그(shots_v6.csv)는 '발사 순간의 스냅샷'이다.
#     그래서 빗나간 발을 봐도 다음 두 가지를 구분할 수 없다.
#
#       (a) 조준이 덜 끝났는데 쐈다      -> 데드밴드/정렬 문제
#       (b) 조준은 맞았는데 예측이 틀렸다 -> 표적 추정 문제
#
#     둘은 처방이 정반대다. (a)는 데드밴드를 좁혀야 하고,
#     (b)는 좁혀 봐야 사격 기회만 줄고 명중률은 그대로다.
#     실제로 v7 튜닝 때 데드밴드를 5배 좁혀도 평균 사거리 오차가
#     +6.7 -> +7.2 m 로 변하지 않아 (b)임을 알았는데,
#     그 판단에 매 틱의 조준 오차 궤적이 있었다면 훨씬 빨랐다.
#
#     틱 로그는 매 /get_action 마다 '조준 오차가 지금 얼마이고
#     데드밴드가 얼마인지'를 남긴다. 발사 직전 몇 틱을 보면
#     조준이 수렴해서 쏜 건지, 인내 로직에 밀려 쏜 건지 알 수 있다.
TICK_HDR = [
    "sim_time", "phase", "fc_state",
    # 교전 상황
    "dist", "v_cross", "band_near", "band_far", "opt_range",
    "reloading", "in_env", "in_band", "trackable", "want_shoot",
    # 조준 상태  <- 진단의 핵심
    "turret_x", "turret_y", "aim_bearing", "aim_elev",
    "yaw_err", "pitch_err", "yaw_db", "pitch_db",
    "aligned_yaw", "aligned_pitch",
    # 사격 판정
    "p_hit", "p_threshold", "track_duty",
    "sig_lat", "sig_lon", "half_lat", "half_lon",
    "lon_short", "lon_long", "lon_shift", "drdt", "tof", "lead",
    # 자세 · 표적
    "my_x", "my_z", "body_x", "my_speed", "body_rate", "hull_settled",
    "enemy_x", "enemy_z", "enemy_body", "enemy_speed",
    # 출력 명령
    "qe_cmd", "qe_w", "rf_cmd", "rf_w", "fire",
    # 보정 · 환경
    "bias_range", "bias_bearing", "ctrl_dt",
    # ── B22 (2026-08-12) 맨 뒤 4열 추가 ────────────────────────
    #   맨 뒤에 붙여야 예전 틱 로그 분석이 안 깨진다 (shots.csv 의 dy 와 같은 방식).
    #   전부 읽기 전용이며 조준·판정 로직은 건드리지 않았다.
    "body_y",     # 차체 앞뒤 기울기 [deg]. 평지면 0
    "body_z",     # 차체 좌우 기울기(롤) [deg]. 평지면 0  <- 핵심
    "dy",         # 고저차. shots.csv 에만 있어서 '쏜 발'만 보였다. 선택 편향 제거용
    "mv_state",   # 기동 상태 APPROACH/WITHDRAW/ORBIT/FLANK/RECENTER/UNSTICK
]


class TickLog:
    """매 틱 상태를 메모리에 쌓아 두고 /save 때 CSV 로 내보낸다."""
    MAX = 250_000          # 0.41 s 기준 약 28 시간. 사실상 무제한.

    def __init__(self):
        self.rows = []
        self.dropped = 0

    def add(self, row):
        if len(self.rows) >= self.MAX:
            self.dropped += 1
            return
        self.rows.append(row)

    def clear(self):
        self.rows = []
        self.dropped = 0


def _r(v, n=3):
    """None 안전 반올림"""
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


class ShotLog:
    # v8: 120 -> 2000.
    #   2026-08-06 측정에서 211발을 쐈는데 상한 때문에 91발이 잘려나갔다.
    #   조건을 바꿔가며 누적 측정하면 앞 조건이 통째로 사라진다.
    MAX = 2000
    T_MIN, T_MAX = 0.15, 4.0      # 비행시간 대비 허용 배수
    MATCH_R = 45.0                # 조준점 반경 [m] (넉넉히 - 한 번에 한 발만 난다)

    def __init__(self):
        self.records = []
        self.fired = 0
        self.tank_hits = 0
        self.obstacle_hits = 0
        self.terrain_hits = 0
        self.incoming = 0
        self.unmatched = 0
        self.pending = None
        self.zone_stat = {"front": [0, 0.0], "side": [0, 0.0], "rear": [0, 0.0]}
        # B7 조건별 집계
        self.by_cond = {}          # key -> [fired, hits]

    def _cond_key(self, moving, dist):
        m = "이동" if moving else "정지"
        b = "≤35m" if dist <= 35 else ("35-50m" if dist <= 50 else
                                       ("50-70m" if dist <= 70 else ">70m"))
        return f"{m} {b}"

    def on_fire(self, tm: Telemetry, sol, trk):
        self.fired += 1
        moving = (tm.enemy_speed or 0.0) > 0.6 or trk.speed > 0.6
        key = self._cond_key(moving, sol.distance)
        self.by_cond.setdefault(key, [0, 0])[0] += 1
        self.pending = {
            "id": self.fired,
            "time": round(tm.t, 2) if tm.t is not None else 0.0,
            "t_raw": tm.t,
            "fire_pos": tuple(round(v, 2) for v in tm.my),
            "target_pos": tuple(round(v, 2) for v in tm.enemy),
            "enemy_body": round(tm.enemy_body_x, 1),
            "aim_point": tuple(round(v, 2) for v in sol.aim_point),
            "expect": (sol.aim_point[0], sol.aim_point[2]),
            "dist": round(sol.distance, 1),
            "aim": (round(sol.bearing, 2), round(sol.elevation, 2)),
            "turret": (round(tm.turret_x, 2), round(tm.turret_y, 2)),
            "tof": round(sol.flight, 2), "tof_raw": sol.flight,
            "lead": round(sol.lead, 1),
            "p_hit": round(sol.p_hit, 3),
            "sig": (round(sol.sig_lat, 2), round(sol.sig_lon, 2)),
            "half": (round(sol.half_lat, 2), round(sol.half_lon, 2)),
            "lon_win": (round(sol.lon_short, 2), round(sol.lon_long, 2)),
            "lon_shift": round(sol.lon_shift, 2),
            "own_speed": round(tm.my_speed, 2),
            "enemy_speed": round(tm.enemy_speed, 2),
            "cond": key, "moving": moving,
            # B15: 사거리 스윕의 목표 사거리 (평소에는 None)
            "sweep_range": sweep.target if sweep.active else None,
            # 교전 형태 라벨 - 나중에 기동포격 분류에 쓴다
            #   정지-정지 / 정지-이동(적만) / 이동-정지(아군만) / 이동-이동
            "engage": ("이동" if (tm.my_speed or 0.0) > 0.2 else "정지") + "-" +
                      ("이동" if moving else "정지"),
            "impact": None, "miss": None, "range_err": None, "cross_err": None,
            "aspect": None, "zone": None, "damage": None,
            "result": "비행중", "kind": "pending",
        }

    def _is_ours(self, ix, iz, now):
        p = self.pending
        if p is None or ix is None:
            return False
        if now is not None and p["t_raw"] is not None and p["tof_raw"]:
            dt = now - p["t_raw"]
            if not (self.T_MIN * p["tof_raw"] <= dt <= self.T_MAX * p["tof_raw"] + 1.5):
                return False
        ex, ez = p["expect"]
        return math.hypot(ix - ex, iz - ez) <= self.MATCH_R

    def on_impact(self, d, now=None, enemy_body=0.0, bias=None):
        ix, iy, iz = d.get("x"), d.get("y"), d.get("z")
        raw = str(d.get("hit", "")).lower()
        pos = (round(ix, 2), round(iy, 2), round(iz, 2)) if ix is not None else None

        if raw == "player":
            self.incoming += 1
            self.records.insert(0, {
                "id": "-", "time": round(now, 2) if now else "-",
                "fire_pos": None, "target_pos": None, "dist": "-",
                "aim": None, "tof": "-", "p_hit": "-", "own_speed": "-",
                "enemy_speed": "-", "impact": pos, "miss": None, "cond": "-",
                "range_err": None, "cross_err": None,
                "aspect": None, "zone": None, "damage": None,
                "result": "피격 (적탄)", "kind": "incoming"})
            del self.records[self.MAX:]
            return

        if not self._is_ours(ix, iz, now):
            self.unmatched += 1
            return

        rec = self.pending
        if "enemy" in raw or "tank" in raw:
            kind, label = "tank", "명중"
            self.tank_hits += 1
            self.by_cond.setdefault(rec["cond"], [0, 0])[1] += 1
            if pos and rec.get("fire_pos"):
                asp = impact_aspect(rec["fire_pos"], pos, enemy_body)
                rec["aspect"] = round(asp, 1)
                rec["zone"] = aspect_zone(asp)
        elif "terrain" in raw or "ground" in raw:
            kind, label = "terrain", "지면"
            self.terrain_hits += 1
        else:
            kind, label = "obstacle", "장애물"
            self.obstacle_hits += 1

        rec["impact"] = pos
        fp, ap = rec.get("fire_pos"), rec.get("aim_point")
        if pos and ap:
            rec["miss"] = round(math.hypot(ix - ap[0], iz - ap[2]), 2)
        if pos and fp and ap:
            # 시선 좌표계 분해 (종=사거리, 횡=방위)
            aim_b = bearing(fp, ap)
            aim_d = math.hypot(ap[0] - fp[0], ap[2] - fp[2])
            imp_b = bearing(fp, (ix, iy, iz))
            imp_d = math.hypot(ix - fp[0], iz - fp[2])
            rec["range_err"] = round(imp_d - aim_d, 2)
            rec["cross_err"] = round(math.radians(ang_diff(imp_b, aim_b)) * max(1.0, aim_d), 2)
            # B4/B10: 자기 보정 되먹임 - **지면 착탄만** 학습한다.
            #
            # ** v6.1 에서 고친 치명적 버그 **
            #   전차에 명중한 탄의 착탄점은 '차체 표면'이라 조준점보다
            #   계통적으로 1~2 m 짧게 기록된다. 이것을 탄도 오차로 오해하면
            #   보정기가 '탄이 짧게 떨어진다'고 판단해 계속 사거리를 늘리고,
            #   결국 상한(-6 m)까지 폭주한다.
            #   실측(2026-08-04, 28발): 모든 사격이 약 6 m 길게 조준되고 있었다.
            #   지면 착탄만이 편향 없는 탄도 잔차를 준다.
            if bias is not None and CFG["bias"] and kind == "terrain":
                # C9 로 일부러 길게 민 양은 오차가 아니므로 빼고 학습한다
                bias.observe(fp, ap, (ix, iy, iz),
                             expected_long=rec.get("lon_shift", 0.0) or 0.0)

        rec["result"] = f"{label} ({raw})"
        rec["kind"] = kind
        self.records.insert(0, rec)
        del self.records[self.MAX:]
        self.pending = None

    def attach_damage(self, dmg: float):
        for r in self.records:
            if r["kind"] == "tank" and r.get("damage") is None:
                r["damage"] = round(dmg, 1)
                z = r.get("zone")
                if z in self.zone_stat:
                    self.zone_stat[z][0] += 1
                    self.zone_stat[z][1] += dmg
                return

    @property
    def hit_rate(self):
        return (self.tank_hits / self.fired * 100.0) if self.fired else 0.0

    @property
    def mean_miss(self):
        v = [r["miss"] for r in self.records
             if isinstance(r.get("miss"), (int, float)) and r["kind"] != "incoming"]
        return sum(v) / len(v) if v else None

    def bias_report(self):
        rs = [r["range_err"] for r in self.records
              if isinstance(r.get("range_err"), (int, float))]
        cs = [r["cross_err"] for r in self.records
              if isinstance(r.get("cross_err"), (int, float))]
        if not rs:
            return None
        n = len(rs)
        mr, mc = sum(rs) / n, sum(cs) / n
        sr = math.sqrt(sum((x - mr) ** 2 for x in rs) / n)
        sc = math.sqrt(sum((x - mc) ** 2 for x in cs) / n)
        return n, mr, sr, mc, sc

    def zone_summary(self):
        out = []
        for z, ko in (("front", "전면"), ("side", "측면"), ("rear", "후면")):
            n, s = self.zone_stat[z]
            out.append((ko, n, s / n if n else None))
        return out


# ══════════════════════════════════════════════════════════
# 4. 봇 - 정지 사격 상태기계 (B1)
# ══════════════════════════════════════════════════════════
class Bot:
    def __init__(self):
        self.tm = Telemetry()
        self.bal = Ballistics()
        self.bias = BiasEstimator()
        self.fc = FireControl(self.bal, TurretParams(), reload_s=6.6,
                              target=TargetSize(), bias=self.bias,
                              p_hit_min=CFG["p_hit_min"],
                              settle_ticks=CFG["settle_ticks"],
                              db_safety=CFG["db_safety"],
                              track_duty_max=CFG["track_duty_max"],
                              patience=CFG["patience"],
                              p_hit_floor=CFG["p_hit_floor"],
                              lon_gain=CFG["lon_gain"],
                              tof_max=CFG["tof_max"],
                              lon_long_max=CFG["lon_long_max"],
                              body_rate_max=CFG["body_rate_max"])
        self.mv = Maneuver(self.bal)
        self.trk = TargetTracker(limits=MotionLimits())
        self.log = ShotLog()
        self.ticks = TickLog()
        self._prev_enemy_hp = None
        self.enemy = EnemyBot(behavior=ENEMY_BEHAVIOR, keep_r=ENEMY_KEEP_R,
                              mask=ENEMY_MASK, speed_cap=ENEMY_SPEED_CAP)
        self._last_act_t = None
        self.ctrl_dt = CTRL_DT
        self._dt_samples = []   # v7 B12: 중앙값용 표본
        self.phase = "MOVE"
        self._halt_since = None
        self._aim_since = None
        self.v_cross = 0.0

    def _measure_dt(self, t):
        """
        실제 /get_action 주기를 측정해 제어 상수에 반영.

        ── v7 변경: 지수이동평균 -> 최근 200 표본의 중앙값 (B12) ──
        제어 주기는 PC 마다 다르다(실측 0.140 / 0.235 / 0.410 s).
        EMA 는 한 번의 렉(예: GC 정지로 1.5 s)에도 크게 끌려가고
        원래 값으로 돌아오는 데 수십 틱이 걸린다.
        dt 는 포탑 weight 계산의 분모라, 과대평가되면 조준이 폭주한다.
        중앙값은 이런 이상치에 구조적으로 둔감하다.
        (파트너 하네스 v2 의 measure_ctrl_dt 와 같은 방식)
        """
        if t is None:
            return
        if self._last_act_t is not None:
            dt = t - self._last_act_t
            if 0.02 < dt < 3.0:
                self._dt_samples.append(dt)
                del self._dt_samples[:-200]
                s = sorted(self._dt_samples)
                self.ctrl_dt = s[len(s) // 2]
                self.fc.t.dt = self.ctrl_dt
                self.mv.dt = self.ctrl_dt
        self._last_act_t = t

    def reset(self):
        self.tm = Telemetry()
        self.trk = TargetTracker(limits=MotionLimits())
        self.fc.last_fire_t = None
        self.fc.state = "IDLE"
        self.fc._settled = 0
        self._prev_enemy_hp = None
        self.enemy = EnemyBot(behavior=ENEMY_BEHAVIOR, keep_r=ENEMY_KEEP_R,
                              mask=ENEMY_MASK, speed_cap=ENEMY_SPEED_CAP)
        self._last_act_t = None
        self.ctrl_dt = CTRL_DT
        self._dt_samples = []   # v7 B12: 중앙값용 표본
        self.phase = "MOVE"
        self._halt_since = None
        self._aim_since = None
        self.v_cross = 0.0
        # 바이어스와 사격 통계는 에피소드를 넘겨 누적한다 (학습 유지)

    def apply_cfg(self):
        self.fc.p_hit_min = CFG["p_hit_min"]
        self.fc.settle_ticks = CFG["settle_ticks"]
        self.fc.db_safety = CFG["db_safety"]
        self.fc.track_duty_max = CFG["track_duty_max"]
        self.fc.patience = CFG["patience"]
        self.fc.p_hit_floor = CFG["p_hit_floor"]
        self.fc.lon_gain = CFG["lon_gain"]
        self.fc.pitch_db_max = CFG["pitch_db_max"]   # B14
        self.fc.tof_max = CFG["tof_max"]             # C11
        self.fc.lon_long_max = CFG["lon_long_max"]   # C12
        self.fc.body_rate_max = CFG["body_rate_max"] # C13

    def on_info(self, d):
        self.tm.update(d)
        tm = self.tm
        if tm.ready and tm.t is not None:
            self.trk.update(tm.t, tm.enemy)
        if tm.ready:
            self.enemy.share_world(tm.my)
            self.enemy.set_body(tm.enemy_body_x)
            self.enemy.set_time(tm.t)
        hp = tm.enemy_hp
        if hp is not None and self._prev_enemy_hp is not None and hp < self._prev_enemy_hp:
            self.log.attach_damage(self._prev_enemy_hp - hp)
        if hp is not None:
            self._prev_enemy_hp = hp

    # ── 상태기계 ─────────────────────────────────────────
    def act(self, payload):
        self.tm.update(payload)
        tm = self.tm
        self._measure_dt(tm.t)
        if not tm.ready:
            return neutral()
        if tm.t is not None:
            self.trk.update(tm.t, tm.enemy)

        self.apply_cfg()
        now = tm.t or 0.0
        d = tm.dist

        # B15: 사거리 스윕이 켜져 있으면 교전 밴드를 목표 사거리에 고정한다.
        #      (평소에는 봇이 P(hit) 최대 거리를 스스로 고르므로
        #       로그의 사거리 분포가 좁은 구간에 몰린다)
        if sweep.active:
            sweep.step_if_done(self.log.fired, self.log.tank_hits)
            tr = sweep.target
            if tr is not None:
                # 교전 밴드를 목표 사거리에 '붙인다'.
                #   band_half 를 평소 9 m 로 두면 기동 계층이 목표 ±9 m 를
                #   유효 구간으로 보고 그 안을 배회한다. 그러면 엄격 게이트
                #   (목표 ±2 m)에 걸려 한 발도 못 쏘고 시간만 간다.
                #   시험 중에는 밴드 자체를 좁혀 목표 거리를 유지하게 한다.
                CFG["band_near"] = max(22.0, tr - sweep.band_half)
                CFG["band_far"] = tr + sweep.band_half
                CFG["band_half"] = sweep.band_half
                CFG["flank"] = False        # 후방 침투 대신 거리 유지만

        # 표적의 시선 횡단 속도 -> 최적 교전 거리와 추적 가능성을 결정 (B8)
        los = math.radians(bearing(tm.my, tm.enemy))
        cx, cz = math.cos(los), -math.sin(los)
        v_cross = abs(self.trk.vel[0] * cx + self.trk.vel[2] * cz)
        self.v_cross = v_cross
        near, far = self.mv.band(v_cross, self.fc, self.trk)

        reloading = (self.fc.last_fire_t is not None
                     and now - self.fc.last_fire_t < self.fc.reload_s)
        in_band = (near - 3.0) <= d <= (far + 8.0)
        in_env = self.bal.min_range() + 1.0 <= d <= self.bal.max_range()[0] - 1.0
        trackable = required_yaw_rate(v_cross, d) <= YAW_RATE * CFG["track_duty_max"]

        # 사격할 수 있는 상황인가
        want_shoot = (not reloading) and in_env and in_band and trackable

        # B15: 사거리 스윕 중에는 밴드 판정을 엄격하게 잠근다.
        #   평소 in_band 는 (near-3) ~ (far+8) 로 느슨하다. 교전에서는
        #   기회를 놓치지 않으려는 의도지만, 사거리 시험에서는 그 슬랙이
        #   그대로 오염이 된다 (목표 60 m 인데 54.9 m 에서 발사되는 식).
        #   시험 중에는 목표 사거리 ± band_half 안에서만 쏜다.
        if sweep.active and sweep.target is not None:
            want_shoot = want_shoot and abs(d - sweep.target) <= sweep.band_half

        if CFG["moving_fire"]:
            self.phase = "MOVE"
        elif not want_shoot:
            self.phase = "MOVE"
            self._halt_since = None
            self._aim_since = None
        else:
            if self.phase == "MOVE":
                self.phase = "HALT"
                self._halt_since = now
            if self.phase == "HALT":
                slow = tm.my_speed <= CFG["halt_speed"]
                timeout = (self._halt_since is not None
                           and now - self._halt_since > CFG["halt_timeout"])
                if slow or timeout:
                    self.phase = "AIM"
                    self._aim_since = now
            elif self.phase == "AIM":
                # 조준이 계속 안 끝나면 자세/거리를 바꾸러 나간다 (교착 방지)
                if (self._aim_since is not None
                        and now - self._aim_since > CFG["aim_timeout"]):
                    self.phase = "MOVE"
                    self._halt_since = self._aim_since = None

        # B15: 사거리 시험 중에는 교전 기동을 쓰지 않는다.
        #
        #   평소 기동 계층은 P(hit) 이 최대가 되는 지점을 스스로 찾아
        #   표적 주위를 선회하고 후방으로 파고든다. 교전에서는 옳지만,
        #   사거리 시험에서는 (1) 목표 거리를 지키지 못하고
        #   (2) 매 발 자세각(aspect)이 달라져 변수가 하나 더 늘어난다.
        #   시험은 조건을 고정해야 한다. 표적을 정면으로 보고 거리만 맞춘다.
        if sweep.active and sweep.target is not None and not CFG["moving_fire"]:
            ws, ad, body_rate, hull_settled, my_vel = self._sweep_move(tm, d, sweep.target)
        elif self.phase in ("HALT", "AIM"):
            ws = {"command": "STOP", "weight": 1.0}
            ad = {"command": "", "weight": 0.0}
            body_rate = 0.0
            hull_settled = tm.my_speed <= CFG["halt_speed"]
            my_vel = (0.0, 0.0, 0.0)
        else:
            ws, ad, body_rate = self.mv.compute(tm, v_cross, self.fc, self.trk)
            sgn = -1.0 if ws["command"] == "S" else (1.0 if ws["command"] == "W" else 0.0)
            rad = math.radians(tm.body_x)
            my_vel = (math.sin(rad) * tm.my_speed * sgn, 0.0,
                      math.cos(rad) * tm.my_speed * sgn)
            hull_settled = (tm.my_speed <= CFG["halt_speed"]
                            and abs(body_rate) < 1e-6)

        # 사격 통제
        prev_fire_t = self.fc.last_fire_t
        turret = self.fc.update(
            my_pos=tm.my, turret_x=tm.turret_x, turret_y=tm.turret_y,
            target_pos=tm.enemy, tracker=self.trk,
            target_heading=(tm.enemy_body_x if tm.enemy_speed and tm.enemy_speed > 0.4
                            else None),
            sim_time=now, hull_settled=hull_settled, my_vel=my_vel,
            body_rate_dps=body_rate,
            allow_moving_fire=CFG["moving_fire"],
            # B15: 스윕 중 목표 사거리를 벗어나면 하드 금지
            inhibit_fire=(sweep.active and not want_shoot))

        # 경험적 예측오차 학습용 예약 (C4)
        sol = self.fc.last_solution
        if sol and sol.valid:
            self.trk.enqueue_prediction(sol.flight + self.ctrl_dt)

        fired_now = turret.get("fire") and self.fc.last_fire_t != prev_fire_t
        if fired_now:
            self.log.on_fire(tm, sol, self.trk)
            self.phase = "MOVE"
            self._halt_since = None

        # ── B13: 틱 로그 ─────────────────────────────────
        # phase 를 MOVE 로 되돌리기 '전' 값을 남겨야 발사 시점 상태가 보인다.
        if TICK_LOG:
            self._log_tick(tm, sol, turret, ws, ad, d, v_cross, near, far,
                           reloading, in_env, in_band, trackable, want_shoot,
                           body_rate, hull_settled, now,
                           phase_at_fire=("AIM" if fired_now else self.phase),
                           fired=bool(fired_now))

        return {"moveWS": ws, "moveAD": ad,
                "turretQE": turret["turretQE"], "turretRF": turret["turretRF"],
                "fire": turret["fire"]}

    def _sweep_move(self, tm, d, target_r, tol=0.7):
        """
        B15: 사거리 시험용 단순 기동.
            표적을 정면으로 보고, 목표 사거리까지 전진/후진한 뒤 정지.
            선회·후방 침투 없음. 매 발의 기하를 동일하게 유지한다.
        반환 (ws, ad, body_rate, hull_settled, my_vel)
        """
        stop = {"command": "STOP", "weight": 1.0}
        none = {"command": "", "weight": 0.0}

        # 차체를 표적 방향으로 정렬 (조준과 무관하지만 주행 방향을 정한다)
        to_e = bearing(tm.my, tm.enemy)
        berr = ang_diff(to_e, tm.body_x)
        ad, body_rate = none, 0.0
        if abs(berr) > BODY_DEADBAND:
            w = min(1.0, max(0.05, abs(berr) / (YAW_RATE * self.ctrl_dt)))
            ad = {"command": "D" if berr > 0 else "A", "weight": round(w, 3)}
            body_rate = YAW_RATE * w * (1.0 if berr > 0 else -1.0)

        err = d - target_r                     # +면 너무 멀다
        if abs(err) <= tol:
            ws = stop
        elif abs(berr) > 25.0:
            ws = stop                          # 차체가 안 돌았으면 먼저 돌린다
        else:
            # 남은 거리에 비례한 속도. 가까워지면 살살 (오버슈트 방지)
            w = min(1.0, max(0.15, abs(err) / 12.0))
            ws = {"command": "W" if err > 0 else "S", "weight": round(w, 2)}

        moving = ws["command"] != "STOP" or ad["command"] != ""
        settled = (abs(tm.my_speed) <= CFG["halt_speed"]
                   and abs(body_rate) < 1e-6)
        if moving:
            r = math.radians(tm.body_x)
            sgn = -1.0 if ws["command"] == "S" else (1.0 if ws["command"] == "W" else 0.0)
            my_vel = (math.sin(r) * tm.my_speed * sgn, 0.0,
                      math.cos(r) * tm.my_speed * sgn)
        else:
            my_vel = (0.0, 0.0, 0.0)
        # 시험 중 상태 표시용
        self.phase = "AIM" if (abs(err) <= tol and settled) else "MOVE"
        return ws, ad, body_rate, settled, my_vel

    def _log_tick(self, tm, sol, turret, ws, ad, d, v_cross, near, far,
                  reloading, in_env, in_band, trackable, want_shoot,
                  body_rate, hull_settled, now, phase_at_fire, fired):
        fc = self.fc
        qe = turret.get("turretQE") or {}
        rf = turret.get("turretRF") or {}
        ye = getattr(fc, "yaw_err", None)
        pe = getattr(fc, "pitch_err", None)
        ydb = getattr(fc, "yaw_db", None)
        pdb = getattr(fc, "pitch_db", None)
        ok_y = (abs(ye) <= ydb) if (ye is not None and ydb is not None) else None
        ok_p = (abs(pe) <= pdb) if (pe is not None and pdb is not None) else None
        v = sol if (sol and sol.valid) else None
        self.ticks.add([
            _r(now, 2), phase_at_fire, fc.state,
            _r(d, 2), _r(v_cross, 2), _r(near, 1), _r(far, 1),
            _r(getattr(self.mv, "opt_range", None), 1),
            reloading, in_env, in_band, trackable, want_shoot,
            _r(tm.turret_x), _r(tm.turret_y),
            _r(v.bearing) if v else None, _r(v.elevation) if v else None,
            _r(ye), _r(pe), _r(ydb), _r(pdb), ok_y, ok_p,
            _r(v.p_hit) if v else None, _r(fc.threshold(now)),
            _r(getattr(fc, "track_duty", None)),
            _r(v.sig_lat, 2) if v else None, _r(v.sig_lon, 2) if v else None,
            _r(v.half_lat, 2) if v else None, _r(v.half_lon, 2) if v else None,
            _r(v.lon_short, 2) if v else None, _r(v.lon_long, 2) if v else None,
            _r(v.lon_shift, 2) if v else None, _r(v.drdt, 2) if v else None,
            _r(v.flight, 3) if v else None, _r(v.lead, 2) if v else None,
            _r(tm.my[0], 2), _r(tm.my[2], 2), _r(tm.body_x, 2),
            _r(tm.my_speed, 2), _r(body_rate, 2), hull_settled,
            _r(tm.enemy[0], 2), _r(tm.enemy[2], 2), _r(tm.enemy_body_x, 2),
            _r(tm.enemy_speed, 2),
            qe.get("command", ""), qe.get("weight", 0.0),
            rf.get("command", ""), rf.get("weight", 0.0),
            bool(turret.get("fire")),
            _r(self.bias.range_bias, 3), _r(self.bias.bearing_bias, 3),
            _r(self.ctrl_dt, 4),
            # B22: 차체 기울기 · 고저차 · 기동상태 (읽기 전용)
            _r(tm.body_y, 2), _r(tm.body_z, 2),
            (_r(tm.my[1] - v.aim_point[1], 2)
             if (v is not None and tm.my is not None) else None),
            getattr(self.mv, "state", None),
        ])


bot = Bot()


# ══════════════════════════════════════════════════════════
# 5. 엔드포인트
# ══════════════════════════════════════════════════════════
@app.route("/init", methods=["GET", "POST"])
def init():
    bot.reset()
    print("=== 에피소드 초기화 ===")
    return jsonify({
        "startMode": "start",
        "blStartX": START_BLUE[0], "blStartY": START_BLUE[1], "blStartZ": START_BLUE[2],
        "rdStartX": START_RED[0], "rdStartY": START_RED[1], "rdStartZ": START_RED[2],
        "trackingMode": True, "detectMode": False, "logMode": False,
        "stereoCameraMode": False,
        "enemyTracking": ENEMY_TRACKING,
        "saveSnapshot": False, "saveLog": False, "saveLidarData": False,
        "lux": 30000, "destoryObstaclesOnHit": False,
    })


@app.route("/start", methods=["GET", "POST"])
def start():
    return jsonify({"control": ""})


@app.route("/info", methods=["POST"])
def info():
    bot.on_info(request.get_json(force=True, silent=True) or {})
    return jsonify({"status": "success", "control": ""})


@app.route("/get_action", methods=["POST"])
def get_action():
    return jsonify(bot.act(request.get_json(force=True, silent=True) or {}))


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    bot.log.on_impact(request.get_json(force=True, silent=True) or {},
                      now=bot.tm.t, enemy_body=bot.tm.enemy_body_x, bias=bot.bias)
    return jsonify({"status": "OK"})


@app.route("/collision", methods=["POST"])
def collision():
    return jsonify({"status": "success"})


@app.route("/detect", methods=["POST"])
def detect():
    return jsonify([])


@app.route("/stereo_image", methods=["POST"])
def stereo():
    return jsonify({"result": "success"})


@app.route("/set_destination", methods=["POST"])
def set_dest():
    return jsonify({"status": "OK"})


@app.route("/update_obstacle", methods=["POST"])
def upd_obs():
    return jsonify({"status": "success"})


@app.route("/set")
def set_cfg():
    """런타임 튜닝 (B6)  예: /set?band_near=30&band_far=45&p_hit_min=0.7"""
    changed = []
    for k, v in request.args.items():
        if k == "behavior":
            global ENEMY_BEHAVIOR
            ENEMY_BEHAVIOR = v
            bot.enemy.behavior = v
            changed.append(f"behavior={v}")
            continue
        if k in CFG:
            cur = CFG[k]
            try:
                if v.lower() in ("none", "off", "null", ""):
                    # None 을 허용하는 항목 (예: lon_long_max, tof_max) 끄기
                    CFG[k] = None
                elif isinstance(cur, bool):
                    CFG[k] = v.lower() in ("1", "true", "on", "yes")
                elif cur is None:
                    CFG[k] = float(v)          # 꺼져 있던 항목을 켤 때
                else:
                    CFG[k] = type(cur)(v)
                changed.append(f"{k}={CFG[k]}")
            except ValueError:
                pass
    bot.apply_cfg()
    return ('<meta charset="utf-8">변경: ' + (", ".join(changed) or "없음") +
            ' <a href="/">돌아가기</a>')


_EXPORT_HDR = ["id", "time", "cond", "engage", "sweep_range", "dist", "miss", "range_err", "cross_err",
               "p_hit", "sig_lat", "sig_lon", "half_lat", "half_lon",
               "lon_short", "lon_long", "lon_shift",
               "aim_bearing", "aim_elev", "turret_x", "turret_y",
               "tof", "lead", "own_speed", "enemy_speed",
               "fire_x", "fire_z", "aim_x", "aim_z", "impact_x", "impact_z",
               "aspect", "zone", "damage", "result", "kind",
               # ── B-log (2026-08-12): 고저차 4열 ─────────────────────────
               #   왜 뒤에 붙이나
               #     analyze_shots.py 는 csv.DictReader + r.get() 로 읽는다.
               #     맨 뒤 추가는 기존 열 위치를 안 건드리므로 8/11 이전
               #     로그(열 없음)도 그대로 파싱된다.
               #   왜 지금 넣나
               #     지금 dy 는 시뮬레이터 정답이다. 라이더가 붙으면 추정값이
               #     된다. 그때 추정을 채점할 기준선이 없으면 기복 있는
               #     Forest 를 통째로 재측정해야 한다.
               #   비용
               #     fire_pos · aim_point · impact 는 이미 3-튜플로 메모리에
               #     있고 [1] 만 버리고 있었다. 새 계산 없음.
               #   주의
               #     dy 부호는 탄도 해와 반드시 같아야 한다.
               #     992행  dy = my[1] - aim[1]   (+ 면 내가 더 높다)
               "dy", "fire_y", "aim_y", "impact_y"]


def _export_row(r):
    am = r.get("aim") or (None, None)
    tu = r.get("turret") or (None, None)
    sg = r.get("sig") or (None, None)
    hf = r.get("half") or (None, None)
    lw = r.get("lon_win") or (None, None)
    fp = r.get("fire_pos") or (None, None, None)
    ap = r.get("aim_point") or (None, None, None)
    ip = r.get("impact") or (None, None, None)
    return [r.get("id"), r.get("time"), r.get("cond"),
            r.get("engage"), r.get("sweep_range"), r.get("dist"),
            r.get("miss"), r.get("range_err"), r.get("cross_err"),
            r.get("p_hit"), sg[0], sg[1], hf[0], hf[1],
            lw[0], lw[1], r.get("lon_shift"),
            am[0], am[1], tu[0], tu[1],
            r.get("tof"), r.get("lead"), r.get("own_speed"),
            r.get("enemy_speed"), fp[0], fp[2], ap[0], ap[2],
            ip[0], ip[2],
            r.get("aspect"), r.get("zone"), r.get("damage"),
            r.get("result"), r.get("kind"),
            # B-log: 고저차. 순수 기록이며 조준·판정에 전혀 쓰이지 않는다.
            (round(fp[1] - ap[1], 2)
             if (fp[1] is not None and ap[1] is not None) else None),
            fp[1], ap[1], ip[1]]


@app.route("/export")
def export():
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_EXPORT_HDR)
    for r in reversed(bot.log.records):
        w.writerow(_export_row(r))
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=shots_v6.csv"})


@app.route("/save")
def save_csv():
    """
    세 종류 로그를 한 번에 저장한다.
      logs/<날짜>_<태그>/shots.csv · speed.csv · ticks.csv
    """
    d = _log_dir()
    n_s = _write_csv(os.path.join(d, "shots.csv"), _EXPORT_HDR,
                     [_export_row(r) for r in reversed(bot.log.records)])
    msg = f'<b>shots.csv</b> {n_s}행'
    if TICK_LOG:
        n_p = _write_csv(os.path.join(d, "speed.csv"), SPEED_COLS,
                         [[r[i] for i in _SPEED_IDX] for r in bot.ticks.rows])
        n_t = _write_csv(os.path.join(d, "ticks.csv"), TICK_HDR, bot.ticks.rows)
        msg += f' · <b>speed.csv</b> {n_p}행 · <b>ticks.csv</b> {n_t}행'
        if bot.ticks.dropped:
            msg += f' (버림 {bot.ticks.dropped})'
    _write_session(d)
    return ('<meta charset="utf-8">저장 완료<br><b>'
            + os.path.relpath(d, _here()) + '/</b><br>' + msg +
            f'<br>명중률 {bot.log.hit_rate:.1f}%'
            '<br><br><a href="/clear_ticks">틱 로그 비우기</a>'
            ' (다음 조건 측정 전에 권장) · <a href="/">돌아가기</a>')


def _write_session(d):
    """
    이 로그가 어떤 설정에서 나왔는지 함께 남긴다.
    설정을 안 적어두면 몇 시간 뒤에 "이 파일이 뭐였지"가 된다.
    """
    import json
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": _tag_name(),
        "enemy_behavior": ENEMY_BEHAVIOR,
        "ctrl_dt_measured": round(bot.ctrl_dt, 4),
        "shots": bot.log.fired, "hits": bot.log.tank_hits,
        "hit_rate": round(bot.log.hit_rate, 2),
        "sweep": {"ranges": sweep.ranges, "shots_per": sweep.shots_per,
                  "result": sweep.result} if sweep.ranges else None,
        "cfg": dict(CFG),
    }
    with open(os.path.join(d, "session.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ── B16: 로그 종류별 개별 저장 ──────────────────────────────
#   사격 로그와 속도 로그는 보는 목적이 다르다.
#     사격 로그  발사 1발 = 1행. "왜 빗나갔나"
#     속도 로그  매 틱 1행, 속도·위치·자세만. "누가 언제 움직였나"
#                -> 기동포격 분류(정지-정지 / 이동-정지 / 정지-이동 / 이동-이동)의 근거
#     틱 로그    매 틱 1행, 54 컬럼 전부. "조준이 어떻게 수렴했나"
SPEED_COLS = ["sim_time", "phase", "dist",
              "my_x", "my_z", "body_x", "my_speed", "body_rate", "hull_settled",
              "enemy_x", "enemy_z", "enemy_body", "enemy_speed",
              "v_cross", "fire", "ctrl_dt"]
_SPEED_IDX = [TICK_HDR.index(c) for c in SPEED_COLS]


def _write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def _tag_name():
    return (request.args.get("tag", "").strip() or CURRENT_TAG or "run")


def _log_root():
    """
    로그 보관 위치.
      코드 폴더의 부모에 logs/ 가 있으면 거기에 (권장 구조)
          260805/
            ├─ v7/     코드
            └─ logs/   로그
      없으면 코드 폴더 안에 만든다 (단독 실행 대비)
    """
    up = os.path.abspath(os.path.join(_here(), os.pardir, "logs"))
    if os.path.isdir(up):
        return up
    return os.path.join(_here(), "logs")


def _log_dir(tag=None):
    """
    logs/<날짜>_<태그>/ 아래에 모은다.
    같은 태그로 다시 저장하면 덮어쓴다 (측정을 이어서 할 때 편하다).
    """
    tag = tag or _tag_name()
    d = os.path.join(_log_root(), f"{time.strftime('%Y%m%d')}_{tag}")
    os.makedirs(d, exist_ok=True)
    return d


@app.route("/save_shots")
def save_shots():
    """사격 로그만 저장 -> shots_v6_<tag>.csv"""
    d = _log_dir()
    n = _write_csv(os.path.join(d, "shots.csv"), _EXPORT_HDR,
                   [_export_row(r) for r in reversed(bot.log.records)])
    return ('<meta charset="utf-8">사격 로그 저장<br><b>'
            + os.path.relpath(d, _here()) + '/shots.csv</b> · ' + str(n) + '행<br>'
            f'명중률 {bot.log.hit_rate:.1f}%<br><br><a href="/">돌아가기</a>')


@app.route("/save_speed")
def save_speed():
    """속도 로그만 저장 -> speed_v6_<tag>.csv (틱 로그에서 이동 관련 컬럼만)"""
    if not TICK_LOG:
        return '<meta charset="utf-8">TICK_LOG 이 꺼져 있다 <a href="/">돌아가기</a>'
    d = _log_dir()
    rows = [[r[i] for i in _SPEED_IDX] for r in bot.ticks.rows]
    n = _write_csv(os.path.join(d, "speed.csv"), SPEED_COLS, rows)
    return ('<meta charset="utf-8">속도 로그 저장<br><b>'
            + os.path.relpath(d, _here()) + '/speed.csv</b> · ' + str(n) + '행'
            '<br><br><a href="/">돌아가기</a>')


@app.route("/save_ticks")
def save_ticks():
    """틱 로그 전체 저장 -> ticks_v6_<tag>.csv (54 컬럼)"""
    if not TICK_LOG:
        return '<meta charset="utf-8">TICK_LOG 이 꺼져 있다 <a href="/">돌아가기</a>'
    d = _log_dir()
    n = _write_csv(os.path.join(d, "ticks.csv"), TICK_HDR, bot.ticks.rows)
    return ('<meta charset="utf-8">틱 로그 저장<br><b>'
            + os.path.relpath(d, _here()) + '/ticks.csv</b> · ' + str(n) + '행'
            '<br><br><a href="/">돌아가기</a>')


# ── B15: 사거리 스윕 ────────────────────────────────────────
@app.route("/sweep")
def sweep_start():
    """
    사거리별 정지-정지 사격 시험.
      /sweep?start=25&end=120&step=5&shots=3
    적을 정지시키고, 지정한 사거리마다 그 거리로 이동->정지->N발 사격.
    """
    if request.args.get("stop"):
        sweep.stop()
        return '<meta charset="utf-8">스윕 중지 <a href="/">돌아가기</a>'
    lo = float(request.args.get("start", 25))
    hi = float(request.args.get("end", 120))
    st = float(request.args.get("step", 5))
    ns = int(request.args.get("shots", 3))
    bh = float(request.args.get("band_half", 2.0))
    if not sweep.start(lo, hi, st, ns, bh):
        return '<meta charset="utf-8">사거리 목록이 비었다 <a href="/">돌아가기</a>'
    sweep.begin_range(bot.log.fired, bot.log.tank_hits)
    # 적을 세운다 (정지-정지 시험)
    global ENEMY_BEHAVIOR
    ENEMY_BEHAVIOR = "static"
    bot.enemy.behavior = "static"
    est = len(sweep.ranges) * ns * 7.0 / 60.0
    return ('<meta charset="utf-8">사거리 스윕 시작<br>'
            f'구간 {len(sweep.ranges)}개 ({lo:.0f} ~ {hi:.0f} m, {st:.0f} m 간격) '
            f'· 구간당 {ns}발<br>'
            f'적 행동을 <b>static</b> 으로 바꿨다 (정지-정지 시험)<br>'
            f'예상 소요 <b>{est:.0f}분 이상</b> (이동 시간 별도)<br><br>'
            '<a href="/">돌아가기</a>')


@app.route("/sweep_result")
def sweep_result():
    if not sweep.result:
        return ('<meta charset="utf-8">아직 완료된 구간이 없다 '
                '<a href="/">돌아가기</a>')
    rows = "".join(
        f"<tr><td>{r:.0f} m</td><td>{n}</td><td>{h}</td>"
        f"<td>{(h/n*100 if n else 0):.1f}%</td></tr>"
        for r, n, h in sweep.result)
    return ('<meta charset="utf-8">'
            '<style>body{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:24px}'
            'td,th{padding:6px 16px;border-bottom:1px solid #21262d}a{color:#58a6ff}</style>'
            '<h3>사거리 스윕 결과</h3><table>'
            '<tr><th>목표 사거리</th><th>사격</th><th>명중</th><th>명중률</th></tr>'
            + rows + '</table><br><a href="/">돌아가기</a>')


@app.route("/logs")
def list_logs():
    """저장된 로그 폴더 목록"""
    root = _log_root()
    if not os.path.isdir(root):
        return ('<meta charset="utf-8">아직 저장된 로그가 없다 '
                '<a href="/">돌아가기</a>')
    rows = []
    for d in sorted(os.listdir(root), reverse=True):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        files = sorted(os.listdir(p))
        size = sum(os.path.getsize(os.path.join(p, f)) for f in files) / 1024
        rows.append(f"<tr><td><b>{d}</b></td><td>{', '.join(files)}</td>"
                    f"<td>{size:.0f} KB</td></tr>")
    return ('<meta charset="utf-8">'
            '<style>body{background:#0d1117;color:#c9d1d9;font-family:monospace;'
            'padding:24px}td,th{padding:6px 16px;border-bottom:1px solid #21262d;'
            'font-size:13px}a{color:#58a6ff}</style>'
            '<h3>저장된 로그 (logs/)</h3><table>'
            '<tr><th>폴더</th><th>파일</th><th>크기</th></tr>'
            + "".join(rows) + '</table><br><a href="/">돌아가기</a>')


@app.route("/set_tag")
def set_tag():
    """저장 파일 이름에 붙을 태그를 바꾼다."""
    global CURRENT_TAG
    CURRENT_TAG = request.args.get("tag", "").strip() or "run1"
    return (f'<meta charset="utf-8">태그 = <b>{CURRENT_TAG}</b> '
            '<a href="/">돌아가기</a>')


@app.route("/clear_ticks")
def clear_ticks():
    """조건을 바꾸기 전에 틱 로그를 비운다. 사격 통계는 건드리지 않는다."""
    n = len(bot.ticks.rows)
    bot.ticks.clear()
    return ('<meta charset="utf-8">틱 로그 초기화 '
            f'({n}행 삭제) <a href="/">돌아가기</a>')


@app.route("/stats")
def stats():
    lg = bot.log
    br = lg.bias_report()
    return jsonify({
        "fired": lg.fired, "tank_hits": lg.tank_hits,
        "hit_rate": round(lg.hit_rate, 2),
        "terrain": lg.terrain_hits, "obstacle": lg.obstacle_hits,
        "unmatched": lg.unmatched, "incoming": lg.incoming,
        "mean_miss": round(lg.mean_miss, 3) if lg.mean_miss else None,
        "by_cond": {k: {"fired": v[0], "hits": v[1],
                        "rate": round(v[1] / v[0] * 100, 1) if v[0] else 0}
                    for k, v in sorted(lg.by_cond.items())},
        "residual": (None if not br else
                     {"n": br[0], "range_mean": round(br[1], 3),
                      "range_sd": round(br[2], 3),
                      "cross_mean": round(br[3], 3), "cross_sd": round(br[4], 3)}),
        "bias": {"range": round(bot.bias.range_bias, 3),
                 "bearing": round(bot.bias.bearing_bias, 4), "n": bot.bias.n},
        "cfg": CFG, "enemy_behavior": ENEMY_BEHAVIOR,
        "ctrl_dt": round(bot.ctrl_dt, 4),
        "pred_k_err": round(bot.trk.k_err, 4), "pred_n": bot.trk.n_scored,
    })


@app.route("/reset")
def reset_stats():
    bot.log.__init__()
    bot.bias.__init__()
    bot.ticks.clear()
    return '<meta charset="utf-8">통계·보정·틱로그 초기화 <a href="/">돌아가기</a>'


# ══════════════════════════════════════════════════════════
# 6. 대시보드
# ══════════════════════════════════════════════════════════
@app.route("/")
def index():
    tm, fc, lg, trk = bot.tm, bot.fc, bot.log, bot.trk
    tag = CURRENT_TAG
    today = time.strftime('%Y%m%d')
    sol = fc.last_solution
    rmin, rmax = bot.bal.min_range(), bot.bal.max_range()[0]
    near, far = bot.mv.band(bot.v_cross, bot.fc, bot.trk)

    def fmt(v, u="", n=1):
        return f"{v:.{n}f}{u}" if isinstance(v, (int, float)) else "-"

    rows = ""
    for r in lg.records[:60]:
        c = {"tank": "#3fb950", "obstacle": "#e3b341", "terrain": "#8b949e",
             "incoming": "#f85149"}.get(r["kind"], "#8b949e")
        zc = {"rear": "#f85149", "side": "#e3b341",
              "front": "#58a6ff"}.get(r.get("zone"), "#8b949e")
        zt = {"rear": "후면", "side": "측면", "front": "전면"}.get(r.get("zone"), "-")
        rows += (f"<tr><td>#{r['id']}</td><td>{r['time']}</td>"
                 f"<td>{r.get('cond','-')}</td><td>{r['dist']}</td>"
                 f"<td style='color:#e3b341'>{fmt(r.get('miss'),' m',2)}</td>"
                 f"<td style='color:#a5a5ff'>{fmt(r.get('range_err'),'',2)} / "
                 f"{fmt(r.get('cross_err'),'',2)}</td>"
                 f"<td>{fmt(r.get('p_hit'),'',2)}</td>"
                 f"<td>{fmt(r.get('own_speed'),'',1)}/{fmt(r.get('enemy_speed'),'',1)}</td>"
                 f"<td style='color:{zc}'>{zt}</td>"
                 f"<td>{fmt(r.get('damage'),'',1)}</td>"
                 f"<td style='color:{c};font-weight:600'>{r['result']}</td></tr>")
    if not rows:
        rows = "<tr><td colspan='11' style='text-align:center;color:#8b949e'>사격 기록 없음</td></tr>"

    crows = "".join(
        f"<tr><td>{k}</td><td>{v[0]}</td><td>{v[1]}</td>"
        f"<td style='color:{'#3fb950' if v[0] and v[1]/v[0]>=0.9 else '#e3b341'}'>"
        f"{(v[1]/v[0]*100 if v[0] else 0):.1f}%</td></tr>"
        for k, v in sorted(lg.by_cond.items())) or \
        "<tr><td colspan='4' style='color:#8b949e'>기록 없음</td></tr>"

    zrows = "".join(f"<tr><td>{ko}</td><td>{n}</td><td>{fmt(avg,'',2)}</td></tr>"
                    for ko, n, avg in lg.zone_summary())

    br = lg.bias_report()
    btxt = ("표본 없음" if not br else
            f"n={br[0]}  사거리잔차 {br[1]:+.2f} ± {br[2]:.2f} m  ·  "
            f"방위잔차 {br[3]:+.2f} ± {br[4]:.2f} m")

    d = tm.dist
    asp = tm.my_aspect
    asp_txt = "-" if asp is None else \
        f"{asp:+.0f}° ({'후방' if abs(asp)>120 else '측면' if abs(asp)>60 else '전방'})"
    sol_txt = "-"
    if sol and sol.valid:
        sol_txt = (f"방위 {sol.bearing:.2f} / 앙각 {sol.elevation:+.2f} / "
                   f"비행 {sol.flight:.2f}s / 리드 {sol.lead:.1f}m / "
                   f"P(hit) {sol.p_hit*100:.0f}% / "
                   f"σ 횡{sol.sig_lat:.2f} 종{sol.sig_lon:.2f} m / "
                   f"표적반치수 횡{sol.half_lat:.2f} 종{sol.half_lon:.2f} m")
    elif sol:
        sol_txt = sol.reason

    ecolor = "#3fb950" if bot.enemy.calls > 0 else "#f85149"
    ewarn = ("" if bot.enemy.calls > 0 else
             '<div class="card" style="border-color:#f85149"><div class="k" '
             'style="color:#f85149">Enemy EndPoint 미호출</div>'
             '<div style="font-size:13px;line-height:1.7">'
             f'포트 {ENEMY_PORT} 로 요청이 오지 않습니다. 확인 순서:<br>'
             f'1. <b>http://localhost:{ENEMY_PORT}/ping</b> 으로 서버 생존 확인<br>'
             '2. 시뮬레이터 Setting → <b>Use Enemy Server 체크</b>, '
             f'Enemy Request Port = <b>{ENEMY_PORT}</b><br>'
             '3. Save 후 Run 으로 에피소드 <b>재시작</b></div></div>')

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Tank FCS v6</title><meta http-equiv="refresh" content="1">
<style>
body{{background:#0d1117;color:#c9d1d9;font-family:"Segoe UI",monospace;margin:0;padding:22px}}
h1{{color:#58a6ff;font-size:21px;margin:0 0 4px}}
.sub{{color:#8b949e;font-size:12px;margin-bottom:16px;line-height:1.8}}
.row{{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 18px;flex:1;min-width:130px}}
.k{{font-size:11px;color:#8b949e;margin-bottom:3px}}
.v{{font-size:22px;font-weight:700;color:#f0f6fc}}
table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #21262d;font-size:12px}}
th{{background:#21262d;color:#8b949e}}
h2{{font-size:15px;color:#8b949e;margin:20px 0 8px}}
a{{color:#58a6ff}}
</style></head><body>
<h1>Tank FCS v6 — 정지 사격</h1>
<div class="sub">포락 {rmin:.0f}~{rmax:.0f} m · 교전밴드 <b style="color:#79c0ff">{near:.0f}~{far:.0f} m</b> ·
사격 임계 P(hit)≥{CFG['p_hit_min']:.2f} · 정지사격 {'OFF' if CFG['moving_fire'] else 'ON'} ·
자기보정 {'ON' if CFG['bias'] else 'OFF'} · 제어주기 실측 <b style="color:#79c0ff">{bot.ctrl_dt:.3f}s</b> ·
적 행동 <b style="color:#79c0ff">{ENEMY_BEHAVIOR}</b><br>
튜닝: <a href="/set?band_near=30&band_far=42">근접밴드 30~42</a> ·
<a href="/set?p_hit_min=0.75">P≥0.75</a> ·
<a href="/set?db_safety=0.35">데드밴드 엄격</a> ·
<a href="/set?behavior=static">적 정지</a> ·
<a href="/set?behavior=evade">적 회피</a> ·
<a href="/stats">JSON 통계</a><br>
사거리 스윕 (정지-정지 시험): <b style="color:#79c0ff">{sweep.progress()}</b> ·
<a href="/sweep?start=25&end=120&step=5&shots=3">25~120m / 5m / 3발</a> ·
<a href="/sweep?start=25&end=125&step=10&shots=5">25~125m / 10m / 5발</a> ·
<a href="/sweep?stop=1">중지</a> ·
<a href="/sweep_result">결과표</a></div>

<div class="row">
  <div class="card"><div class="k">사격</div><div class="v">{lg.fired}</div></div>
  <div class="card"><div class="k">명중</div><div class="v" style="color:#3fb950">{lg.tank_hits}</div></div>
  <div class="card"><div class="k">명중률</div><div class="v" style="color:{'#3fb950' if lg.hit_rate>=90 else '#e3b341'}">{lg.hit_rate:.1f}%</div></div>
  <div class="card"><div class="k">평균 착탄오차</div><div class="v">{fmt(lg.mean_miss,' m',2)}</div></div>
  <div class="card"><div class="k">지면/장애물</div><div class="v" style="font-size:17px">{lg.terrain_hits} / {lg.obstacle_hits}</div></div>
  <div class="card"><div class="k">피격</div><div class="v" style="color:#f85149">{lg.incoming}</div></div>
</div>

<div class="row">
  <div class="card"><div class="k">봇 페이즈</div><div class="v" style="font-size:17px;color:#79c0ff">{bot.phase}</div></div>
  <div class="card"><div class="k">사격 상태</div><div class="v" style="font-size:17px">{fc.state}</div></div>
  <div class="card"><div class="k">기동 상태</div><div class="v" style="font-size:17px">{bot.mv.state}</div></div>
  <div class="card"><div class="k">거리</div><div class="v" style="font-size:17px">{fmt(d,' m')}</div></div>
  <div class="card"><div class="k">적 기준 우리 위치</div><div class="v" style="font-size:17px">{asp_txt}</div></div>
  <div class="card"><div class="k">HP 아/적</div><div class="v" style="font-size:17px">{tm.my_hp} / {tm.enemy_hp}</div></div>
</div>

<div class="row">
  <div class="card"><div class="k">적 속도 / 선회</div><div class="v" style="font-size:17px">{trk.speed:.1f} m/s · {trk.turn_rate_dps:+.0f}°/s</div></div>
  <div class="card"><div class="k">횡단속도 → 목표거리</div><div class="v" style="font-size:17px">{bot.v_cross:.1f} m/s → {bot.mv.opt_range:.0f} m</div></div>
  <div class="card"><div class="k">포탑 추적 부하</div><div class="v" style="font-size:17px;color:{'#f85149' if fc.track_duty>CFG['track_duty_max'] else '#3fb950'}">{fc.track_duty*100:.0f}%</div></div>
  <div class="card"><div class="k">아군 속도</div><div class="v" style="font-size:17px">{tm.my_speed:.2f} m/s</div></div>
  <div class="card"><div class="k">예측오차 계수</div><div class="v" style="font-size:17px">{trk.k_err:.3f} <span style="font-size:11px;color:#8b949e">n={trk.n_scored}</span></div></div>
  <div class="card"><div class="k">조준오차 방위/앙각</div><div class="v" style="font-size:17px">{fc.yaw_err:+.2f}° / {fc.pitch_err:+.2f}°</div></div>
  <div class="card"><div class="k">데드밴드</div><div class="v" style="font-size:17px">{fc.yaw_db:.2f}° / {fc.pitch_db:.3f}°</div></div>
  <div class="card"><div class="k">적 봇 ({ENEMY_BEHAVIOR})</div>
      <div class="v" style="font-size:17px;color:{ecolor}">{bot.enemy.state}
      <span style="font-size:11px;color:#8b949e"> {bot.enemy.calls}회</span></div></div>
</div>
{ewarn}

<div class="card"><div class="k">사격 해</div>
  <div style="font-size:13px;color:#79c0ff">{sol_txt}</div>
  <div class="k" style="margin-top:8px">자기 보정 (C5)</div>
  <div style="font-size:13px;color:#a5d6ff">{bot.bias.summary()}</div>
  <div class="k" style="margin-top:8px">실사격 잔차 (누적)</div>
  <div style="font-size:13px;color:#a5d6ff">{btxt}</div>
  <div class="k" style="margin-top:10px">로그 저장 &rarr;
    <code style="color:#79c0ff">logs/{today}_{tag}/</code></div>
  <div style="font-size:12px;margin-top:4px">
    <a href="/save?tag={tag}"><b>세 개 한번에 저장</b></a> &nbsp;|&nbsp;
    <a href="/save_shots?tag={tag}">사격 로그</a> ·
    <a href="/save_speed?tag={tag}">속도 로그</a> ·
    <a href="/save_ticks?tag={tag}">틱 로그</a> &nbsp;|&nbsp;
    <a href="/logs">저장된 로그 보기</a> ·
    <a href="/export">내려받기</a> ·
    <a href="/clear_ticks">틱 비우기</a> ·
    <a href="/reset">통계 초기화</a>
  </div>
  <div style="font-size:11px;color:#8b949e;margin-top:6px">
    지금 태그: <b style="color:#79c0ff">{tag}</b> &nbsp;바꾸기:
    <a href="/set_tag?tag=sweep">sweep</a> ·
    <a href="/set_tag?tag=static">static</a> ·
    <a href="/set_tag?tag=evade">evade</a> ·
    <a href="/set_tag?tag=circle">circle</a> ·
    <a href="/set_tag?tag=move_static">move_static</a> ·
    <a href="/set_tag?tag=move_move">move_move</a>
    &nbsp;(직접: <code>/set_tag?tag=이름</code>)
  </div></div>

<h2>조건별 명중률</h2>
<table><thead><tr><th>조건</th><th>사격</th><th>명중</th><th>명중률</th></tr></thead>
<tbody>{crows}</tbody></table>

<h2>피탄 부위별 피해</h2>
<table><thead><tr><th>부위</th><th>명중 수</th><th>평균 피해</th></tr></thead>
<tbody>{zrows}</tbody></table>

<h2>사격 기록 (최근 60발)</h2>
<table><thead><tr>
<th>#</th><th>시각</th><th>조건</th><th>거리</th><th>착탄오차</th>
<th>종/횡 오차</th><th>P(hit)</th><th>속도 아/적</th><th>부위</th><th>피해</th><th>결과</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


@app.route("/<path:unknown>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def catch_all(unknown):
    return jsonify({})


# ══════════════════════════════════════════════════════════
# 7. 적 전차 서버 (포트 5100)
# ══════════════════════════════════════════════════════════
enemy_app = Flask("enemy")


@enemy_app.route("/get_action", methods=["POST"])
def enemy_action():
    return jsonify(bot.enemy.act(request.get_json(force=True, silent=True) or {}))


@enemy_app.route("/update_bullet", methods=["POST"])
def enemy_bullet():
    return jsonify({"status": "OK"})


@enemy_app.route("/ping")
def enemy_ping():
    return jsonify({"ok": True, "calls": bot.enemy.calls,
                    "behavior": bot.enemy.behavior, "state": bot.enemy.state})


@enemy_app.route("/<path:unknown>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def enemy_any(unknown):
    return jsonify({})


def run_enemy():
    enemy_app.run(host="0.0.0.0", port=ENEMY_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    b = Ballistics()
    print("=" * 60)
    print("  Tank FCS v6 (정지 사격)   http://localhost:5000")
    print(f"  교전 포락 {b.min_range():.1f} ~ {b.max_range()[0]:.1f} m")
    print(f"  교전 거리 탐색 {CFG['band_near']:.0f} ~ {CFG['band_far']:.0f} m"
          f"  (비행시간 {b.flight(b.solve_elevation(CFG['band_near'])):.2f}"
          f" ~ {b.flight(b.solve_elevation(CFG['band_far'])):.2f} s)")
    print("  * 실제 교전 거리는 표적 속도에 따라 매 틱 자동 선택됩니다")
    print(f"  사격 임계 P(hit) >= {CFG['p_hit_min']}")
    print(f"  적 전차 조종  포트 {ENEMY_PORT},  행동 '{ENEMY_BEHAVIOR}'")
    print("  시뮬레이터: Mode=Simulation, Request Port=5000,")
    print(f"             Enemy Request Port={ENEMY_PORT}, Use Enemy Server 체크")
    print("             Terrain = Simple Flat")
    print("=" * 60)
    threading.Thread(target=run_enemy, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
