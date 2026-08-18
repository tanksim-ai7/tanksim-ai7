# -*- coding: utf-8 -*-
# ── 버전 ────────────────────────────────────────────────
#   파일   enemy_bot.py
#   버전   v7   (2026-08-07)
#   역할   적 전차 조종 (포트 5100). 현재 사격하지 않는다.
#   변경 이력은 같은 폴더의  변경이력.md  를 볼 것.
#   ※ 파일명은 바꾸지 않는다 (import 가 이름으로 걸려 있다).
#      버전 구분은 이 배너 + 날짜 폴더(260806/260807/…) 로 한다.
# ────────────────────────────────────────────────────────
"""
enemy_bot.py (v6) - 적 전차(Red) 조종 서버

v5 대비 변경
  E1  static 판정 순서 수정
      v5 는 `if self.behavior == "patrol" or p is None:` 가 static 검사보다
      먼저 있어, 아군 위치가 아직 공유되지 않은 에피소드 초반에는
      behavior="static" 인데도 순찰 경로로 주행했다.
      -> '정지표적 명중률' 측정이 실제로는 이동표적 측정이 되고 있었다.
  E2  linear / strafe 행동 추가
      등속 직선 표적(예측이 쉬움) / 일정 속도 횡단 표적을 만들어
      난이도 단계별로 명중률을 나눠 측정할 수 있게 했다.
  E3  serpentine 주기/진폭을 상수로 분리
      v5 는 호출당 위상 +0.10 rad 고정이라 제어주기가 바뀌면 주기도 바뀌었다.
      시뮬 시각 기반으로 바꿔 PC 성능과 무관하게 재현된다.
  E4  차체 각속도 명령 계산 시 dt 를 실측값으로 사용
  E5  ** 조향 제어기 발진 수정 (가장 중요) **
      v5 는 weight = |오차| / (40 x dt) 로, '한 틱에 오차를 전부 없애는'
      deadbeat P 제어였다. 그런데 차체각 피드백(set_body)은 Player 서버의
      /info 를 거쳐 오므로 최소 한 틱 늦다.
      지연 1틱 + 게인 1.0 = 지속 발진.  실제로 차체각이
      208 -> 224 -> 240 -> 224 -> 208 처럼 매 틱 +-16.4 deg 지그재그했고,
      |오차|>50 deg 구간에서 STOP 이 걸려 속도까지 20 -> 2 m/s 로 급락했다.
      -> 표적 궤적이 물리적으로 불가능할 만큼 튀어, 어떤 예측기로도
         0.8 s 앞을 5 m 이내로 못 맞히는 상태였다 (등속 직선 순찰인데도!).
      수정: 게인 0.55 (임계감쇠에 가깝게) + 각속도 변화율 제한.
            KP=1.0 으로 되돌리면 v5 거동을 재현할 수 있다.

행동 정책
    static    정지 (기준 조건 - 정지표적 명중률 측정용)
    linear    한 방향으로 등속 직선 (예측 용이)
    strafe    아군 시선에 수직으로 왕복 횡단
    circle    아군을 중심으로 일정 반경 선회
    evade     거리를 유지하며 사행(serpentine) - 가장 어려운 표적
    charge    돌진
    patrol    아군 무시, 정해진 경로 순찰
"""
import math
import random
from typing import Optional, Tuple

Vec3 = Tuple[float, float, float]

MAP_MIN, MAP_MAX = 0.0, 300.0
MARGIN = 40.0
YAW_RATE = 40.0
CTRL_DT = 0.41
BODY_DEADBAND = 4.0

# E5: 조향 P 게인. 1.0 = v5 (deadbeat, 피드백 1틱 지연과 만나 발진)
#     0.55 이하가 안정. 0.5~0.6 권장.
STEER_KP = 0.55
STEER_SLEW = 0.35      # weight 변화율 제한 (틱당). 급격한 조향 반전 억제.
DRIVE_STOP_ERR = 75.0  # 이 각도보다 크게 틀어졌을 때만 정지 (v5 는 50 deg)

SERP_PERIOD = 11.0     # 사행 주기 [s]  (E3: 시간 기반)
SERP_AMPL = 45.0       # 사행 진폭 [deg]
STRAFE_PERIOD = 14.0   # 횡단 왕복 주기 [s]

PATROL_PTS = [(70.0, 70.0), (230.0, 70.0), (230.0, 230.0), (70.0, 230.0)]


def _ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _bearing(frm, to):
    return math.degrees(math.atan2(to[0] - frm[0], to[1] - frm[1])) % 360.0


class EnemyBot:
    """
    behavior : static / linear / strafe / circle / evade / charge / patrol
    keep_r   : 유지하려는 거리 [m]
    mask     : True 면 아군 위치를 sense_r 밖에서는 모르는 것으로 처리
    """

    def __init__(self, behavior="evade", keep_r=90.0, mask=False,
                 sense_r=140.0, seed=0, speed_cap=1.0):
        self.behavior = behavior
        self.keep_r = keep_r
        self.mask = mask
        self.sense_r = sense_r
        self.speed_cap = speed_cap        # 주행 weight 상한 (표적 속도 조절)
        self.rng = random.Random(seed)

        self.pos: Optional[Tuple[float, float]] = None
        self.body_x = 0.0
        self.turret_x = 0.0
        self.player: Optional[Tuple[float, float]] = None
        self.state = "IDLE"
        self.patrol_i = 0
        self.orbit_dir = 1
        self.calls = 0
        self.t: Optional[float] = None
        self._t0: Optional[float] = None
        self.dt = CTRL_DT
        self._last_t: Optional[float] = None
        self._linear_dir: Optional[float] = None
        self._steer_w = 0.0        # E5: 직전 조향 weight (변화율 제한용)

    # ── Player 서버가 /info 를 받을 때 호출해 준다 ──────────
    def share_world(self, player_pos: Vec3):
        self.player = (player_pos[0], player_pos[2])

    def set_body(self, deg: float):
        self.body_x = deg

    def set_time(self, t: Optional[float]):
        if t is None:
            return
        if self._t0 is None:
            self._t0 = t
        if self._last_t is not None:
            d = t - self._last_t
            if 0.05 < d < 3.0:
                self.dt = 0.2 * d + 0.8 * self.dt      # E4
        self._last_t = t
        self.t = t

    @property
    def elapsed(self) -> float:
        if self.t is None or self._t0 is None:
            return 0.0
        return self.t - self._t0

    def _known_player(self):
        if self.player is None or self.pos is None:
            return None
        if self.mask and math.dist(self.pos, self.player) > self.sense_r:
            return None
        return self.player

    # ── 목표 지점 결정 ────────────────────────────────────
    def _goal(self):
        x, z = self.pos

        # E1: static 을 가장 먼저 판정한다 (아군 위치와 무관하게 항상 정지)
        if self.behavior == "static":
            self.state = "STATIC"
            return None

        if self.behavior == "patrol":
            self.state = "PATROL"
            gx, gz = PATROL_PTS[self.patrol_i]
            if math.dist((x, z), (gx, gz)) < 18.0:
                self.patrol_i = (self.patrol_i + 1) % len(PATROL_PTS)
            return gx, gz

        # E2: 아군 위치를 몰라도 되는 등속 직선 표적
        if self.behavior == "linear":
            self.state = "LINEAR"
            if self._linear_dir is None:
                self._linear_dir = 90.0            # +X 방향
            a = math.radians(self._linear_dir)
            gx, gz = x + math.sin(a) * 60.0, z + math.cos(a) * 60.0
            if not (MARGIN + 10 < gx < MAP_MAX - MARGIN - 10):
                self._linear_dir = (self._linear_dir + 180.0) % 360.0
            return gx, gz

        p = self._known_player()
        if p is None:
            self.state = "PATROL"
            gx, gz = PATROL_PTS[self.patrol_i]
            if math.dist((x, z), (gx, gz)) < 18.0:
                self.patrol_i = (self.patrol_i + 1) % len(PATROL_PTS)
            return gx, gz

        d = math.dist((x, z), p)
        away = _bearing(p, (x, z))          # 아군 -> 나 방향

        if self.behavior == "charge":
            self.state = "CHARGE"
            return p

        if self.behavior == "strafe":
            # 아군 시선에 수직으로 왕복. 반경은 유지.
            self.state = "STRAFE"
            ph = 2 * math.pi * self.elapsed / STRAFE_PERIOD
            lat = math.sin(ph) * 70.0
            ang = math.radians(away + lat)
            return (p[0] + math.sin(ang) * self.keep_r,
                    p[1] + math.cos(ang) * self.keep_r)

        if self.behavior == "circle":
            self.state = "CIRCLE"
            ang = math.radians(away + 55.0 * self.orbit_dir)
            return (p[0] + math.sin(ang) * self.keep_r,
                    p[1] + math.cos(ang) * self.keep_r)

        # evade - 거리를 유지하며 좌우로 사행 (E3: 시간 기반 위상)
        self.state = "EVADE"
        ph = 2 * math.pi * self.elapsed / SERP_PERIOD
        lat = math.sin(ph) * SERP_AMPL
        ang = math.radians(away + lat)
        r = self.keep_r if d < self.keep_r * 1.3 else self.keep_r * 0.8
        return (p[0] + math.sin(ang) * r, p[1] + math.cos(ang) * r)

    # ── 매 /get_action ────────────────────────────────────
    def act(self, payload):
        self.calls += 1
        pos = payload.get("position") or {}
        tur = payload.get("turret") or {}
        if "x" in pos:
            self.pos = (float(pos["x"]), float(pos["z"]))
        self.turret_x = float(tur.get("x", self.turret_x) or 0.0)
        if payload.get("time") is not None:
            self.set_time(float(payload["time"]))

        idle = {"moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "", "weight": 0.0},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False}
        if self.pos is None:
            return idle

        x, z = self.pos
        goal = self._goal()

        # 맵 경계를 벗어나면 중앙으로 (가장자리 정체 방지)
        # static 은 예외 - 정지표적은 어떤 경우에도 움직이지 않는다
        if self.behavior != "static" and not (
                MARGIN < x < MAP_MAX - MARGIN and MARGIN < z < MAP_MAX - MARGIN):
            self.state = "RECENTER"
            self.orbit_dir *= -1
            goal = (150.0, 150.0)

        if goal is None:
            return idle
        gx, gz = (max(MARGIN, min(MAP_MAX - MARGIN, goal[0])),
                  max(MARGIN, min(MAP_MAX - MARGIN, goal[1])))

        if math.dist((x, z), (gx, gz)) < 6.0:
            return idle

        tgt_yaw = _bearing((x, z), (gx, gz))
        err = _ang_diff(tgt_yaw, self.body_x)

        # E5: 게인을 낮추고(0.55) 변화율을 제한해 발진을 막는다
        ad = {"command": "", "weight": 0.0}
        signed = 0.0
        if abs(err) > BODY_DEADBAND:
            w = min(1.0, max(0.10, STEER_KP * abs(err) / (YAW_RATE * self.dt)))
            signed = w if err > 0 else -w
        prev = self._steer_w
        signed = max(prev - STEER_SLEW, min(prev + STEER_SLEW, signed))
        self._steer_w = signed
        if abs(signed) >= 0.10:
            ad = {"command": "D" if signed > 0 else "A",
                  "weight": round(abs(signed), 2)}

        ws = ({"command": "W", "weight": round(self.speed_cap, 2)}
              if abs(err) < DRIVE_STOP_ERR else {"command": "STOP", "weight": 1.0})

        # 포탑은 아군 쪽을 향하게 (사격은 하지 않음 - 실험 통제)
        qe = {"command": "", "weight": 0.0}
        p = self._known_player()
        if p:
            terr = _ang_diff(_bearing((x, z), p), self.turret_x)
            if abs(terr) > 1.5:
                tw = min(1.0, max(0.1, abs(terr) / (YAW_RATE * self.dt)))
                qe = {"command": "E" if terr > 0 else "Q", "weight": round(tw, 2)}

        return {"moveWS": ws, "moveAD": ad, "turretQE": qe,
                "turretRF": {"command": "", "weight": 0.0}, "fire": False}
