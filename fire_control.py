# -*- coding: utf-8 -*-
# ── 버전 ────────────────────────────────────────────────
#   파일   fire_control.py
#   버전   v10   (2026-08-07)
#   역할   사격 통제 핵심. C1~C14.
#   변경 이력은 같은 폴더의  변경이력.md  를 볼 것.
#   ※ 파일명은 바꾸지 않는다 (import 가 이름으로 걸려 있다).
#      버전 구분은 이 배너 + 날짜 폴더(260806/260807/…) 로 한다.
# ────────────────────────────────────────────────────────
"""
fire_control.py  (v6) - Tank Challenge 사격 통제

v5 대비 변경 요약  (자세한 내용은 CHANGES.xlsx 참조)
  C1  표적 추정기 교체     지수이동평균(EMA) -> 이동창 최소제곱(LSQ)
                           EMA 는 dt 0.41 s 에서 위상 지연이 커 리드가 항상 뒤처졌다.
  C2  예측 모델 교체       등가속(CA) -> 등속선회(CTRV, 수치적분)
                           적은 직선 가속이 아니라 '선회'로 회피한다.
  C3  표적 자세 반영       표적 사각형을 시선 방향으로 투영해 종/횡 반치수를 구한다.
                           v5 는 항상 폭=3.6, 길이=7.5 로 고정해 측면 표적에서
                           종방향 허용오차를 실제보다 2배 크게 잡고 있었다.
  C4  경험적 불확실성      예측을 기록해뒀다가 실제 위치와 대조해 sigma 를 실측한다.
                           v5 는 '적이 최대 성능으로 기동한다'는 최악 가정이라
                           실제보다 보수적이거나(안 쏨) 낙관적이었다.
  C5  자기 보정            착탄점 - 조준점 잔차를 적분해 사거리/방위 바이어스를 학습.
                           탄속/중력/발사고/지연의 잔여 모형오차를 실사격으로 제거한다.
  C6  통합 사격 판정       데드밴드 통과 여부가 아니라 P(hit) 로 판정한다.
                           조준오차 + 예측오차 + 표적 투영치수를 한 식에 넣는다.
  C7  발사 순간 정합       사격 판정에 쓰는 자세는 '이번 틱에 명령을 넣지 않았을 때의
                           포탑 각도'다. 발사 틱에는 포탑 명령을 0 으로 낸다.
                           (v5 는 회전 명령과 fire 를 동시에 내보내 조준이 틀어졌다)

  C8  포구 오프셋 도입 (v6.1, 2026-08-04 실측 반영)
                           탄이 playerPos 가 아니라 그보다 4.74 m 앞의 포구에서
                           나간다는 항을 넣었다. 앙각별 계통 잔차가 사라졌다.
                           사거리 RMS 0.529 m -> 0.093 m

  C9  종방향 비대칭 허용오차 (v6.2, 실사격 반영)
                           탄이 완만히 하강하므로 '길게' 조준하면 차체 윗부분을
                           때린다(허용 약 22 m). '짧게'면 앞 지면에 박힌다(3.75 m).
                           의도적으로 창 중심까지 밀어 조준한다 (lon_gain).
                           (weight 하한 w_min 0.1 -> 0.02 는 알고리즘 변경이
                            아니라 P8 실측 결과다. 아래 실측 상수 블록 참조.
                            v6.3 에서 'C10' 으로 적었으나 아래 C10 과 번호가
                            겹쳐 표기를 거뒀다. 2026-08-06)

── 2026-08-06 실사격 161발로 추가한 것 (v8~v10) ────────────────
  C10 발사 시점 정합       발사 판정을 '탄이 실제로 떠나는 순간'의 포탑 각도로
                           한다. 비안정화 포탑이라 발사 지연 0.125 s 동안
                           차체와 함께 더 돈다.
  C11 비행시간 상한        이동표적에 한해 비행시간 0.50 s 를 넘으면 쏘지 않고
                           접근한다. 0.5 s 를 넘으면 명중률이 87 -> 61 -> 25%
                           로 무너진다 (실측 73발).
  C12 종방향 창 상한       (기각) 긴 쪽 허용을 4 m 로 제한. 이동-이동이
                           87.5 -> 50.0% 로 떨어져 되돌렸다. lon_long_max=None.
  C13 선회 중 사격 금지    발사 순간 |차체 회전율| 15 deg/s 초과면 쏘지 않는다.
                           개입 전 116발에서 회전 <=15 은 42/42(100%),
                           >15 은 58/74(78.4%) 였다.
  C14 조준점 중심 이동     lon_gain 0.25 -> 0.15. 종방향 창의 실제 경계가
                           +5 m 부근이라(모델은 21~25 m 로 믿었다) 매 발
                           +2.7 m 길게 조준하던 것이 꼬리를 만들었다.

실측 상수 (2026-08-04 measure_harness P0~P8, 총 90발 + 텔레메트리 19,637행)
    포구 속도  v = 56.356 m/s     * v 와 g 는 v^2/g = 324.3 으로만 결정됨
    포구 높이  h =  2.229 m       (앙각 0, playerPos.y 기준)
    포구 오프셋 L = 4.741 m       (playerPos 보다 앞선 거리)  <- v6.1 신규
    앙각 바이어스 -0.143 deg
    중력       g =  9.81 m/s^2    (표준중력으로 고정)
    앙각 범위  -5.00 ~ +10.00 deg  -> 교전 포락 21.0 ~ 129.6 m
    포탑 방위  40.00 deg/s x weight   (w 0.01~1.0 에서 완전 선형)
    포탑 앙각   5.00 deg/s x weight
    weight 하한 0.01 까지 선형 (P8 실측). 포탑각 해상도 0.00001 deg, 양자화 없음.
               -> w_min 0.02 채택. 데드밴드 하한 0.164 deg = 38 m 에서 0.11 m
    재장전     6.39 ~ 6.54 s  (중앙값 6.41)
    발사 지연  0.125 s  (마지막 /info 스냅샷 기준, P5 실측)
    /info 주기 0.130 s
    탄착 산포  0.06 m (사실상 0 - 완전 결정론적)
    차량       v_max 19.68 m/s,  a 2.32 m/s^2,  제동 -32 m/s^2
               선회 40.0 deg/s (제자리·주행 중 동일)

    적합 품질: 사거리 RMS 0.093 m / 최대 잔차 0.208 m (30개 앙각점)
               비행시간 RMS 0.031 s (관측 양자화 0.130 s 이하 = 노이즈 한계)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Callable, List

Vec3 = Tuple[float, float, float]


# ══════════════════════════════════════════════════════════
# 0. 각도 / 거리 유틸  (Unity: yaw 0 = +Z, 시계방향 증가)
# ══════════════════════════════════════════════════════════
def bearing(frm: Vec3, to: Vec3) -> float:
    return math.degrees(math.atan2(to[0] - frm[0], to[2] - frm[2])) % 360.0


def ang_diff(target: float, current: float) -> float:
    """-180 ~ +180. 양수면 시계방향(E)으로 돌려야 함"""
    return (target - current + 180.0) % 360.0 - 180.0


def dist2d(a: Vec3, b: Vec3) -> float:
    return math.hypot(b[0] - a[0], b[2] - a[2])


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ══════════════════════════════════════════════════════════
# 1. 실측 상수
# ══════════════════════════════════════════════════════════
@dataclass
class BallisticParams:
    """
    2026-08-04 실측 재적합 (P2 30점 + P6 8점, 전부 결정론적).

    ** v6.1 에서 바뀐 것: 포구 오프셋 muzzle_len 도입 **
        v5/v6 모델은 탄이 playerPos 에서 나간다고 가정했다.
        그 결과 앙각별 잔차에 뚜렷한 S자 계통 패턴이 남았다
        (-5° 에서 +0.86 m, +4° 에서 -1.15 m, +10° 에서 +0.81 m).
        포신 길이만큼 앞에서 나간다는 항을 넣자 패턴이 완전히 사라졌다.
            사거리 RMS  0.529 m -> 0.093 m   (5.7 배 개선)
            최대 잔차   0.978 m -> 0.208 m

    v 와 g 는 개별 분리가 불가능하다 (v^2/g = 324.3 만 결정됨).
    비행시간 관측이 /info 주기 0.130 s 로 양자화되어 판별력이 없다.
    -> 표준중력 9.81 로 고정하고 나머지를 적합했다.
    """
    v: float = 56.3557        # 포구 속도 [m/s]
    h: float = 2.2292         # 포구 높이 (앙각 0, playerPos.y 기준) [m]
    g: float = 9.81           # 중력가속도 [m/s^2]  (고정)
    muzzle_len: float = 4.7414  # 포구가 playerPos 보다 앞선 거리 [m]
    pitch_bias: float = -0.1425  # 보고 앙각과 실제 발사각의 차 [deg]
    theta_min: float = -5.0   # 앙각 하한 [deg]  (P4 실측, 정확히 -5.00)
    theta_max: float = 10.0   # 앙각 상한 [deg]  (P4 실측, 정확히 +10.00)


@dataclass
class TargetSize:
    """표적 치수. v6 는 시선 방향으로 투영해서 쓴다 (C3)"""
    width: float = 3.6        # K2 전차 폭 [m]  (진행방향 수직)
    length: float = 7.5       # 차체 길이 [m]  (진행방향)
    height: float = 2.2       # 차체 높이 [m]  <- C9 신규. 보수적으로 잡음
                              # (실측 K2 약 2.4 m. 낮게 잡을수록 안전)


@dataclass
class TurretParams:
    """
    ** w_min: 0.1 -> 0.02  (P8 실측, 2026-08-04) **
        v5/v6 는 weight 하한이 0.1 이라고 '가정' 했을 뿐 측정한 적이 없었다.
        이 가정이 조준 정밀도의 하한을 정한다:
            방위 최소 이동 = 40 x w_min x dt,  데드밴드 하한 = 그 절반
        2차 실사격에서 방위 오차 sigma 가 0.546 deg 로 나왔는데,
        이는 w_min=0.1 일 때의 데드밴드 하한 0.82 deg 의 정확히 절반이다.
        즉 정밀도가 물리 한계가 아니라 이 가정에 막혀 있었다.

        P8 실측: weight 0.01 에서 각속도 0.400 deg/s (이론값 0.4 와 정확히 일치).
                 포탑각이 0.00001 deg 해상도로 매끄럽게 변한다. 양자화 바닥 없음.
        -> 0.02 를 채택 (0.01 도 동작하지만 여유를 둔다).

        방위 최소 이동  1.640 -> 0.328 deg
        데드밴드 하한   0.820 -> 0.164 deg
        38 m 에서 조준 오차  0.54 -> 0.11 m
    """
    yaw_rate: float = 40.0    # deg/s @ weight 1.0  (P1 실측, 완전 선형)
    pitch_rate: float = 5.0   # deg/s @ weight 1.0  (P1 실측)
    dt: float = 0.41          # 제어 주기 [s]
    w_min: float = 0.02       # P8 실측 (0.01 까지 선형 확인)
    w_max: float = 1.0
    w_round: int = 3          # weight 소수점 자리 (w_min 을 낮추면 함께 늘려야 함)

    @property
    def yaw_min_step(self) -> float:
        return self.yaw_rate * self.w_min * self.dt      # 0.328 deg

    @property
    def pitch_min_step(self) -> float:
        return self.pitch_rate * self.w_min * self.dt    # 0.041 deg


@dataclass
class MotionLimits:
    """
    실측 차량 성능 (2026-08-04 P3/P7) - 표적 예측 폭주 방지용 클램프.

    P7 실측: 주행 중 선회도 제자리 선회와 같은 40.0 deg/s (weight 비례).
             선회 weight 1.0 에서 속도가 17.5 -> 15.6 m/s 로 약간 떨어질 뿐이다.
    P3 실측: 최고 속도는 weight 에 정확히 비례 (0.3/0.6/1.0 -> 5.88/11.77/19.68).
             제동은 -32 m/s^2 로 사실상 즉각 (19.4 m/s 에서 정지까지 0.48 s).
             -> 정지 사격의 시간 비용이 거의 없다.
    """
    a_max: float = 2.40        # 가속도 [m/s^2]  (P3 중앙값 2.32, 상위10% 3.99)
    turn_max: float = 40.0     # 선회 각속도 [deg/s] (P3 제자리·P7 주행 중 모두 40.00)
    v_max: float = 19.7        # 최고 속도 [m/s]  (P3 실측 19.68)
    decel: float = 32.0        # 제동 [m/s^2]  (P3 실측)


# ══════════════════════════════════════════════════════════
# 2. 탄도 정해 / 역해
# ══════════════════════════════════════════════════════════
@dataclass
class Ballistics:
    p: BallisticParams = None
    near_tol: float = 3.0     # 최소 사거리 미달 허용치 [m]

    def __post_init__(self):
        if self.p is None:
            self.p = BallisticParams()
        self._rmax_cache: Dict[int, Tuple[float, float]] = {}
        self._el_cache: Dict[Tuple[int, int], Optional[float]] = {}

    def _geom(self, theta_deg: float, dy: float):
        """(sin, cos, 포구 높이) - 앙각 바이어스와 포구 오프셋을 반영"""
        a = math.radians(theta_deg + self.p.pitch_bias)
        s, c = math.sin(a), math.cos(a)
        return s, c, self.p.h + dy + self.p.muzzle_len * s

    def flight(self, theta_deg: float, dy: float = 0.0) -> float:
        """포구를 떠나 지면에 닿기까지의 비행 시간 [s]"""
        v, g = self.p.v, self.p.g
        s, _, H = self._geom(theta_deg, dy)
        disc = (v * s) ** 2 + 2 * g * H
        if disc < 0:
            return float("nan")
        return (v * s + math.sqrt(disc)) / g

    def range_at(self, theta_deg: float, dy: float = 0.0) -> float:
        """playerPos 기준 수평 사거리 [m].  = 포구 전방 오프셋 + 탄도 사거리"""
        v = self.p.v
        s, c, _ = self._geom(theta_deg, dy)
        t = self.flight(theta_deg, dy)
        if math.isnan(t):
            return float("nan")
        return self.p.muzzle_len * c + v * c * t

    def max_range(self, dy: float = 0.0) -> Tuple[float, float]:
        """(최대사거리, 그때의 앙각). 앙각 상한이 지배한다."""
        key = int(round(dy * 20))
        if key in self._rmax_cache:
            return self._rmax_cache[key]
        best = (-1.0, self.p.theta_min)
        t = self.p.theta_min
        while t <= self.p.theta_max + 1e-9:
            r = self.range_at(t, dy)
            if not math.isnan(r) and r > best[0]:
                best = (r, t)
            t += 0.05
        self._rmax_cache[key] = best
        return best

    def min_range(self, dy: float = 0.0) -> float:
        return self.range_at(self.p.theta_min, dy)

    def dr_dtheta(self, theta_deg: float, dy: float = 0.0) -> float:
        """앙각 1도당 사거리 변화 [m/deg] - 앙각 오차를 거리 오차로 환산"""
        r1 = self.range_at(theta_deg - 0.25, dy)
        r2 = self.range_at(theta_deg + 0.25, dy)
        if math.isnan(r1) or math.isnan(r2):
            return 8.0
        return max(1.0, abs(r2 - r1) / 0.5)

    def solve_elevation_cached(self, dist_m: float, dy: float = 0.0) -> Optional[float]:
        """0.01 m 격자로 반올림해 캐시. 반복 탐색 비용을 없앤다.
        (0.01 m 양자화는 표적 폭 3.6 m 대비 무시할 수준)"""
        key = (int(dist_m * 100), int(dy * 100))
        if key in self._el_cache:
            return self._el_cache[key]
        v = self.solve_elevation(key[0] / 100.0, key[1] / 100.0)
        if len(self._el_cache) > 20000:
            self._el_cache.clear()
        self._el_cache[key] = v
        return v

    def solve_elevation(self, dist_m: float, dy: float = 0.0) -> Optional[float]:
        rmax, th_at_max = self.max_range(dy)
        if dist_m > rmax:
            return None
        lo, hi = self.p.theta_min, th_at_max
        rmin = self.range_at(lo, dy)
        if dist_m < rmin:
            return lo if (rmin - dist_m) <= self.near_tol else None
        for _ in range(44):
            mid = (lo + hi) / 2
            if self.range_at(mid, dy) < dist_m:
                lo = mid
            else:
                hi = mid
        th = (lo + hi) / 2
        return max(self.p.theta_min, min(self.p.theta_max, th))


# ══════════════════════════════════════════════════════════
# 3. 표적 상태 추정기  (C1, C2, C4)
# ══════════════════════════════════════════════════════════
class TargetTracker:
    """
    이동창 최소제곱으로 위치/속도/가속도를 추정하고,
    등속선회(CTRV) 모형으로 비행시간 뒤 위치를 예측한다.

    왜 EMA 가 아닌 LSQ 인가
        EMA(alpha=0.45, dt=0.41s) 의 유효 시상수는 약 0.5 s 다.
        적이 10 m/s 로 달리면 속도 추정이 항상 한 박자 늦어
        리드 방향으로 3~5 m 씩 밀린다. 표적 반폭이 1.8 m 이므로 치명적이다.
        LSQ 는 창 안의 모든 점을 동시에 쓰므로 위상 지연이 없다.

    왜 CA 가 아닌 CTRV 인가
        회피 기동은 '직선 가속'이 아니라 '선회'다. 등가속 모형으로
        선회를 외삽하면 곡선의 접선 방향으로 튀어나가 바깥쪽을 겨눈다.
        CTRV 는 속도 벡터를 회전시키며 적분하므로 곡선을 따라간다.
    """

    def __init__(self, window: float = 1.35, limits: Optional[MotionLimits] = None,
                 max_pts: int = 8, window_v: float = 0.95):
        self.window = window        # 곡률(선회율)용 긴 창
        self.window_v = window_v    # 속도용 짧은 창 (C1a)
        self.lim = limits or MotionLimits()
        self.hist: deque = deque(maxlen=max_pts)
        self.pos: Optional[Vec3] = None
        self.t: Optional[float] = None
        self.vel: Vec3 = (0.0, 0.0, 0.0)
        self.acc: Vec3 = (0.0, 0.0, 0.0)
        self.omega: float = 0.0        # 선회 각속도 [rad/s], + = 시계방향
        self.a_long: float = 0.0       # 접선 가속 [m/s^2]
        # 경험적 예측오차 (C4)
        self._pred_q: List[Tuple[float, float, float, float]] = []  # (t_target, px, pz, horizon)
        self.k_err: float = 0.0        # 오차 ≈ k_err * horizon^2  [m/s^2] (상위 분위수)
        self.k_mean: float = 0.0
        self.k_quant: float = 0.0
        self.k_recent: float = 0.0
        self.k_quantile: float = 0.80  # 장기 분위수
        self.k_recent_n: int = 3       # 최근 몇 개를 즉시 지표로 쓸지
        self.k_recent_w: float = 1.0   # 최근 지표 가중 (0 이면 끄기)
        self._ks: deque = deque(maxlen=60)
        self.n_scored: int = 0

    # ── 관측 ─────────────────────────────────────────────
    def update(self, t: float, pos: Vec3):
        if t is None or pos is None:
            return
        if self.hist and t <= self.hist[-1][0]:
            return                                    # 시간 역행/중복 무시
        self._score(t, pos)                           # 예약된 예측 채점 (C4)
        self.hist.append((t, pos[0], pos[1], pos[2]))
        while len(self.hist) > 2 and (t - self.hist[0][0]) > self.window:
            self.hist.popleft()
        self.t, self.pos = t, pos
        self._fit()

    def _fit(self):
        """
        C1a  속도는 짧은 창, 곡률은 긴 창으로 분리해서 추정한다.

        왜 분리하는가 (실측)
            창을 짧게(0.75 s = 2~3 점) 하면 속도 추정의 지연이 줄어
            회피(사행) 표적 명중률이 92.5% -> 96.9% 로 올라간다.
            그런데 선회율 omega 는 2차 항이 필요해 최소 4 점이 있어야 하고,
            짧은 창에서는 아예 추정이 안 된다. 그 결과 20 m/s 로 궤도를 도는
            표적에서는 예측이 무너져 한 발도 못 쏘게 된다(sigma 급증).
            -> 위치·속도는 최근 2~3 점(저지연), 선회율·접선가속은 전체 창(안정)
               에서 따로 뽑아 합친다. 두 조건 모두 만족한다.
        """
        n = len(self.hist)
        if n < 2:
            return
        t0 = self.hist[-1][0]

        # (1) 곡률용 - 전체 창에 2차 적합
        acc = (0.0, 0.0, 0.0)
        vel_long = None
        if n >= 4:
            ts = [r[0] - t0 for r in self.hist]
            cx = _polyfit(ts, [r[1] for r in self.hist], 2)
            cz = _polyfit(ts, [r[3] for r in self.hist], 2)
            if cx is not None and cz is not None:
                acc = (2 * cx[2], 0.0, 2 * cz[2])
                vel_long = (cx[1], 0.0, cz[1])

        # (2) 속도용 - 최근 window_v 안의 점만 (저지연)
        sub = [r for r in self.hist if (t0 - r[0]) <= self.window_v + 1e-9]
        if len(sub) < 2:
            sub = list(self.hist)[-2:]
        if len(sub) == 2:
            dt = sub[1][0] - sub[0][0]
            if abs(dt) < 1e-6:
                return
            vel = ((sub[1][1] - sub[0][1]) / dt, 0.0, (sub[1][3] - sub[0][3]) / dt)
        else:
            ts = [r[0] - t0 for r in sub]
            deg = 2 if len(sub) >= 4 else 1
            cx = _polyfit(ts, [r[1] for r in sub], deg)
            cz = _polyfit(ts, [r[3] for r in sub], deg)
            if cx is None or cz is None:
                vel = vel_long or self.vel
            else:
                vel = (cx[1], 0.0, cz[1])

        self.vel = vel
        self.acc = acc

        # 물리 한계로 클램프
        sp = math.hypot(self.vel[0], self.vel[2])
        if sp > self.lim.v_max:
            k = self.lim.v_max / sp
            self.vel = (self.vel[0] * k, 0.0, self.vel[2] * k)
            sp = self.lim.v_max
        am = math.hypot(self.acc[0], self.acc[2])
        if am > self.lim.a_max * 1.5:
            k = self.lim.a_max * 1.5 / am
            self.acc = (self.acc[0] * k, 0.0, self.acc[2] * k)

        # 접선/법선 분해 -> a_long, omega
        if sp > 0.4:
            ux, uz = self.vel[0] / sp, self.vel[2] / sp
            self.a_long = ux * self.acc[0] + uz * self.acc[2]
            a_lat = ux * self.acc[2] - uz * self.acc[0]     # 오른쪽(+X 쪽) 성분
            self.omega = -a_lat / sp
            wmax = math.radians(self.lim.turn_max)
            self.omega = max(-wmax, min(wmax, self.omega))
            self.a_long = max(-self.lim.a_max, min(self.lim.a_max, self.a_long))
        else:
            self.a_long = 0.0
            self.omega = 0.0

    # ── 예측 ─────────────────────────────────────────────
    def predict(self, dt: float, steps: int = 12) -> Vec3:
        """dt 뒤 위치. 등속선회 + 접선가속을 수치적분."""
        if self.pos is None:
            return (0.0, 0.0, 0.0)
        if dt <= 0:
            return self.pos
        x, z = self.pos[0], self.pos[2]
        vx, vz = self.vel[0], self.vel[2]
        sp = math.hypot(vx, vz)
        if sp < 1e-6:
            return self.pos
        hd = math.atan2(vx, vz)
        h = dt / steps
        for _ in range(steps):
            sp = max(0.0, min(self.lim.v_max, sp + self.a_long * h))
            hd += self.omega * h
            x += math.sin(hd) * sp * h
            z += math.cos(hd) * sp * h
        return (x, self.pos[1], z)

    # ── 경험적 예측오차 (C4) ─────────────────────────────
    def enqueue_prediction(self, horizon: float):
        """지금 시점에서 horizon 뒤를 예측해 두고, 그때 실제와 대조한다."""
        if self.t is None or horizon <= 0.05:
            return
        p = self.predict(horizon)
        self._pred_q.append((self.t + horizon, p[0], p[2], horizon))
        if len(self._pred_q) > 40:
            del self._pred_q[:-40]

    def _score(self, t: float, pos: Vec3):
        """
        예약된 예측을 실제 위치와 대조한다.

        ** 주의 (v6 초안의 버그) **
            예측 지평(tof + dt)은 0.9 s 같은 임의의 값인데 관측은 0.41 s
            간격으로만 들어온다. 예약 시각과 관측 시각이 최대 0.41 s 어긋나므로
            그대로 비교하면 20 m/s 표적에서 8 m 의 가짜 오차가 생긴다.
            -> 직전 관측과 현재 관측 사이를 선형 보간해서 비교한다.
        """
        if not self.hist:
            self._pred_q = []
            return
        t0, x0, _, z0 = self.hist[-1]
        span = t - t0
        if span <= 1e-6:
            return
        keep = []
        for (tt, px, pz, hz) in self._pred_q:
            if tt > t + 1e-6:
                keep.append((tt, px, pz, hz))
                continue
            if tt < t0 - 1e-6:
                continue                       # 너무 오래된 예약 - 버린다
            u = (tt - t0) / span               # 0..1 보간 계수
            ax = x0 + (pos[0] - x0) * u
            az = z0 + (pos[2] - z0) * u
            err = math.hypot(ax - px, az - pz)
            self._ks.append(err / max(1e-3, hz * hz))
            self.n_scored += 1
        self._pred_q = keep
        if len(self._ks) >= 8:
            # 평균이 아니라 상위 분위수를 쓴다.
            #
            # ** 왜 평균이면 안 되는가 **
            #   예측오차는 대부분의 시간 아주 작고, 표적이 조향을 바꾸는
            #   짧은 순간에만 크게 튄다(두꺼운 꼬리). 평균으로 sigma 를 잡으면
            #   P(hit) 이 늘 1.00 으로 나와 사격 게이트가 아예 작동하지 않는다.
            #   실측: 평균 기반 sigma 0.38 m vs 실제 예측오차 평균 1.31 m / p90 2.51 m
            s = sorted(self._ks)
            self.k_quant = s[min(len(s) - 1, int(len(s) * self.k_quantile))]
            self.k_mean = sum(s) / len(s)
            # 최근 몇 발의 예측이 실제로 얼마나 맞았는지 (즉시 반응하는 지표).
            # 표적이 방금 조향을 바꿨다면 여기부터 튀어오르므로,
            # '지금은 쏘면 안 되는 순간'을 실시간으로 잡아낸다.
            self.k_recent = max(list(self._ks)[-self.k_recent_n:])
            self.k_err = max(self.k_quant, self.k_recent * self.k_recent_w)

    def sigma(self, dt: float) -> Tuple[float, float]:
        """
        dt 뒤 예측오차 1-sigma (횡, 종) [m].
        실측 계수가 쌓이기 전에는 물리 한계 기반 값을 쓰고,
        쌓인 뒤에는 실측과 물리 한계의 작은 쪽을 쓴다.
        """
        # 물리 한계 기반 (v5 방식, 3-sigma 를 최대기동으로 봄)
        w = math.radians(self.lim.turn_max)
        lat_model = 0.5 * (max(self.speed, 1.0) * w) * dt * dt / 3.0
        lon_model = 0.5 * self.lim.a_max * dt * dt / 3.0
        if len(self._ks) >= 8:
            # 오차는 대부분 진행방향 수직(선회)에서 오므로 횡:종 = 1 : 0.55 로 배분.
            emp = self.k_err * dt * dt
            lat = min(lat_model, max(emp * 0.90, 0.05))
            lon = min(lon_model, max(emp * 0.55, 0.05))
        else:
            lat, lon = lat_model, lon_model
        return max(lat, 0.03), max(lon, 0.03)

    @property
    def speed(self) -> float:
        return math.hypot(self.vel[0], self.vel[2])

    @property
    def heading(self) -> float:
        return math.degrees(math.atan2(self.vel[0], self.vel[2])) % 360.0

    @property
    def turn_rate_dps(self) -> float:
        return math.degrees(self.omega)


def _polyfit(xs: List[float], ys: List[float], deg: int) -> Optional[List[float]]:
    """정규방정식 최소제곱. 반환 [c0, c1, c2]"""
    n = deg + 1
    if len(xs) < n:
        return None
    A = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = sum(x ** (i + j) for x in xs)
        A[i][n] = sum(y * (x ** i) for x, y in zip(xs, ys))
    # 가우스 소거
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-12:
            return None
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        for j in range(c, n + 1):
            A[c][j] /= pv
        for r in range(n):
            if r == c:
                continue
            f = A[r][c]
            if f:
                for j in range(c, n + 1):
                    A[r][j] -= f * A[c][j]
    out = [A[i][n] for i in range(n)]
    while len(out) < 3:
        out.append(0.0)
    return out


# ══════════════════════════════════════════════════════════
# 4. 자기 보정 (C5)
# ══════════════════════════════════════════════════════════
class BiasEstimator:
    """
    착탄점과 조준점의 잔차를 시선 좌표계(종/횡)로 분해해 적분한다.

    왜 필요한가
        v/g/h 는 21 점 회귀로 RMS 0.53 m 까지 맞췄지만, 이는 '평균적으로'
        맞다는 뜻이지 특정 거리대에서 0 이라는 뜻이 아니다. 또 발사 지연,
        포구 위치 오프셋, 명령-반영 타이밍은 모형에 아예 없다.
        실사격 잔차를 되먹임하면 이 잔여 오차가 자동으로 사라진다.

    안전장치
        - 게인을 낮게(0.30) 두고 상한을 걸어 발산을 막는다
        - 표본 3발 미만이면 보정을 적용하지 않는다
        - 이상치(|잔차| > 12 m)는 버린다  (장애물 피탄 등)
    """

    def __init__(self, gain: float = 0.30,
                 range_limit: float = 6.0, bearing_limit: float = 2.5,
                 min_samples: int = 3, outlier: float = 12.0):
        self.gain = gain
        self.range_limit = range_limit
        self.bearing_limit = bearing_limit
        self.min_samples = min_samples
        self.outlier = outlier
        self._range = 0.0
        self._bearing = 0.0
        self.n = 0
        self.hist: List[Tuple[float, float]] = []

    @property
    def range_bias(self) -> float:
        return self._range if self.n >= self.min_samples else 0.0

    @property
    def bearing_bias(self) -> float:
        return self._bearing if self.n >= self.min_samples else 0.0

    def observe(self, fire_pos: Vec3, aim_point: Vec3, impact: Vec3,
                expected_long: float = 0.0):
        """
        조준점 대비 착탄점 잔차를 학습.

        expected_long  일부러 길게 조준한 양 [m] (C9 의 lon_shift).
                       이만큼은 '오차'가 아니라 의도한 것이므로 빼고 학습한다.
                       빼지 않으면 보정기가 반대 방향으로 폭주한다.
        """
        aim_b = bearing(fire_pos, aim_point)
        aim_d = dist2d(fire_pos, aim_point)
        imp_b = bearing(fire_pos, impact)
        imp_d = dist2d(fire_pos, impact)
        d_rng = imp_d - aim_d - expected_long       # + = 멀리 감
        d_brg = ang_diff(imp_b, aim_b)              # + = 시계방향으로 밀림
        cross = math.radians(d_brg) * max(1.0, aim_d)
        if abs(d_rng) > self.outlier or abs(cross) > self.outlier:
            return
        self.n += 1
        self.hist.append((round(d_rng, 2), round(cross, 2)))
        del self.hist[:-60]
        g = self.gain
        self._range = max(-self.range_limit,
                          min(self.range_limit, self._range + g * d_rng))
        self._bearing = max(-self.bearing_limit,
                            min(self.bearing_limit, self._bearing + g * d_brg))

    def summary(self) -> str:
        if self.n == 0:
            return "표본 없음"
        r = [h[0] for h in self.hist[-20:]]
        c = [h[1] for h in self.hist[-20:]]
        mr = sum(r) / len(r)
        mc = sum(c) / len(c)
        return (f"n={self.n}  사거리보정 {self._range:+.2f} m  "
                f"방위보정 {self._bearing:+.3f}°  "
                f"최근잔차 종 {mr:+.2f} / 횡 {mc:+.2f} m")


# ══════════════════════════════════════════════════════════
# 5. 표적 투영 치수 (C3)
# ══════════════════════════════════════════════════════════
def target_extents(los_bearing: float, target_heading: float,
                   size: TargetSize) -> Tuple[float, float]:
    """
    시선 방향 기준 표적 사각형의 반치수.
      반환 (half_lat, half_lon)  [m]
        half_lat  시선에 수직한 방향 (방위 오차가 먹는 폭)
        half_lon  시선 방향        (앙각/사거리 오차가 먹는 깊이)

    정면으로 볼 때  half_lat = W/2 = 1.80,  half_lon = L/2 = 3.75
    측면으로 볼 때  half_lat = L/2 = 3.75,  half_lon = W/2 = 1.80
    v5 는 항상 (1.80, 3.75) 로 고정해, 측면 표적에서 종방향을 2 배 낙관했다.
    """
    d = math.radians(los_bearing - target_heading)
    c, s = abs(math.cos(d)), abs(math.sin(d))
    half_lon = 0.5 * (size.length * c + size.width * s)
    half_lat = 0.5 * (size.length * s + size.width * c)
    # 위 값은 회전 사각형의 '외접 상자'라 비스듬한 자세에서 면적을 과대평가한다.
    # 같은 면적을 갖는 등가 사각형으로 축소해 모서리 밖을 명중으로 세지 않는다.
    box = 4.0 * half_lat * half_lon
    rect = size.width * size.length
    if box > rect > 0:
        k = math.sqrt(rect / box)
        half_lat *= k
        half_lon *= k
    return half_lat, half_lon


def longitudinal_window(bal: "Ballistics", theta_deg: float, dy: float,
                       half_lon: float, height: float,
                       long_cap: Optional[float] = None) -> Tuple[float, float]:
    """
    종방향(시선 방향) 허용 오차. **비대칭이다.**  (C9)

    왜 비대칭인가
        탄은 완만한 하강 궤적으로 들어온다(착탄 직전 낙하각 6.6~11.8도).
        조준이 짧으면 탄이 표적 앞 지면에 박혀 무조건 빗나간다.
        조준이 길면 탄은 아직 공중에 있고, 차체 높이 안에 들어오는 동안
        차체 윗부분을 때린다. 40 m 에서 1 m 낙하에 8.6 m 를 가므로
        높이 2.2 m 는 종방향으로 무려 19 m 의 여유를 만든다.

        v6 초안은 이것을 대칭 ±반길이(3.75 m)로 모델링했다.
        실제보다 5 배 보수적이었고, 그래서 P(hit) 을 과소평가했다.

    ── C12 (2026-08-06 실사격 51발) : 이 기하 모델은 실측과 어긋난다 ──

        위 논리는 기하학적으로는 옳지만 시뮬레이터의 명중 판정이
        그렇게 관대하지 않다. 실측이 그것을 부정했다.

          로그의 lon_long        21.5 ~ 26.0 m  (중앙값 25.0)
          실제로 명중한 최대 종오차      +3.43 m
          실패탄 3발의 종오차     +9.7 / +11.6 / +11.8 m

        즉 모델은 25 m 를 허용한다고 믿었지만 실제 경계는 3.5 m 근처다.
        7 배 낙관이다. 이 낙관 때문에 두 가지가 망가졌다.

          1) lon_shift = (l_long - l_short) * 0.5 * lon_gain 이
             매 발 +2.7 m 를 만들어, 봇이 스스로 길게 조준했다.
             (51발 전부 lon_shift 2.21~2.79 m)
          2) P(hit) 이 51발 전부 1.00 이 되어 사격 게이트가 무력화됐다.
             명중탄도 1.00, 실패탄도 1.00 이라 아무것도 걸러내지 못한다.

        근접전에서 특히 심하다. 포신이 아래를 향하면(aim_elev -1.0~-1.7°)
        slope 가 0 에 가까워 height/slope 가 발산한다.
        실패 3발이 모두 거리 24~27 m 였던 이유다.

        그래서 실측 상한(long_cap)을 둔다. 상면 관통을 부정하는 것이 아니라,
        '측정으로 확인된 범위까지만 믿는다'는 뜻이다.
        더 넓은 창이 실제로 존재한다면 lon_cap 을 올려가며 재측정하면 된다.

    반환 (짧은 쪽 허용, 긴 쪽 허용) [m]
    """
    p = bal.p
    t = bal.flight(theta_deg, dy)
    if math.isnan(t) or t <= 0:
        return half_lon, half_lon
    s_, c_, _ = bal._geom(theta_deg, dy)
    vy = p.v * s_ - p.g * t          # 착탄 순간 수직 속도 (음수)
    vx = max(1e-6, p.v * c_)
    slope = abs(vy) / vx             # 낙하 기울기
    if slope < 1e-6:
        return half_lon, half_lon
    l_long = half_lon + height / slope
    if long_cap is not None:
        l_long = min(l_long, max(half_lon, long_cap))
    return half_lon, l_long


def required_yaw_rate(v_cross: float, dist_m: float) -> float:
    """
    횡단 속도 v_cross [m/s] 인 표적을 거리 dist_m 에서 추적하는 데 필요한
    포탑 각속도 [deg/s].  = v_cross / d  [rad/s]

    이것이 포탑 최대 각속도(40 deg/s)를 넘으면 조준이 원리적으로 불가능하다.
    20 m/s 로 횡단하는 표적은 28.6 m 안쪽에서는 절대 못 맞힌다.
    """
    return math.degrees(v_cross / max(1.0, dist_m))


def desired_range(v_cross: float, yaw_rate: float = 40.0,
                  duty: float = 0.45, lo: float = 30.0, hi: float = 70.0) -> float:
    """
    표적 횡단 속도만 보고 정하는 단순 교전 거리 [m]. (optimal_range 의 대체/하한)

    상충 관계
      가까울수록  비행시간이 짧아 예측오차가 작다  (오차 ∝ t^2)
      멀수록      필요한 포탑 각속도가 작아 추적 여유가 생긴다
    """
    if v_cross <= 0.2:
        return lo
    need = math.degrees(v_cross) / max(1e-6, yaw_rate * duty)
    return max(lo, min(hi, need))


def optimal_range(bal: "Ballistics", tracker: Optional["TargetTracker"],
                  half_lat: float, half_lon: float, v_cross: float,
                  turret: "TurretParams", duty_max: float = 0.85,
                  lo: float = 24.0, hi: float = 95.0,
                  step: float = 2.0) -> Tuple[float, float]:
    """
    예상 명중확률이 가장 높은 교전 거리를 직접 찾는다. (B8)

    왜 고정 밴드가 아닌가
        두 제약이 반대 방향으로 작용한다.
          예측오차  sigma ~ k * (dt + tof(d))^2      -> 가까울수록 좋다
          포탑 추적 duty = (v_cross / d) / yaw_rate  -> 멀수록 좋다
        어느 쪽이 지배하는지는 표적 속도에 따라 완전히 달라진다.
        정지표적이면 최소 사거리 근처가 최선이고, 20 m/s 로 횡단하는
        표적이면 60 m 밖이 최선이다. 고정 밴드로는 둘 다 만족할 수 없다.

    반환 (최적거리, 그때의 예상 P(hit))
    """
    best_p, best_d = -1.0, lo
    rmin = bal.min_range()
    rmax = bal.max_range()[0]
    d = max(lo, rmin + 1.5)
    hi = min(hi, rmax - 1.5)
    while d <= hi + 1e-9:
        th = bal.solve_elevation_cached(d)
        if th is not None:
            duty = math.degrees(v_cross / d) / turret.yaw_rate
            if duty <= duty_max:
                tof = bal.flight(th)
                h = turret.dt + tof
                sl, sn = tracker.sigma(h) if tracker else (0.05, 0.05)
                # 포탑 분해능이 만드는 조준 잔차도 더한다
                q_lat = math.radians(turret.yaw_min_step * 0.5) * d
                q_lon = turret.pitch_min_step * 0.5 * bal.dr_dtheta(th)
                p = hit_probability(math.hypot(sl, q_lat * 0.6),
                                    math.hypot(sn, q_lon * 0.6),
                                    half_lat, half_lon)
                # 같은 확률이면 가까운 쪽 (반응 시간·피탄 부위에 유리)
                if p > best_p + 1e-4:
                    best_p, best_d = p, d
        d += step
    return best_d, max(0.0, best_p)


def band_prob(lo: float, hi: float, mu: float, sig: float) -> float:
    """[lo, hi] 구간에 N(mu, sig) 이 들어갈 확률"""
    if sig < 1e-6:
        return 1.0 if lo <= mu <= hi else 0.0
    return _phi((hi - mu) / sig) - _phi((lo - mu) / sig)


def hit_probability_asym(sig_lat: float, sig_lon: float,
                         half_lat: float,
                         lon_short: float, lon_long: float,
                         bias_lat: float = 0.0, bias_lon: float = 0.0) -> float:
    """
    C9: 종방향을 비대칭 구간으로 다루는 명중 확률.
        bias_lon 이 양수면 '길게' 조준된 상태 (표적 중심 기준).
    """
    return (band_prob(-half_lat, half_lat, bias_lat, sig_lat) *
            band_prob(-lon_short, lon_long, bias_lon, sig_lon))


def hit_probability(sig_lat: float, sig_lon: float,
                    half_lat: float, half_lon: float,
                    bias_lat: float = 0.0, bias_lon: float = 0.0) -> float:
    """예측/조준 오차가 정규분포일 때 표적 사각형 안에 들어갈 확률"""
    def band(half, sig, bias):
        if sig < 1e-6:
            return 1.0 if abs(bias) <= half else 0.0
        return _phi((half - bias) / sig) - _phi((-half - bias) / sig)
    return band(half_lat, sig_lat, bias_lat) * band(half_lon, sig_lon, bias_lon)


def impact_aspect(fire_pos: Vec3, impact_pos: Vec3, target_heading_deg: float) -> float:
    """피탄 부위 판정용 상대각. 0=전면, ±90=측면, 180=후면"""
    ax = fire_pos[0] - impact_pos[0]
    az = fire_pos[2] - impact_pos[2]
    approach = math.degrees(math.atan2(ax, az))
    return ((approach - target_heading_deg + 180.0) % 360.0) - 180.0


def aspect_zone(aspect_deg: float) -> str:
    a = abs(aspect_deg)
    if a <= 60.0:
        return "front"
    if a >= 120.0:
        return "rear"
    return "side"


# ══════════════════════════════════════════════════════════
# 6. 사격 해
# ══════════════════════════════════════════════════════════
@dataclass
class Solution:
    valid: bool
    bearing: float = 0.0        # 명령할 포탑 방위 (바이어스 보정 포함)
    bearing_geo: float = 0.0    # 보정 전 기하 방위
    elevation: float = 0.0      # 명령할 앙각
    distance: float = 0.0       # 조준점까지 실제 거리
    flight: float = 0.0         # 비행 시간
    lead: float = 0.0           # 리드 거리
    aim_point: Vec3 = (0, 0, 0)
    drdt: float = 8.0           # dR/dtheta [m/deg]
    half_lat: float = 1.8
    half_lon: float = 3.75
    lon_short: float = 3.75     # 짧은 쪽 허용 오차 [m]  (C9)
    lon_long: float = 3.75      # 긴 쪽 허용 오차 [m]
    lon_shift: float = 0.0      # 창 중심을 맞추려고 길게 민 양 [m]
    sig_lat: float = 0.0
    sig_lon: float = 0.0
    p_hit: float = 1.0
    reason: str = ""


# ══════════════════════════════════════════════════════════
# 7. 사격 통제
# ══════════════════════════════════════════════════════════
class FireControl:
    """
    상태:
      IDLE    표적 없음 / 포락 밖
      SLEW    조준 중
      HOLD    조준은 됐으나 P(hit) 부족 - 표적이 예측 불가하게 기동 중
      SETTLE  발사 직전 포탑 정지 틱
      FIRE    발사
      RELOAD  재장전 대기
    """

    def __init__(self,
                 ballistics: Optional[Ballistics] = None,
                 turret: Optional[TurretParams] = None,
                 reload_s: float = 6.55,
                 target: Optional[TargetSize] = None,
                 bias: Optional[BiasEstimator] = None,
                 p_hit_min: float = 0.62,
                 settle_ticks: int = 0,
                 db_safety: float = 0.5,
                 track_duty_max: float = 0.80,
                 fire_delay: Optional[float] = None,
                 patience: float = 18.0,
                 p_hit_floor: float = 0.28,
                 tof_max: Optional[float] = None,
                 body_rate_max: Optional[float] = None,
                 jitter_cap_yaw: float = 0.9,
                 jitter_cap_pitch: float = 0.35,
                 lon_gain: float = 0.8,
                 lon_shift_max: float = 7.0,
                 lon_long_max: Optional[float] = None,
                 pitch_db_max: float = 0.60):
        self.bal = ballistics or Ballistics()
        self.t = turret or TurretParams()
        self.tgt = target or TargetSize()
        self.bias = bias or BiasEstimator()
        self.reload_s = reload_s
        self.p_hit_min = p_hit_min
        self.patience = patience        # 이 시간 넘게 못 쏘면 임계값을 낮춘다 [s]
        self.p_hit_floor = p_hit_floor  # 낮출 수 있는 하한
        self.tof_max = tof_max                   # 비행시간 상한 [s] (C11)
        self.body_rate_max = body_rate_max        # 차체 회전율 상한 [deg/s] (C13)
        self.jitter_cap_yaw = jitter_cap_yaw      # 지터 보정 상한 [deg]
        self.jitter_cap_pitch = jitter_cap_pitch  # 지터 보정 상한 [deg]
        # C9 의도적 '길게 조준'
        self.lon_gain = lon_gain            # 0 이면 끔, 1 이면 창 정중앙
        self.lon_shift_max = lon_shift_max  # 최대로 밀 거리 [m]
        # C12: 상면 관통 창의 실측 상한 [m]. None 이면 기하 모델 그대로.
        self.lon_long_max = lon_long_max
        self.pitch_db_max = pitch_db_max    # 앙각 데드밴드 상한 [deg]
        self._t0: Optional[float] = None
        self.settle_ticks = settle_ticks
        self.db_safety = db_safety
        self.track_duty_max = track_duty_max
        self.track_duty = 0.0       # 현재 표적 추적에 쓰이는 포탑 용량 비율
        # 관측 시각 -> 실제 발사까지의 지연 [s].
        #
        # ** P5 실측 (2026-08-04) **
        #   포탑을 40 deg/s 로 돌리며 사격하고 착탄 방위를 역산한 결과,
        #   탄은 마지막 /info 스냅샷보다 평균 0.125 s 뒤의 포탑 각도로 나간다.
        #   (개별값 0.084 ~ 0.171 s, n=8).  /info 주기가 0.130 s 이므로
        #   이 지연은 사실상 '텔레메트리가 한 주기 낡은 것'과 같다.
        #   v6 초안의 추정값 dt/2 = 0.205 s 보다 0.08 s 짧다
        #   -> 10 m/s 표적에서 리드가 0.8 m 어긋나던 것을 제거.
        self.fire_delay = 0.125 if fire_delay is None else fire_delay

        self.yaw_db = self.t.yaw_min_step * 0.5
        self.pitch_db = self.t.pitch_min_step * 0.5
        self.state = "IDLE"
        self.last_fire_t: Optional[float] = None
        self.last_solution: Optional[Solution] = None
        self._settled = 0
        self.yaw_err = 0.0
        self.pitch_err = 0.0

    # ── 조준 해 ──────────────────────────────────────────
    def solve(self, my: Vec3, target_now: Vec3,
              predictor: Optional[Callable[[float], Vec3]] = None,
              target_heading: Optional[float] = None,
              tracker: Optional[TargetTracker] = None,
              extra_delay: float = 0.0,
              lead_offset: float = 0.0,
              iters: int = 6) -> Solution:
        """
        예측 조준. 비행시간을 고정점 반복으로 수렴시킨다.

        extra_delay  관측 시각에서 실제 발사까지의 지연 [s].
                     표적은 이 시간 동안에도 움직이므로 예측 지평은
                     (지연 + 비행시간) 이다.  v5 는 이 항이 없었다.
        lead_offset  '한 틱 뒤에 발사한다면' 을 계산할 때 쓰는 추가 오프셋.

        ** v5/v6 초안의 버그 **
            포탑 명령용 해(sol_n)를 만들 때 '한 틱 뒤 표적 위치'를 인자로만
            넘기고, 예측기는 여전히 현재 시각 기준으로 호출했다.
            결과적으로 sol_n == sol 이 되어 피드포워드가 사라졌고,
            포탑 제어가 순수 P 제어가 되어 등속 이동하는 방위각(램프 입력)에
            정상상태 오차 (방위각속도 x dt) 를 남겼다.
            횡단 20 m/s @ 50 m 이면 22.8 deg/s x 0.41 s = 9.3 deg 의
            조준 오차가 영원히 사라지지 않아 한 발도 못 쐈다.
        """
        horizon0 = lead_offset + extra_delay
        aim = predictor(horizon0) if (predictor and horizon0 > 0) else target_now
        th = None
        tof = 0.0
        for _ in range(iters):
            d = dist2d(my, aim)
            dy = my[1] - aim[1]
            d_cmd = d - self.bias.range_bias           # C5 사거리 보정
            th = self.bal.solve_elevation_cached(d_cmd, dy)
            if th is None:
                rmax, _ = self.bal.max_range(dy)
                rmin = self.bal.min_range(dy)
                why = "사거리 초과" if d_cmd > rmax else "너무 가까움"
                return Solution(False,
                                reason=f"{why} ({d:.0f}m, 포락 {rmin:.0f}~{rmax:.0f}m)")
            tof = self.bal.flight(th, dy)
            if predictor is not None:
                aim = predictor(horizon0 + tof)
            else:
                break

        d = dist2d(my, aim)
        dy = my[1] - aim[1]
        drdt = self.bal.dr_dtheta(th, dy)
        geo_b = bearing(my, aim)
        hd = target_heading
        if hd is None and tracker is not None and tracker.speed > 0.5:
            hd = tracker.heading
        if hd is None:
            hd = geo_b                                  # 자세 불명 -> 정면으로 가정
        hl, hn = target_extents(geo_b, hd, self.tgt)
        sl, sn = (tracker.sigma(horizon0 + tof) if tracker else (0.0, 0.0))

        # C9: 종방향 허용 오차는 비대칭이다 (짧으면 지면, 길면 차체 윗부분)
        l_short, l_long = longitudinal_window(self.bal, th, dy, hn,
                                              self.tgt.height, self.lon_long_max)
        # 창의 중심에 오도록 일부러 길게 민다. lon_gain 으로 세기 조절.
        shift = max(0.0, min(self.lon_shift_max,
                             (l_long - l_short) * 0.5 * self.lon_gain))
        if shift > 0.05:
            th2 = self.bal.solve_elevation_cached(d - self.bias.range_bias + shift, dy)
            if th2 is not None:
                th = th2
                tof = self.bal.flight(th, dy)
                drdt = self.bal.dr_dtheta(th, dy)
                l_short, l_long = longitudinal_window(self.bal, th, dy, hn,
                                                      self.tgt.height,
                                                      self.lon_long_max)
            else:
                shift = 0.0

        return Solution(True,
                        bearing=(geo_b - self.bias.bearing_bias) % 360.0,
                        bearing_geo=geo_b,
                        elevation=th,
                        distance=d,
                        flight=tof,
                        lead=dist2d(target_now, aim),
                        aim_point=aim,
                        drdt=drdt,
                        half_lat=hl, half_lon=hn,
                        lon_short=l_short, lon_long=l_long, lon_shift=shift,
                        sig_lat=sl, sig_lon=sn)

    # ── 데드밴드 (C3 + C4) ───────────────────────────────
    def deadbands(self, sol: Solution) -> Tuple[float, float]:
        """
        '예측오차가 먹고 남은 여유'만큼만 조준오차를 허용한다.
        v5 는 표적 치수의 고정 비율(0.55)이라 예측이 나쁜 상황에서도
        똑같이 헐거웠다.
        """
        d = max(1.0, sol.distance)
        lat_room = max(0.25, sol.half_lat - 1.6 * sol.sig_lat) * self.db_safety
        # C9: 종방향은 비대칭 창의 반폭을 쓴다 (조준을 창 중심에 맞춰 뒀으므로)
        lon_half = 0.5 * (sol.lon_short + sol.lon_long)
        lon_room = max(0.25, lon_half - 1.6 * sol.sig_lon) * self.db_safety
        yaw = math.degrees(math.atan(lat_room / d))
        pitch = min(lon_room / sol.drdt, self.pitch_db_max)
        # 포탑 분해능보다 작게 잡으면 영원히 수렴하지 않는다
        yaw = max(yaw, self.t.yaw_min_step * 0.5)
        pitch = max(pitch, self.t.pitch_min_step * 0.5)
        return yaw, pitch

    # ── 사격 임계값 (인내 로직) ──────────────────────────
    def threshold(self, now: float) -> float:
        """
        오래 쏘지 못하면 임계 명중확률을 점진적으로 낮춘다.

        왜 필요한가
            아주 빠르게 횡단하는 표적(20 m/s 선회)은 어떤 거리에서도
            예상 P(hit) 이 0.5 를 넘지 않는다. 고정 임계값만 쓰면
            봇이 '완벽한 기회'를 기다리다 한 발도 쏘지 않고 끝난다.
            일정 시간이 지나면 '가능한 최선의 사격'으로 태세를 바꾼다.
        """
        if self.patience <= 0:
            return self.p_hit_min
        ref = self.last_fire_t if self.last_fire_t is not None else self._t0
        if ref is None:
            self._t0 = now
            return self.p_hit_min
        idle_s = now - ref - self.reload_s
        if idle_s <= self.patience:
            return self.p_hit_min
        u = min(1.0, (idle_s - self.patience) / max(1e-6, self.patience))
        return self.p_hit_min + (self.p_hit_floor - self.p_hit_min) * u

    # ── 포탑 명령 ────────────────────────────────────────
    def _slew(self, err: float, rate: float, deadband: float) -> Tuple[str, float]:
        if abs(err) <= deadband:
            return "", 0.0
        w = abs(err) / (rate * self.t.dt)
        w = max(self.t.w_min, min(self.t.w_max, w))
        return ("pos" if err > 0 else "neg"), round(w, self.t.w_round)

    # ── 메인 ─────────────────────────────────────────────
    def update(self,
               my_pos: Vec3, turret_x: float, turret_y: float,
               target_pos: Optional[Vec3],
               tracker: Optional[TargetTracker] = None,
               target_heading: Optional[float] = None,
               sim_time: float = 0.0,
               hull_settled: bool = True,
               my_vel: Vec3 = (0.0, 0.0, 0.0),
               body_rate_dps: float = 0.0,
               allow_moving_fire: bool = False,
               inhibit_fire: bool = False) -> Dict:
        """
        매 /get_action 마다 호출.

        hull_settled  차체가 완전히 멈춰 있는가. 포탑이 비안정화라
                      차체가 움직이면 발사 순간 포탑 월드각이 틀어진다.
                      allow_moving_fire=False 면 이때 사격을 보류한다.
        inhibit_fire  상위 계층의 하드 사격 금지. 조준은 계속하되 쏘지 않는다.
                      (v7 B15: 사거리 시험에서 목표 사거리를 벗어난 지점의
                       사격을 원천 차단하는 데 쓴다. 상위에서 want_shoot 을
                       거짓으로 만들어도, 차체가 우연히 멈춰 있으면
                       여기까지 내려와 발사되어 버리기 때문이다.)
        """
        idle = {"turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False}

        if target_pos is None:
            self.state = "IDLE"
            self._settled = 0
            return idle

        dt = self.t.dt
        predictor = tracker.predict if (tracker and tracker.pos is not None) else None

        # (A) 발사 판정용 해 - '지금 각도 그대로 이 틱 안에 발사한다'는 가정
        fd = self.fire_delay
        my_fire = (my_pos[0] + my_vel[0] * fd,
                   my_pos[1] + my_vel[1] * fd,
                   my_pos[2] + my_vel[2] * fd)
        sol = self.solve(my_fire, target_pos, predictor, target_heading,
                         tracker, extra_delay=fd)
        self.last_solution = sol
        if not sol.valid:
            self.state = "IDLE"
            self._settled = 0
            return idle

        # (B) 포탑 명령용 해 - 이번 명령이 반영되는 시각(t+dt)에
        #     '그때 발사한다면' 필요한 조준을 미리 맞춰 둔다.
        #     lead_offset=dt 로 예측 지평을 한 틱 밀어 램프 추종 오차를 없앤다.
        my_next = (my_pos[0] + my_vel[0] * dt,
                   my_pos[1] + my_vel[1] * dt,
                   my_pos[2] + my_vel[2] * dt)
        sol_n = self.solve(my_next, target_pos, predictor, target_heading,
                           tracker, extra_delay=fd, lead_offset=dt)
        if not sol_n.valid:
            sol_n = sol

        self.yaw_db, self.pitch_db = self.deadbands(sol)

        # 1틱 타이밍 지터 보정
        #
        #   발사 판정은 sol(지금 쏜다), 포탑 명령은 sol_n(한 틱 뒤에 쏜다)로
        #   서로 다른 해를 쓴다. 표적이 빠르게 움직이면 두 해가 구조적으로
        #   어긋나고(실측: 선회 20 m/s 표적에서 앙각 0.31 deg = 사거리 2.2 m),
        #   포탑은 sol_n 에 수렴하므로 sol 기준 오차가 그 값에서 멈춘다.
        #   -> 데드밴드가 그보다 좁으면 '정렬됨' 판정이 영원히 나지 않아
        #      한 발도 쏘지 못한다. 자기 자신의 1틱 불확실성보다 정밀한
        #      조준을 요구하지 않도록 데드밴드 하한을 올린다. (상한은 건다)
        jy = abs(ang_diff(sol_n.bearing, sol.bearing))
        jp = abs(sol_n.elevation - sol.elevation)
        self.yaw_db = max(self.yaw_db, min(0.6 * jy, self.jitter_cap_yaw))
        self.pitch_db = max(self.pitch_db, min(0.6 * jp, self.jitter_cap_pitch))

        # 추적 가능성: 표적 횡단 성분이 만드는 시선 각속도가
        # 포탑 최대 각속도를 넘으면 조준 자체가 성립하지 않는다.
        self.track_duty = 0.0
        if tracker is not None and tracker.speed > 0.2:
            los = math.radians(sol.bearing_geo)
            # 시선에 수직인 단위벡터
            cx, cz = math.cos(los), -math.sin(los)
            v_cross = abs(tracker.vel[0] * cx + tracker.vel[2] * cz)
            v_cross = math.hypot(v_cross,
                                 abs(my_vel[0] * cx + my_vel[2] * cz))
            need = required_yaw_rate(v_cross, sol.distance)
            self.track_duty = need / self.t.yaw_rate

        # ── C10 (v8): 발사 판정은 '탄이 떠나는 순간'의 포탑 각도로 ──
        #
        # 왜 바꾸는가 (2026-08-06 실사격 73발 분해)
        #   포탑은 비안정화라 차체가 돌면 함께 돈다. fire=True 를 낸 뒤
        #   탄이 실제로 나가기까지 fire_delay(0.125 s) 가 걸리고,
        #   그 사이 차체가 body_rate 만큼 더 돈다.
        #   40 deg/s 로 선회 중이면 0.125 s 에 5 deg = 38 m 에서 3.3 m 다.
        #   기존 코드는 '지금 이 순간'의 turret_x 로 판정해 이 몫을 무시했다.
        #
        #   실측: 이동-이동 73발에서 사격 오차(착탄-조준점) 표준편차가
        #         횡 2.47 m 로 예측 오차(1.13 m)보다 컸다.
        #         데드밴드가 허용하는 값(±0.67 m)의 3.7 배다.
        turret_x_exit = turret_x + body_rate_dps * self.fire_delay
        yaw_err = ang_diff(sol.bearing, turret_x_exit)
        pitch_err = sol.elevation - turret_y
        self.yaw_err, self.pitch_err = yaw_err, pitch_err

        # 조준 오차를 미터로 환산해 P(hit) 를 계산 (C6)
        bias_lat = math.radians(yaw_err) * max(1.0, sol.distance)
        # 표적 중심 기준 종방향 조준 위치 = 의도한 길게 밀기 + 앙각 오차분
        bias_lon = sol.lon_shift + pitch_err * sol.drdt
        sol.p_hit = hit_probability_asym(sol.sig_lat, sol.sig_lon,
                                         sol.half_lat,
                                         sol.lon_short, sol.lon_long,
                                         bias_lat, bias_lon)

        # 명령 오차 (한 틱 뒤 목표 - 차체 회전분 피드포워드)
        body_rot = body_rate_dps * dt
        yaw_cmd_err = ang_diff(sol_n.bearing, turret_x) - body_rot
        pitch_cmd_err = sol_n.elevation - turret_y
        ydir, yw = self._slew(yaw_cmd_err, self.t.yaw_rate, self.yaw_db)
        pdir, pw = self._slew(pitch_cmd_err, self.t.pitch_rate, self.pitch_db)

        # 발사 가부는 '지금 이 순간의 조준 오차'로만 판단한다.
        #
        # ** v6 초안의 설계 결함 **
        #   '다음 틱 포탑 명령도 0일 것'을 함께 요구했는데, 이동표적은
        #   포탑이 계속 따라가야 하므로 그 조건이 영원히 참이 되지 않는다.
        #   -> 이동표적에 한 발도 못 쐈다. v5 의 판정(y_ok/p_ok)이 옳았다.
        aligned = (abs(yaw_err) <= self.yaw_db and abs(pitch_err) <= self.pitch_db)

        reloading = (self.last_fire_t is not None
                     and sim_time - self.last_fire_t < self.reload_s)

        def turret_cmd(y_d, y_w, p_d, p_w):
            return {"turretQE": {"command": ("E" if y_d == "pos" else "Q") if y_d else "",
                                 "weight": y_w},
                    "turretRF": {"command": ("R" if p_d == "pos" else "F") if p_d else "",
                                 "weight": p_w},
                    "fire": False}

        if self.track_duty > self.track_duty_max:
            # 포탑이 표적 각속도를 따라갈 수 없다 -> 조준은 계속하되 사격 금지.
            # 기동 계층이 이 값을 보고 거리를 벌린다.
            self.state = "NOTRACK"
            self._settled = 0
            return turret_cmd(ydir, yw, pdir, pw)

        if inhibit_fire:
            # 조준은 유지하되 발사만 막는다
            self.state = "INHIBIT"
            self._settled = 0
            return turret_cmd(ydir, yw, pdir, pw)

        if not aligned:
            self.state = "SLEW"
            self._settled = 0
            return turret_cmd(ydir, yw, pdir, pw)

        if reloading:
            self.state = "RELOAD"
            self._settled = 0
            return idle

        if not hull_settled and not allow_moving_fire:
            self.state = "SETTLE"          # 차체가 아직 움직인다 - 대기
            self._settled = 0
            return idle

        # ── C11 (v8): 비행시간 상한 게이트 ──
        #
        #   실측(이동-이동 73발) — 비행시간이 명중률을 가른다:
        #     ~0.35 s   90.9%    사격 횡오차 sigma 1.23 m
        #     ~0.50 s   87.0%                      1.67 m
        #     ~0.65 s   60.9%                      2.97 m
        #     ~1.50 s   25.0%                      3.16 m
        #   0.5 s 를 넘으면 급격히 무너진다. 오차가 비행시간에 비례해
        #   커지므로, 긴 비행은 확률이 아니라 구조적으로 불리하다.
        #   표적이 움직일 때만 적용한다 (정지표적은 125 m 에서도 100% 였다).
        if (self.tof_max is not None and tracker is not None
                and tracker.speed > 0.6 and sol.flight > self.tof_max):
            self.state = "TOFCAP"          # 너무 멀다 - 접근해서 쏜다
            self._settled = 0
            return turret_cmd(ydir, yw, pdir, pw)

        # ── C13 (v10): 차체 선회 중 사격 금지 ──────────────────
        #
        # 근거 (2026-08-06 실사격 116발, tag=move_move_v8 + base_60)
        #   발사 순간의 |차체 회전율| 로 나누면 완전히 갈린다.
        #
        #     |body_rate|      발수   명중률   |횡오차| 평균
        #        0~ 5 deg/s     42   100.0%      0.70 m
        #       25~30            3   100.0%      0.77 m
        #       30~90           71    77.5%      1.87 m
        #
        #   사후 게이트를 걸면 45/45 = 100% (95% 하한 92.1%).
        #   조건별로도 이동-이동 21/21, 이동-정지 20/20 이다.
        #
        # 교란이 아니다. 아군 속도로 층화해도 회전만 낮으면 100% 다.
        #     회전<15 & 아속<9  : 34/34 100%
        #     회전<15 & 아속 9+ :  8/ 8 100%
        #     회전 15+ & 아속<9 : 24/28  86%
        #     회전 15+ & 아속 9+: 34/46  74%
        #   '우리가 빠르면 못 맞힌다'가 아니라 '우리가 돌면 못 맞힌다' 다.
        #
        # 왜 C10 으로 안 되었나
        #   C10 은 발사 지연 동안의 회전을 1차(선형)로 보정한다.
        #   실제로는 (a) body_rate 자체가 변하고(각가속),
        #   (b) 포탑이 40 deg/s 를 차체 상쇄에 다 써서 표적 추종 여유가
        #   없어진다(track_duty 상관 +0.425 로 최대). 선형 보정으로는
        #   메울 수 없는 몫이다.
        #   실제로 116발 전부 yaw_err 가 데드밴드 안이었다(0/116 초과).
        #   봇은 매번 '조준 완료'로 판단하고 쐈다. 즉 데드밴드 문제가
        #   아니라 회전 중에는 해 자체가 틀린다.
        #
        # 임계값 15 deg/s 를 고른 이유
        #   차체는 직진(0~5) 아니면 최대선회(40~45) 뿐이고 그 사이가 비었다.
        #   5~30 사이 어디에 두어도 결과가 같으므로 가운데를 잡았다.
        #
        # 대가: 사격 기회의 61% 를 버린다. 재장전이 6.4 s 라 선회가
        #       끝나기를 기다릴 시간은 있다. 사격 수가 급감하면
        #       body_rate_max 를 올리거나 기동 계층에서 '조준 중엔
        #       직진' 을 넣어야 한다.
        if (self.body_rate_max is not None
                and abs(body_rate_dps) > self.body_rate_max):
            self.state = "TURNING"         # 차체 선회 중 - 멎기를 기다린다
            self._settled = 0
            return turret_cmd(ydir, yw, pdir, pw)

        if sol.p_hit < self.threshold(sim_time):
            self.state = "HOLD"            # 예측 불확실 - 좋은 창을 기다린다
            self._settled = 0
            return idle

        # C7: 발사 직전 포탑 정지 틱. 회전 명령과 fire 를 같이 내면
        #     발사 순간의 포탑 각도가 명령만큼 밀린 상태가 된다.
        if self._settled < self.settle_ticks:
            self._settled += 1
            self.state = "SETTLE"
            return idle

        self.state = "FIRE"
        self._settled = 0
        self.last_fire_t = sim_time
        return {"turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": True}

    # ── 진단 ─────────────────────────────────────────────
    def status(self, turret_x: float, turret_y: float) -> str:
        s = self.last_solution
        if s is None or not s.valid:
            return f"{self.state}  {'' if s is None else s.reason}"
        return (f"{self.state}  d {s.distance:6.1f}m  "
                f"brg {s.bearing:6.1f}/{turret_x:6.1f}  "
                f"el {s.elevation:+5.2f}/{turret_y:+5.2f}  "
                f"tof {s.flight:.2f}s  lead {s.lead:.1f}m  P {s.p_hit:.2f}")


# ══════════════════════════════════════════════════════════
# 하위 호환 (v5 코드가 import 하던 이름)
# ══════════════════════════════════════════════════════════
class VelocityTracker(TargetTracker):
    pass


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    bal = Ballistics()
    rmax, th = bal.max_range()
    rmin = bal.min_range()
    print(f"교전 포락 {rmin:.1f} ~ {rmax:.1f} m  (앙각 "
          f"{bal.p.theta_min:+.1f} ~ {bal.p.theta_max:+.1f} deg)\n")

    t = TurretParams()
    print(f"{'거리':>6s} {'앙각':>8s} {'비행t':>7s} {'dR/dth':>8s} "
          f"{'최소앙각스텝→사거리':>12s} {'방위db→횡':>10s}")
    for d in (25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120, 130):
        th = bal.solve_elevation(d)
        if th is None:
            print(f"{d:5d}m  교전 포락 밖")
            continue
        drdt = bal.dr_dtheta(th)
        print(f"{d:5d}m {th:7.2f}° {bal.flight(th):6.2f}s {drdt:7.2f} "
              f"{drdt*t.pitch_min_step:11.2f}m "
              f"{math.radians(t.yaw_min_step)*d:9.2f}m")

    print("\n표적 투영 치수 (시선-표적방위 상대각별)")
    ts = TargetSize()
    for rel in (0, 30, 45, 60, 90, 135, 180):
        hl, hn = target_extents(0.0, -rel, ts)
        print(f"  상대각 {rel:3d}°  반횡 {hl:5.2f} m  반종 {hn:5.2f} m")
