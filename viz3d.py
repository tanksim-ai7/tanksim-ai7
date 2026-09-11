# -*- coding: utf-8 -*-
"""
viz3d.py — 전술 3D 상황도  (주소는 아래 VIEW_PREFIX 가 정한다)

설계 원칙
    · 다른 팀 코드를 한 줄도 고치지 않는다.
    · fm / drive_controller / detect 객체에서 값을 "읽기만" 한다.
      사격 판정, 경로 계획, 객체 인식에 어떤 영향도 주지 않는다.
    · 이 파일에서 예외가 나도 서버가 죽지 않는다. 전부 감싼다.
    · Blueprint 이름을 'viz3d' 으로 고유하게 두어
      다른 사람의 /view3d 와 엔드포인트가 충돌하지 않는다.

붙이는 방법 (ally-controller.py 맨 아래)

    try:
        from viz3d import attach_viz_taek
        attach_viz_taek(app, fm=fm, drive=drive_controller, detect=tskijun)
    except ImportError:
        pass
"""

import base64
import json
import math
import os
import threading
import time
import urllib.request
import zlib

from flask import Blueprint, Response, jsonify

# ── 화면 주소 ────────────────────────────────────────────
#  2026-09-08  주소를 바꾸려면 이 한 줄만 고치면 된다.
#    예)  "/view3d"   ->  http://localhost:5000/view3d
#         "/fcs"      ->  http://localhost:5000/fcs
#  화면 JS 는 자기가 열린 주소(location.pathname)에서 접두어를 읽으므로
#  여기만 바꾸면 /state · /static-data · /reset-fire 호출도 같이 따라간다.
#  Blueprint 이름("viz3d")은 내부 식별자다. 주소와 무관하니 그대로 둔다.
#
#  2026-09-08  "/view3d/taek" -> "/view3d" 로 줄였다.
VIEW_PREFIX = "/view3d"

bp = Blueprint("viz3d", __name__, url_prefix=VIEW_PREFIX)

_REF = {"fm": None, "drive": None, "detect": None}
_CACHE = {}
_LOCK = threading.Lock()
_HERE = os.path.dirname(os.path.abspath(__file__))

MAP_SPAN = 300.0            # 맵 한 변 [m]
TRAIL_MAX = 500             # 이동 궤적 보관 점 수
SHOT_MAX = 40               # 화면에 남기는 사격 이력


def attach_viz_taek(app, fm=None, drive=None, detect=None):
    """Flask app 에 뷰를 붙인다. 이미 붙어 있으면 참조만 갱신한다."""
    _REF["fm"] = fm
    _REF["drive"] = drive
    _REF["detect"] = detect
    if "viz3d" not in app.blueprints:
        app.register_blueprint(bp)
        print("[viz3d] 전술 3D 상황도 준비 완료"
              "  ->  http://localhost:5000%s" % VIEW_PREFIX)
        # 2026-09-09  적 사격을 5100 번에서 가져오는 스레드
        threading.Thread(target=_foe_poll, daemon=True).start()
        print("[viz3d] 적 사격 수집 시작  <-  %s" % FOE_URL)
    return bp


# ============================================================
# 공통 보조
# ============================================================

def _g(obj, name, default=None):
    try:
        v = getattr(obj, name, default)
        return default if v is None else v
    except Exception:
        return default


def _num(v, nd=2):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, nd)
    except Exception:
        return None


def _xz(p):
    try:
        if p is None:
            return None
        if len(p) >= 3:
            return [_num(p[0]), _num(p[2])]
        if len(p) == 2:
            return [_num(p[0]), _num(p[1])]
    except Exception:
        pass
    return None


def _pack(a, dtype):
    """넘파이 배열을 zlib + base64 로 압축. 300x300 도 가볍게 보낸다."""
    return base64.b64encode(
        zlib.compress(a.astype(dtype).tobytes(), 9)).decode()


# ============================================================
# 이동 궤적
# ============================================================

class _Trail:
    def __init__(self):
        self.my = []
        self.enemy = []
        self._last_t = None

    def push(self, t, my, enemy):
        try:
            if t is None or (self._last_t is not None and t < self._last_t):
                self.my.clear()
                self.enemy.clear()          # 에피소드 재시작 감지
            self._last_t = t
            for buf, p in ((self.my, my), (self.enemy, enemy)):
                if not p:
                    continue
                q = [round(float(p[0]), 1), round(float(p[2]), 1)]
                if buf and abs(buf[-1][0] - q[0]) < 0.4 \
                        and abs(buf[-1][1] - q[1]) < 0.4:
                    continue
                buf.append(q)
                del buf[:-TRAIL_MAX]
        except Exception:
            pass


_trail = _Trail()


# ============================================================
# 정적 데이터 — 지형 / 맵 장애물
# ============================================================

def _load_terrain():
    """
    risk_layers.npz 를 300x300 그대로 압축해 보낸다.
    height 는 uint16 정규화, slope/exposure/blocked 는 uint8.
    """
    if "terrain" in _CACHE:
        return _CACHE["terrain"]

    out = {"ok": False, "n": 0, "span": MAP_SPAN,
           "lo": 0.0, "hi": 1.0, "water": 0.0,
           "hm": "", "sl": "", "ex": "", "bl": "", "ft": ""}
    try:
        import numpy as np
        d = np.load(os.path.join(_HERE, "move", "risk_layers.npz"))
        h = np.nan_to_num(d["height"].astype("float32"))
        n = int(h.shape[0])
        lo, hi = float(h.min()), float(h.max())
        rng = max(1e-6, hi - lo)

        out.update({
            "ok": True, "n": n, "lo": lo, "hi": hi,
            "water": lo + rng * 0.30,          # 수면 기준선
            "hm": _pack((h - lo) / rng * 65535.0, "<u2"),
        })
        if "slope_cost" in d.files:
            out["sl"] = _pack(np.clip(d["slope_cost"], 0, 1) * 255, "u1")
        if "exposure" in d.files:
            e = d["exposure"].astype("float32")
            e = (e - e.min()) / max(1e-6, e.max() - e.min())
            out["ex"] = _pack(e * 255, "u1")
        if "blocked" in d.files:
            out["bl"] = _pack(d["blocked"], "u1")

        # ── 2026-08-26  사격 진지 적합도 (ft) ──────────────────────
        #
        #   "그 자리에 전차를 세우면 얼마나 기우는가" 를 도 단위로 담는다.
        #   점의 경사가 아니라 차체가 놓이는 5 m 반경 안의 최대 경사다.
        #   전차는 6.3 x 3.3 m 면이라 면 전체가 평평해야 안 기운다.
        #
        #   깃발을 마우스로 찍기 때문에 눈으로 보이는 안내가 필요하다.
        #   ally-controller 의 자동 평지 보정과 같은 기준을 쓴다
        #   (flat_snap.py — FOOT_R 5 m).
        try:
            span = MAP_SPAN / (n - 1)
            gz, gx = np.gradient(h, span, span)
            slope = np.degrees(np.arctan(np.hypot(gx, gz)))
            r = max(1, int(round(5.0 / span)))
            foot = slope.copy()
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dz * dz > r * r:
                        continue
                    foot = np.maximum(foot,
                                      np.roll(np.roll(slope, dz, 0), dx, 1))
            out["ft"] = _pack(np.clip(foot, 0, 255), "u1")
        except Exception as e:
            print("[viz3d] 진지 적합도 계산 실패:", e)

        print("[viz3d] 지형 %dx%d  %.1f~%.1f m  적재" % (n, n, lo, hi))
    except Exception as e:
        print("[viz3d] 지형 적재 실패:", e)
    _CACHE["terrain"] = out
    return out


def _load_map_obstacles():
    """DemoMap.map(JSON) 원본 장애물. '정답지'이므로 레이어로 분리해 둔다."""
    if "obstacles" in _CACHE:
        return _CACHE["obstacles"]
    out = []
    try:
        for fn in ("DemoMap.map", "test_Mine_pathfind_1 (1).map"):
            path = os.path.join(_HERE, fn)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for o in data.get("obstacles", []):
                p = o.get("position", {})
                name = str(o.get("prefabName", "?"))
                kind = "".join(c for c in name if c.isalpha()) or "?"
                out.append({"kind": kind,
                            "x": _num(p.get("x"), 1),
                            "y": _num(p.get("y"), 1),
                            "z": _num(p.get("z"), 1)})
            break
    except Exception:
        out = []
    _CACHE["obstacles"] = out
    return out


# ============================================================
# 실시간 상태
# ============================================================

# 전차 차체 크기 (dstar_lite_planner_cost.py 의 get_bb_corners 인자와 동일)
TANK_W, TANK_L = 3.303, 6.339


def _static_index():
    """맵 원본의 정지 오브젝트 좌표 목록. 적 전차 판별에 쓴다."""
    if "static_idx" in _CACHE:
        return _CACHE["static_idx"]
    idx = [(o["x"], o["z"], o["kind"]) for o in _load_map_obstacles()
           if o["x"] is not None and o["z"] is not None]
    _CACHE["static_idx"] = idx
    return idx


def _planner_obstacles(drive):
    """
    /update_obstacle 로 들어와 플래너가 '실제로 아는' 장애물.

    적 전차 판별
        dstar_lite_planner_cost.py 의 enemy_tank_list 등이 전부 비어 있어
        모든 장애물이 'nature' 로 분류된다. 그래서 여기서 따로 추정한다.

        조건 ① 크기가 전차 차체(3.30 x 6.34 m)의 회전 AABB 범위 안
              ② 맵 원본(DemoMap.map)의 정지 오브젝트와 겹치지 않음
        둘을 함께 만족하면 '움직이는 전차 크기 물체' = 적 전차로 본다.
    """
    try:
        planner = _g(drive, "planner") or _g(drive, "path_planner")
        rects = _g(planner, "obstacle_rectangles", []) or []
        static = _static_index()

        # 회전에 따른 AABB 한계
        dmin = TANK_W * 0.85
        dmax = (TANK_W + TANK_L) / math.sqrt(2) * 1.12
        amin, amax = TANK_W * TANK_L * 0.80, (dmax ** 2) * 1.05

        out = []
        for r in rects[:500]:
            try:
                x0, x1 = float(r.x_min), float(r.x_max)
                z0, z1 = float(r.z_min), float(r.z_max)
                w, dep = x1 - x0, z1 - z0
                cx, cz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
                t = _g(r, "type", "nature")

                # ── 적 전차 추정 ──
                guess = None
                if (dmin <= w <= dmax and dmin <= dep <= dmax
                        and amin <= w * dep <= amax):
                    near = min((math.hypot(sx - cx, sz - cz)
                                for sx, sz, _k in static), default=999.0)
                    if near > 4.0:            # 맵 원본에 없는 물체
                        guess = "enemy_tank"

                out.append({
                    "x0": round(x0, 1), "x1": round(x1, 1),
                    "z0": round(z0, 1), "z1": round(z1, 1),
                    "cx": round(cx, 1), "cz": round(cz, 1),
                    "t": guess or t,
                    "g": bool(guess),          # 추정 여부
                })
            except Exception:
                continue
        return out
    except Exception:
        return []


def _shots(fm):
    """사격 이력 — 발사점 · 조준점 · 탄착점."""
    try:
        recs = list(_g(_g(fm, "log"), "records", []) or [])[:SHOT_MAX]
        out = []
        for r in recs:
            fp, ap, ip = r.get("fire_pos"), r.get("aim_point"), r.get("impact")

            # ── 탄착오차를 사선 기준 '앞뒤(range)' 와 '좌우(cross)' 로 분해 ──
            #   앞뒤가 크면  탄도해·고저차·앙각 문제
            #   좌우가 크면  방위·차체 롤·포탑 yaw 문제
            rng_e = crs_e = None
            try:
                if fp and ap and ip:
                    dx, dz = ap[0] - fp[0], ap[2] - fp[2]
                    L = math.hypot(dx, dz)
                    if L > 1e-6:
                        ux, uz = dx / L, dz / L          # 사선 단위벡터
                        ex, ez = ip[0] - ap[0], ip[2] - ap[2]
                        rng_e = ex * ux + ez * uz         # + 길게 / - 짧게
                        crs_e = ex * uz - ez * ux         # + 오른쪽 / - 왼쪽
            except Exception:
                pass

            out.append({
                "rng_err": _num(rng_e, 2), "crs_err": _num(crs_e, 2),
                "id": r.get("id"),
                "t": _num(r.get("t"), 1),
                "fire": _xz(fp), "aim": _xz(ap), "imp": _xz(ip),
                "fy": _num(fp[1], 1) if fp else None,
                "iy": _num(ip[1], 1) if ip else None,
                "dist": _num(r.get("dist"), 1),
                "tof": _num(r.get("tof"), 2),
                "p_hit": _num(r.get("p_hit"), 3),
                "kind": r.get("kind"),
                "zone": r.get("zone"),
                "miss": _num(r.get("miss"), 2),
                "hit": r.get("kind") == "tank",
            })
        return out
    except Exception:
        return []


def _collect_fire():
    fm = _REF["fm"]
    if fm is None:
        return {"available": False, "reason": "fm 참조 없음"}

    out = {"available": True}
    try:
        out.update(fm.status() or {})
    except Exception as e:
        out["status_error"] = str(e)

    tm, fc, trk, log = _g(fm, "tm"), _g(fm, "fc"), _g(fm, "trk"), _g(fm, "log")

    if tm is not None:
        _trail.push(_g(tm, "t"), _g(tm, "my"), _g(tm, "enemy"))
        out["tm"] = {
            "t": _num(_g(tm, "t"), 2),
            "my": _g(tm, "my"), "enemy": _g(tm, "enemy"),
            "body_x": _num(_g(tm, "body_x", 0.0)),
            "turret_x": _num(_g(tm, "turret_x", 0.0)),
            "turret_y": _num(_g(tm, "turret_y", 0.0)),
            "body_y": _num(_g(tm, "body_y", 0.0)),
            "body_z": _num(_g(tm, "body_z", 0.0)),
            "my_speed": _num(_g(tm, "my_speed", 0.0)),
            "enemy_speed": _num(_g(tm, "enemy_speed", 0.0)),
            "enemy_body_x": _num(_g(tm, "enemy_body_x", 0.0)),
            "my_hp": _g(tm, "my_hp"), "enemy_hp": _g(tm, "enemy_hp"),
        }

    if fc is not None:
        sol = _g(fc, "last_solution") or _g(fm, "_last_sol")
        # 2026-09-08  해가 없을 때 사유를 그대로 내보낸다.
        #   "탄도해없음" 까지만 알면 왜 없는지(사거리 초과/너무 가까움/예측 발산)
        #   를 바깥에서 추측해야 했다. Solution.reason 에 이미 적혀 있다.
        if sol is not None and not _g(sol, "valid", False):
            out["sol_reason"] = _g(sol, "reason", "") or "사유 없음"
        if sol is not None and _g(sol, "valid", False):
            out["aim3"] = _g(sol, "aim_point")
            out["sol"] = {
                "elev": _num(_g(sol, "elevation"), 3),
                "bearing": _num(_g(sol, "bearing"), 3),
                "flight": _num(_g(sol, "flight"), 3),
                "drdt": _num(_g(sol, "drdt"), 2),
                "half_lat": _num(_g(sol, "half_lat"), 2),
                "sig_lat": _num(_g(sol, "sig_lat"), 3),
            }
        # 조준 오차와 허용 데드밴드 — SLEW 원인 진단의 핵심
        out["aim"] = {
            "yaw_err":   _num(_g(fc, "yaw_err", 0.0), 3),
            "pitch_err": _num(_g(fc, "pitch_err", 0.0), 3),
            "yaw_db":    _num(_g(fc, "yaw_db", 0.0), 3),
            "pitch_db":  _num(_g(fc, "pitch_db", 0.0), 3),
            "track_duty":_num(_g(fc, "track_duty", 0.0), 3),
            "duty_max":  _num(_g(fc, "track_duty_max", 0.0), 3),
        }
        # ── 시각 어긋남 감지 ──────────────────────────────
        # 에피소드를 Restart 하면 sim_time 이 0 으로 되감기는데
        # FireControl.last_fire_t 는 이전 판의 값을 그대로 들고 있다.
        # 그러면 reload_left 가 거대한 양수가 되어 영원히 사격하지 못한다.
        _lf, _now = _g(fc, "last_fire_t"), _g(tm, "t") if tm is not None else None
        if _lf is not None and _now is not None and _lf > _now + 0.5:
            out["desync"] = {"last_fire": _num(_lf, 1), "now": _num(_now, 1),
                             "gap": _num(_lf - _now, 1)}

        out["gates"] = {
            "p_hit_min": _num(_g(fc, "p_hit_min"), 3),
            "p_hit_floor": _num(_g(fc, "p_hit_floor"), 3),
            "body_rate_max": _num(_g(fc, "body_rate_max"), 2),
            "tof_max": _num(_g(fc, "tof_max"), 3),
            "pitch_db_max": _num(_g(fc, "pitch_db_max"), 3),
            "lon_gain": _num(_g(fc, "lon_gain"), 3),
            "db_safety": _num(_g(fc, "db_safety"), 3),
            "reload_s": _num(_g(fc, "reload_s"), 3),
            "state": _g(fc, "state"),
            # 2026-09-09  최소 교전거리. ally-controller 가 fm 에 달아 둔다.
            #   fire/ 안의 게이트가 아니라 상위(inhibit_fire)에서 막는 값이라
            #   fc 가 아니라 fm 에서 읽는다.
            "min_engage": _num(_g(fm, "min_engage"), 1),
        }

    if trk is not None:
        # 2026-08-25  속성 이름을 실제 TargetTracker 에 맞춰 바로잡았다.
        #   duty 는 트래커가 아니라 FireControl.track_duty 에 있다 (out["aim"]).
        #   n 은 존재하지 않는다. n_scored 가 맞다.
        #   k_err 이 기동사격의 핵심 지표다: 예측오차 ≈ k_err x 비과시간^2
        tof_now = _num(_g(_g(fm, "_last_sol"), "flight", 0.0), 3) or 0.0
        kerr = _g(trk, "k_err", 0.0)
        out["track"] = {
            "speed":   _num(_g(trk, "speed"), 2),
            "heading": _num(_g(trk, "heading"), 1),
            "omega":   _num(math.degrees(_g(trk, "omega", 0.0)), 2),  # 선회 각속도 [deg/s]
            "a_long":  _num(_g(trk, "a_long"), 2),                    # 접선 가속 [m/s^2]
            "k_err":   _num(kerr, 3),
            "n_scored": _g(trk, "n_scored", 0),
            # 지금 비과시간에서 기대되는 예측오차 [m]
            "pred_err": _num(kerr * tof_now * tof_now, 2),
        }
        try:
            lat, lon = trk.sigma(tof_now)
            out["track"]["sig_lat"] = _num(lat, 2)
            out["track"]["sig_lon"] = _num(lon, 2)
        except Exception:
            pass

    if log is not None:
        out["unmatched"] = _g(log, "unmatched", 0)
        out["recorded"] = len(_g(log, "records", []) or [])
        out["lost"] = list(_g(log, "lost", []) or [])

    try:
        out["suggest"] = fm.suggest_range()
    except Exception:
        out["suggest"] = None

    # 2026-08-25  말이 안 되는 표적 좌표를 버린 횟수.
    #   0 이 아니면 적이 월드 밖으로 떨어졌거나 좌표가 깨진 것이다.
    out["bad_target"] = _g(fm, "bad_target", 0)

    out["shots"] = _shots(fm)
    out["trail"] = {"my": _trail.my[-TRAIL_MAX:],
                    "enemy": _trail.enemy[-TRAIL_MAX:]}
    # 2026-09-09  적이 쏜 것도 같이 내보낸다. _FOE 스레드가 채운다.
    out["foe"] = dict(_FOE)
    return out


# ============================================================
# 적 사격  (2026-09-09)
# ============================================================
#
#  왜 여기서 가져오나
#      적 전차는 5100 번 서버가 조종하고, 적이 쏜 기록도 거기에만
#      있다. 이 상황도는 아군 FireModule 만 읽으므로 적 사격이
#      화면에 한 번도 안 나왔다 (9/09 46번 판: 적이 8발을 쐈는데
#      3D 화면에는 아무것도 없었다).
#
#  왜 스레드인가
#      /state 요청 안에서 HTTP 를 부르면 5100 이 느릴 때 이 화면이
#      통째로 멈춘다. 백그라운드로 돌리고 마지막 값만 읽는다.
FOE_URL = "http://127.0.0.1:5100/state"
FOE_POLL_S = 0.5
_FOE = {"ok": False, "shots": [], "fires": 0, "hits": 0,
        "aim": "", "enemy_fire": None, "err": "적 서버 응답 없음"}


def _foe_poll():
    while True:
        try:
            with urllib.request.urlopen(FOE_URL, timeout=2.0) as r:
                j = json.loads(r.read().decode("utf-8"))
            _FOE.update({
                "ok": True,
                "shots": j.get("shots") or [],
                "fires": j.get("fires") or 0,
                "hits": j.get("hits") or 0,
                "aim": j.get("aim") or "",
                "enemy_fire": j.get("enemy_fire"),
                "reload_left": j.get("reload_left"),
                "err": "",
            })
        except Exception as e:
            _FOE["ok"] = False
            _FOE["err"] = f"{type(e).__name__}: {e}"
        time.sleep(FOE_POLL_S)


# 주행 이력 추적 — 목적지가 몇 번 바뀌었고 경로를 몇 번 다시 짰는가
_DRIVE = {"dest_n": 0, "replan_n": 0, "last_dest": None, "last_sig": None,
          "dests": []}


def _collect_drive():
    dv = _REF["drive"]
    if dv is None:
        return {"available": False, "reason": "drive 참조 없음"}
    path = _g(dv, "current_path", []) or []
    try:
        pts = [p for p in (_xz(q) for q in path) if p]
    except Exception:
        pts = []

    pos = _xz(_g(dv, "current_pos"))
    dest = _xz(_g(dv, "dest"))

    # ── 목적지 변경 감지 ──
    if dest is not None:
        d0 = _DRIVE["last_dest"]
        if d0 is None or math.hypot(dest[0] - d0[0], dest[1] - d0[1]) > 1.0:
            _DRIVE["dest_n"] += 1
            _DRIVE["last_dest"] = dest
            _DRIVE["dests"].append(dest)
            del _DRIVE["dests"][:-12]

    # ── 재계획 감지 ──
    sig = (len(pts), round(pts[-1][0], 1) if pts else 0,
           round(pts[-1][1], 1) if pts else 0)
    if sig != _DRIVE["last_sig"]:
        _DRIVE["replan_n"] += 1
        _DRIVE["last_sig"] = sig

    # ── 경로 길이 · 우회율 ──
    plen = 0.0
    for i in range(len(pts) - 1):
        plen += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    straight = (math.hypot(dest[0] - pos[0], dest[1] - pos[1])
                if (pos and dest) else None)

    return {
        "available": True,
        "pos": pos,
        "dest": dest,
        "path": pts,
        "path_len": _num(plen, 1),
        "straight": _num(straight, 1),
        "detour": _num(plen / straight, 2) if (straight and straight > 1) else None,
        "speed_kmh": _num(_g(dv, "info_speed_kmh"), 2),
        "arrived": bool(_g(dv, "arrival_latched", False)),
        "stop_flag": bool(_g(dv, "stop_flag", False)),
        "dest_n": _DRIVE["dest_n"],
        "replan_n": _DRIVE["replan_n"],
        "dests": _DRIVE["dests"][-6:],
        "known": _planner_obstacles(dv),
        # 2026-08-26  목적지가 '전차가 기울지 않는 자리' 인가.
        #   터미널 메시지가 /info 로그에 묻혀 안 보인다는 지적이 있어
        #   화면에도 같은 정보를 띄운다.
        "dest_tilt": _dest_tilt(dest),
    }


def _dest_tilt(dest):
    """목적지에 전차를 세우면 얼마나 기우는가 [deg]. 모르면 None."""
    if not dest:
        return None
    try:
        import flat_snap
        flat_snap.init(_HERE)
        return flat_snap.tilt_at(dest[0], dest[1])
    except Exception:
        return None


def _collect_detect():
    dt = _REF["detect"]
    if dt is None:
        return {"available": False, "objects": [], "reason": "detect 참조 없음"}
    try:
        raw = getattr(dt, "DETECTED_OBJECTS_INFO", []) or []
        objs = []
        for o in raw:
            if isinstance(o, dict):
                objs.append({k: (_num(v, 2) if isinstance(v, (int, float)) else v)
                             for k, v in o.items()})
            else:
                objs.append({"raw": str(o)})
        return {"available": True, "objects": objs[:60], "count": len(raw)}
    except Exception as e:
        return {"available": False, "objects": [], "reason": str(e)}


# ============================================================
# 라우트
# ============================================================

@bp.route("/state")
def state():
    try:
        return jsonify({"ok": True,
                        "fire": _collect_fire(),
                        "drive": _collect_drive(),
                        "detect": _collect_detect()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/gap")
def gap():
    """
    두 전차의 현재 좌표만 돌려준다.  gap.py 가 쓴다.

    2026-09-09  42번 판에서 gap.py 가 /state 를 0.25 초 주기로 찔렀는데
    800 초 동안 표본이 348 개밖에 안 남았다.  평균 2.3 초에 한 개다.
    /state 는 한 번 부를 때마다 궤적 500점 x 2, 사격 이력 40발,
    주행 계획까지 전부 새로 만든다.  그 시간이 폴링 주기보다 길었다.
    표본 간격이 2.3 초면 5 m/s 로 접근하는 적이 11 m 를 건너뛴다.
    최소 접근 거리를 재는 자로는 쓸 수 없다.

    여기서는 tm 의 좌표 두 개만 읽는다.  응답이 200 바이트 안쪽이라
    0.25 초 주기를 그대로 지킬 수 있다.
    """
    try:
        fm = _REF["fm"]
        tm = _g(fm, "tm")
        if tm is None:
            return jsonify({"ok": False, "error": "tm 없음"})
        return jsonify({"ok": True,
                        "t": _num(_g(tm, "t"), 2),
                        "my": _g(tm, "my"),
                        "enemy": _g(tm, "enemy"),
                        # 2026-09-09  적 서버(_poll_ally)가 이 값을 쓴다.
                        #   /state 를 대신 부르던 것을 여기로 옮겼다.
                        "enemy_body_x": _num(_g(tm, "enemy_body_x", 0.0)),
                        "my_speed": _num(_g(tm, "my_speed", 0.0))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/static-data")
def static_data():
    try:
        with _LOCK:
            return jsonify({"ok": True,
                            "obstacles": _load_map_obstacles(),
                            "terrain": _load_terrain()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/reset-fire", methods=["GET", "POST"])
def reset_fire():
    """
    에피소드 재시작으로 어긋난 사격 모듈의 '시각' 상태만 되돌린다.

    왜 필요한가
        시뮬레이터를 Restart 하면 sim_time 이 0 으로 되감긴다.
        그런데 FireControl.last_fire_t 는 이전 판의 큰 값을 그대로 들고 있어
        reload_left = reload_s - (now - last_fire_t) 가 거대한 양수가 된다.
        결과적으로 영원히 RELOAD 상태에 머물며 한 발도 쏘지 못한다.

    무엇을 건드리는가
        last_fire_t, _t0, state 세 개뿐이다.
        조준 파라미터 · 보정값 · 사격 기록은 그대로 둔다.

    근본 해결
        ally-controller.py 의 /init 에서 사격 모듈을 초기화해야 한다.
        (FireModule 에 reset() 이 없으므로 fire_module.py 에 추가 필요)
    """
    try:
        fm = _REF["fm"]
        fc = _g(fm, "fc")
        if fc is None:
            return jsonify({"ok": False, "error": "FireControl 참조 없음"})
        before = _g(fc, "last_fire_t")
        fc.last_fire_t = None
        try:
            fc._t0 = None
        except Exception:
            pass
        try:
            fc.state = "IDLE"
        except Exception:
            pass
        print("[viz3d] 사격 시각 상태 초기화  last_fire_t %s -> None" % before)
        return jsonify({"ok": True, "before": _num(before, 1)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/shots.csv")
def shots_csv():
    try:
        fm = _REF["fm"]
        text = fm.export_csv() if fm else ""
        return Response("﻿" + text, mimetype="text/csv",
                        headers={"Content-Disposition":
                                 "attachment; filename=shots.csv"})
    except Exception as e:
        return Response("error,%s" % e, mimetype="text/csv")


@bp.route("/", strict_slashes=False)
def page():
    return Response(_HTML, mimetype="text/html; charset=utf-8")


# ============================================================
# 화면
# ============================================================

_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>전술 3D 상황도</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<style>
:root{
  --bg:#05070b; --glass:rgba(11,16,24,.82); --line:rgba(120,160,210,.16);
  --fg:#e6edf7; --dim:#7d8ba1; --acc:#38bdf8;
  --friend:#4b9cf5; --hostile:#ef4444; --route:#f5a524;
  --ok:#4ade80; --no:#f87171; --warn:#fbbf24;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--fg);
  font:12.5px/1.5 "Malgun Gothic",-apple-system,system-ui,sans-serif;
  user-select:none;-webkit-font-smoothing:antialiased}
#gl{position:fixed;inset:0}
.pane{position:fixed;background:var(--glass);border:1px solid var(--line);
  border-radius:10px;backdrop-filter:blur(14px) saturate(1.3);
  box-shadow:0 10px 34px rgba(0,0,0,.55);z-index:10}
.pane h3{margin:0;padding:9px 13px;font-size:9.5px;letter-spacing:1.7px;
  color:var(--dim);text-transform:uppercase;font-weight:700;
  border-bottom:1px solid var(--line);display:flex;
  justify-content:space-between;align-items:center}
.pane h3 em{font-style:normal;font:10px Consolas,monospace;color:var(--acc)}

/* ── 상단 판독 띠 ── */
#top{top:0;left:0;right:0;height:40px;border-radius:0;border-width:0 0 1px;
  display:flex;align-items:center;padding:0 14px;gap:0}
#top .b{font-weight:800;letter-spacing:2px;font-size:11px;color:var(--acc);
  padding-right:15px;margin-right:4px;border-right:1px solid var(--line)}
#top .r{padding:0 13px;border-right:1px solid var(--line);min-width:88px;
  display:flex;flex-direction:column;line-height:1.25}
#top .r b{font:700 12.5px Consolas,monospace}
#top .r i{font-style:normal;font-size:9px;color:var(--dim);letter-spacing:.9px}
#top .sp{margin-left:auto;display:flex;gap:5px;border:0;padding:0}
.btn{background:rgba(20,28,40,.9);border:1px solid var(--line);color:var(--dim);
  padding:5px 13px;border-radius:5px;cursor:pointer;font:600 11px inherit;
  transition:.12s}
.btn:hover{color:var(--fg);border-color:rgba(120,160,210,.4)}
.btn.on{background:var(--acc);border-color:var(--acc);color:#04121c}

/* ── 좌 패널 ── */
#left{top:52px;left:12px;width:210px;max-height:calc(100% - 66px);overflow:auto}
.body{padding:9px 13px 12px}
.ck{display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer;font-size:11.5px}
.ck input{accent-color:var(--acc);width:12px;height:12px;cursor:pointer;margin:0}
.sl{padding:6px 0 2px}
.sl label{display:flex;justify-content:space-between;font-size:11px;color:var(--dim)}
.sl label b{color:var(--acc);font:700 11px Consolas,monospace}
.sl input{width:100%;accent-color:var(--acc);height:3px;margin:5px 0 0}
select{width:100%;background:rgba(20,28,40,.95);color:var(--fg);
  border:1px solid var(--line);border-radius:5px;padding:5px 7px;font:inherit;font-size:11.5px}
.mk{display:flex;align-items:center;gap:9px;padding:6px 13px;cursor:pointer;
  border-bottom:1px solid rgba(255,255,255,.035)}
.mk:hover{background:rgba(56,189,248,.09)}
.mk b{display:block;font-size:11.5px}
.mk span{display:block;font:10px Consolas,monospace;color:var(--dim)}

/* ── 우 패널 ── */
/* 2026-08-26  우측 패널이 화면 아래로 넘쳐 마지막 줄들이 잘렸다.
   (1σ 횡/종 · 차체 피치/롤 · 포탑/앙각 · 인지 장애물 등)
   높이를 화면 안으로 제한하고 넘치면 스크롤되게 한다. */
#right{top:52px;right:12px;width:238px;
  max-height:calc(100vh - 64px);overflow-y:auto;overscroll-behavior:contain}
#right::-webkit-scrollbar{width:7px}
#right::-webkit-scrollbar-thumb{background:rgba(125,139,161,.4);border-radius:4px}
#right::-webkit-scrollbar-track{background:transparent}
.hp{padding:8px 13px 4px}
.hp .t{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px}
.hp .t b{font:700 12px Consolas,monospace}
.bar{height:5px;border-radius:3px;background:rgba(255,255,255,.09);overflow:hidden}
.bar i{display:block;height:100%;border-radius:3px;transition:width .3s}
.row{display:flex;justify-content:space-between;padding:3px 13px;font-size:11.5px;
  gap:8px}
/* 2026-08-26  라벨이 '비과시' / '간' 으로 쪼개지던 것을 막는다.
   flex 자식은 기본이 축소 허용이라 좁아지면 글자 단위로 줄이 바뀐다. */
.row>span{white-space:nowrap;flex:0 0 auto}
.row>b{text-align:right;min-width:0}
.row span{color:var(--dim)}
.row b{font:600 11.5px Consolas,monospace}
.sep{height:1px;background:var(--line);margin:6px 0}
.pill{padding:1px 7px;border-radius:8px;font:700 10px inherit}
.pass{background:rgba(74,222,128,.17);color:var(--ok)}
.fail{background:rgba(248,113,113,.17);color:var(--no)}
.off{background:rgba(125,139,161,.17);color:var(--dim)}

/* ── 축소 지도 ──
   2026-08-26  접기만으로는 부족했다. 접어도 자리가 우측 사격 패널
   위라서 맨 아랫줄(차체 피치/롤 · 포탑/앙각 · 인지 장애물)을 계속
   가렸다. 화면 아래 가운데로 옮겨 어느 패널과도 겹치지 않게 한다.
     왼쪽 범례      12 ~ 312 px
     오른쪽 사격판  화면 우측 끝
   그 사이 324 px 부터가 빈 자리다. */
#mini{left:324px;bottom:12px;width:210px;padding:0;overflow:hidden}
#mmh{display:flex;align-items:center;justify-content:space-between;
  padding:5px 9px;font-size:10.5px;color:var(--dim);cursor:pointer;
  user-select:none;letter-spacing:.04em}
#mmh:hover{color:var(--fg)}
#mmh b{font-weight:600}
#mmc{width:210px;height:210px;display:block;border-radius:0 0 10px 10px}
#mini.fold #mmc{display:none}
#mini.fold{width:96px}

/* ── 나침반 · 범례 ── */
#comp{position:fixed;top:56px;left:236px;width:58px;height:58px;z-index:11;
  pointer-events:none}
#leg{left:12px;bottom:12px;padding:8px 13px;font-size:10.5px;color:var(--dim);
  display:flex;flex-direction:column;gap:5px;max-width:300px}
#leg .r1{display:flex;gap:12px;flex-wrap:wrap}
#leg s{text-decoration:none;display:flex;align-items:center;gap:4px}
#leg u{text-decoration:none;width:9px;height:9px;border-radius:2px;display:inline-block}
#toast{position:fixed;left:50%;top:52px;transform:translateX(-50%);z-index:30;
  background:var(--glass);border:1px solid var(--line);border-radius:8px;
  padding:7px 16px;font-size:11.5px;opacity:0;transition:.3s;pointer-events:none}
#toast.on{opacity:1}

/* ── 피격 효과 ── */
@keyframes shake{
  0%,100%{transform:translate(0,0)}   15%{transform:translate(-5px,3px)}
  30%{transform:translate(5px,-3px)}  45%{transform:translate(-4px,-2px)}
  60%{transform:translate(4px,2px)}   80%{transform:translate(-2px,1px)}}
body.shk #gl{animation:shake .42s ease-out}
#vig{position:fixed;inset:0;z-index:25;pointer-events:none;opacity:0;
  transition:opacity .12s;
  box-shadow:inset 0 0 190px 55px rgba(255,40,30,.62)}
#vig.on{opacity:1;transition:opacity 0s}
#vig.hitok{box-shadow:inset 0 0 190px 55px rgba(80,255,150,.5)}
.hp .bar i.flash{animation:hpf .45s ease-out 3}
@keyframes hpf{0%,100%{filter:none}50%{filter:brightness(3)}}
::-webkit-scrollbar{width:7px;height:7px}
::-webkit-scrollbar-thumb{background:#22304a;border-radius:4px}
::-webkit-scrollbar-track{background:transparent}
</style></head><body>

<div id="gl"></div>

<div class="pane" id="top">
  <span class="b">전술 3D 상황도</span>
  <div class="r" style="min-width:172px"><b id="t-xy">–</b><i>격자 좌표</i></div>
  <div class="r"><b id="t-el" style="color:#ffd23f">–</b><i>표고</i></div>
  <div class="r"><b id="t-rg">–</b><i>교전거리</i></div>
  <div class="r"><b id="t-br">–</b><i>방위</i></div>
  <div class="r"><b id="t-sp">–</b><i>아군속도</i></div>
  <div class="r"><b id="t-st">–</b><i>사격통제</i></div>
  <div class="r sp">
    <button class="btn" id="b-follow">아군 추적</button>
    <button class="btn" id="b-top">부감</button>
    <button class="btn" id="b-reset">시점 초기화</button>
  </div>
</div>

<div class="pane" id="left">
  <h3>표시 설정</h3>
  <div class="body">
    <div class="sl"><label>표면 <b id="v-surf"></b></label>
      <select id="surf">
        <option value="terrain">위성 지도</option>
        <option value="slope">경사도</option>
        <option value="expo">피탐 노출도</option>
        <option value="block">통행 불가</option>
        <option value="flat">사격 진지 적합</option>
      </select></div>
    <div class="sl"><label>수직 과장 <b id="v-vs">2.2×</b></label>
      <input type="range" id="vs" min="1" max="5" step="0.1" value="2.2"></div>
    <div class="sl"><label>수면 높이 <b id="v-wl">–</b></label>
      <input type="range" id="wl" min="0" max="100" step="1" value="30"></div>
  </div>
  <h3>레이어</h3>
  <div class="body" id="layers"></div>
  <h3>마커 <em id="mk-n"></em></h3>
  <div id="markers"></div>
</div>

<div class="pane" id="right">
  <h3>전투 현황 <em id="r-t"></em></h3>
  <div class="hp">
    <div class="t"><span>아군 HP</span><b id="h-my">–</b></div>
    <div class="bar"><i id="h-myb" style="width:100%;background:linear-gradient(90deg,#22c55e,#4ade80)"></i></div>
  </div>
  <div class="hp">
    <div class="t"><span>적 HP</span><b id="h-en">–</b></div>
    <div class="bar"><i id="h-enb" style="width:100%;background:linear-gradient(90deg,#dc2626,#f87171)"></i></div>
  </div>
  <div class="sep"></div>
  <div class="row"><span>교전거리</span><b id="r-d">–</b></div>
  <div class="row"><span>권장거리</span><b id="r-sg">–</b></div>
  <div class="row"><span>유효사거리</span><b id="r-ev">–</b></div>
  <div class="row"><span>명중확률</span><b id="r-ph">–</b></div>
  <div class="row"><span>비과시간</span><b id="r-tf">–</b></div>
  <div class="row"><span>재장전</span><b id="r-rl">–</b></div>
  <div class="sep"></div>
  <div class="row"><span>사격 / 명중</span><b id="r-fh">–</b></div>
  <div class="row"><span>명중률</span><b id="r-hr">–</b></div>
  <div class="sep"></div>
  <div id="gates"></div>
  <div class="sep"></div>
  <div id="diag"></div>
  <div class="sep"></div>
  <div id="track"></div>
  <div class="sep"></div>
  <div class="row"><span>차체 피치 / 롤</span><b id="r-hull">–</b></div>
  <div class="row"><span>포탑 / 앙각</span><b id="r-tur">–</b></div>
  <div class="row"><span>인지 장애물</span><b id="r-ko">–</b></div>
  <div class="row"><span>탐지 객체</span><b id="r-do">–</b></div>
  <div class="sep"></div>
  <div id="drive"></div>
</div>

<div class="pane fold" id="mini">
  <div id="mmh"><span>축소 지도</span><b id="mmt">펴기</b></div>
  <canvas id="mmc"></canvas></div>
<canvas id="comp" width="116" height="116"></canvas>

<div class="pane" id="leg">
  <div class="r1">
    <s><u style="background:#4b9cf5"></u>아군</s>
    <s><u style="background:#ef4444"></u>적</s>
    <s><u style="background:#f5a524"></u>계획경로</s>
    <s><u style="background:#4ade80"></u>사선개통</s>
    <s><u style="background:#a78bfa"></u>탄도</s>
    <s><u style="background:#fbbf24"></u>탄착</s>
    <s><u style="background:#f97316"></u>적 탄착</s>
  </div>
  <div style="opacity:.75">좌클릭 회전 · 우클릭 이동 · 휠 확대 · 마커 클릭 시 해당 지점으로</div>
</div>

<div id="vig"></div>
<div id="toast"></div>

<script>
"use strict";
// 2026-09-08  API 접두어를 이 페이지가 열린 주소에서 읽는다.
//   서버의 VIEW_PREFIX 를 바꿔도 여기를 따라 고칠 필요가 없다.
//   (끝 슬래시 유무 둘 다 대응)
const API=location.pathname.replace(/\/+$/,'');
const $=s=>document.querySelector(s);
const SPAN=300;
let ST=null, S=null, TR=null;          // 정적 / 상태 / 지형 디코드
let VS=2.2, WATERP=30, SURF='terrain';
let follow=false;

const LAYERS=[
 ['route','계획 경로',1],['trail','이동 궤적',1],['shots','탄도 · 탄착',1],
 ['foeshots','적 사격',1],
 ['los','사선',1],['rings','사거리 링',1],['aim','조준점',1],['ray','포신 지향선',1],
 ['known','인지 장애물',0],['foes','적 전차 (추정)',1],['mapobs','맵 원본 장애물',1],
 ['det','탐지 객체',1],['label','라벨 · HP',1],['grid','격자',0],['water','수면',1]];
const L={}; LAYERS.forEach(([k,,v])=>L[k]=!!v);
$('#layers').innerHTML=LAYERS.map(([k,n])=>
 `<label class="ck"><input type="checkbox" data-l="${k}" ${L[k]?'checked':''}>${n}</label>`).join('');
$('#layers').onchange=e=>{const k=e.target.dataset.l; if(k){L[k]=e.target.checked; applyLayers();}};

function toast(t){const e=$('#toast');e.textContent=t;e.classList.add('on');
  clearTimeout(e._t);e._t=setTimeout(()=>e.classList.remove('on'),1600);}

/** 에피소드 재시작으로 어긋난 사격 시각 상태를 되돌린다. */
async function fixDesync(){
  try{
    const r=await fetch(API+'/reset-fire',{method:'POST'});
    const j=await r.json();
    toast(j.ok ? `사격 시각 초기화됨 (이전 ${j.before} s)` : '실패: '+j.error);
  }catch(e){ toast('실패: '+e); }
}
window.fixDesync=fixDesync;

// ══════════════════════════════════════════
//  지형 디코드
// ══════════════════════════════════════════
function unpack(b64,Type){
  const bin=atob(b64), u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
  const raw=pako.inflate(u);
  // 정렬 문제를 피하려고 항상 자기 버퍼로 복사한다.
  const copy=new Uint8Array(raw.length); copy.set(raw);
  return new Type(copy.buffer);
}
function decodeTerrain(t){
  if(!t||!t.ok) return null;
  const n=t.n, rng=t.hi-t.lo;
  const hm=unpack(t.hm,Uint16Array);
  const H=new Float32Array(n*n);
  for(let i=0;i<n*n;i++) H[i]=t.lo+hm[i]/65535*rng;
  return {n, lo:t.lo, hi:t.hi, H,
    sl: t.sl?unpack(t.sl,Uint8Array):null,
    ex: t.ex?unpack(t.ex,Uint8Array):null,
    bl: t.bl?unpack(t.bl,Uint8Array):null,
    ft: t.ft?unpack(t.ft,Uint8Array):null};
}
// 월드(x,z) → 격자 인덱스.  3D 메시와 같은 매핑.
function idx(i,j){ return j*TR.n+i; }
function hAt(wx,wz){
  if(!TR) return 0;
  const n=TR.n, d=SPAN/(n-1);
  let fi=wx/d, fj=(SPAN-wz)/d;
  fi=Math.max(0,Math.min(n-1.001,fi)); fj=Math.max(0,Math.min(n-1.001,fj));
  const i=fi|0, j=fj|0, u=fi-i, v=fj-j, H=TR.H;
  return (H[idx(i,j)]*(1-u)+H[idx(i+1,j)]*u)*(1-v)
       + (H[idx(i,j+1)]*(1-u)+H[idx(i+1,j+1)]*u)*v;
}
const W2S=(x,z)=>[x-SPAN/2, SPAN/2-z];
const S2W=(x,z)=>[x+SPAN/2, SPAN/2-z];
// 월드(x,z) → 지형에 붙인 씬 좌표
function V(x,z,dy){const s=W2S(x,z);
  return new THREE.Vector3(s[0], hAt(x,z)*VS+(dy||0), s[1]);}

/**
 * 시뮬레이터 방위각[deg] → three.js 의 Y축 회전[rad]
 *
 *   시뮬레이터: 0° = 월드 +Z(북), 시계방향으로 증가
 *   씬 좌표   : scene_z = 150 - world_z  (Z축이 뒤집혀 있다)
 *   모형      : 정면이 로컬 +Z
 *
 *   방위 θ 의 월드 방향은 (sinθ, cosθ).
 *   씬에서는 (sinθ, -cosθ) 가 되어야 하므로 회전각 φ 는
 *   sinφ = sinθ, cosφ = -cosθ  →  φ = π - θ
 *
 *   -θ 로 두면 정확히 반대를 향한다.
 */
const yawOf = deg => Math.PI - (deg||0)*Math.PI/180;
const dist2=(a,b)=>Math.hypot(b[0]-a[0],b[1]-a[1]);
/** MGRS 풍 격자 좌표 표기. 100 m 방안을 문자로, 나머지를 5자리 숫자로. */
function mgrs(x,z){
  const L='ABCDEFGHJKLMNPQRSTUVWXYZ';
  const gx=L[Math.max(0,Math.min(23,Math.floor(x/100)))]||'A';
  const gz=L[Math.max(0,Math.min(23,Math.floor(z/100)+8))]||'A';
  const e=String(Math.round((x%100)*100)).padStart(5,'0');
  const nn=String(Math.round((z%100)*100)).padStart(5,'0');
  return `52S ${gx}${gz} ${e} ${nn}`;
}
const brg=(a,b)=>(Math.atan2(b[0]-a[0],b[1]-a[1])*180/Math.PI+360)%360;
const waterLevel=()=>TR? TR.lo+(TR.hi-TR.lo)*(WATERP/100) : 0;

// ══════════════════════════════════════════
//  전차 모형
// ══════════════════════════════════════════
function mat(c,r,m){return new THREE.MeshStandardMaterial(
  {color:c,roughness:r===undefined?.78:r,metalness:m===undefined?.22:m});}

function buildTank(P){
  // P: {hull, turret, track, metal, accent, ring, emis}
  const g=new THREE.Group();
  const mHull=mat(P.hull,.72,.25), mTur=mat(P.turret,.7,.28),
        mTrk=mat(P.track,.92,.12), mMet=mat(P.metal,.45,.75),
        mAcc=mat(P.accent,.5,.4);
  // 진영이 멀리서도 구분되도록 은은한 자체발광을 넣는다.
  if(P.emis){ mHull.emissive=new THREE.Color(P.emis); mHull.emissiveIntensity=.55;
              mTur.emissive =new THREE.Color(P.emis); mTur.emissiveIntensity=.55; }
  // 발밑 진영 링
  const fring=new THREE.Mesh(new THREE.RingGeometry(3.7,4.5,40),
    new THREE.MeshBasicMaterial({color:P.ring,transparent:true,opacity:.6,
      side:THREE.DoubleSide,depthWrite:false}));
  fring.rotation.x=-Math.PI/2; fring.position.y=0.10;
  fring.userData={base:P.ring}; g.add(fring);
  const add=(p,mm,x,y,z,rx,ry,rz)=>{const o=new THREE.Mesh(p,mm);
    o.position.set(x,y,z); if(rx)o.rotation.x=rx; if(ry)o.rotation.y=ry;
    if(rz)o.rotation.z=rz; o.castShadow=true; o.receiveShadow=true;
    g.add(o); return o;};

  // ── 차체 ──
  add(new THREE.BoxGeometry(3.0,0.85,6.1), mHull, 0,0.95,0);
  // 경사 전면장갑
  const gl=add(new THREE.BoxGeometry(2.96,0.16,2.35), mHull, 0,1.32,2.34, -0.62);
  // 상부 구조
  add(new THREE.BoxGeometry(2.9,0.5,3.5), mHull, 0,1.60,-0.55);
  // 후면판
  add(new THREE.BoxGeometry(2.9,0.75,0.18), mHull, 0,1.35,-3.02);
  // 펜더
  add(new THREE.BoxGeometry(3.85,0.09,5.9), mMet, 0,1.42,-0.1);

  // ── 궤도 · 전륜 ──
  [-1.42,1.42].forEach(sx=>{
    add(new THREE.BoxGeometry(0.62,1.20,6.30), mTrk, sx,0.72,0);
    add(new THREE.BoxGeometry(0.70,0.20,6.34), mTrk, sx,1.30,0);   // 상부 궤도
    for(let k=0;k<6;k++)
      add(new THREE.CylinderGeometry(0.42,0.42,0.50,14), mMet,
          sx,0.55,-2.35+k*0.94, 0,0,Math.PI/2);
    add(new THREE.CylinderGeometry(0.50,0.50,0.52,12), mMet, sx,0.80, 3.02,0,0,Math.PI/2);
    add(new THREE.CylinderGeometry(0.50,0.50,0.52,12), mMet, sx,0.80,-3.02,0,0,Math.PI/2);
  });

  // ── 포탑 (독립 회전) ──
  const T=new THREE.Group(); T.position.set(0,1.88,-0.35); g.add(T);
  const tadd=(p,mm,x,y,z,rx,ry,rz)=>{const o=new THREE.Mesh(p,mm);
    o.position.set(x,y,z); if(rx)o.rotation.x=rx; if(ry)o.rotation.y=ry;
    if(rz)o.rotation.z=rz; o.castShadow=true; T.add(o); return o;};

  tadd(new THREE.CylinderGeometry(1.42,1.62,0.92,8), mTur, 0,0.46,0, 0,Math.PI/8);
  tadd(new THREE.BoxGeometry(2.05,0.72,1.55), mTur, 0,0.50,1.05);   // 전면 확장
  tadd(new THREE.BoxGeometry(1.75,0.55,1.05), mTur, 0,0.42,-1.42);  // 후방 수납
  tadd(new THREE.BoxGeometry(1.55,0.42,0.55), mMet, 0,0.80,-1.55);  // 잡물함
  tadd(new THREE.CylinderGeometry(0.44,0.44,0.34,12), mTur, -0.55,0.99,-0.32); // 큐폴라
  tadd(new THREE.BoxGeometry(0.30,0.16,0.72), mMet, -0.55,1.22,0.05);          // 기관총
  tadd(new THREE.CylinderGeometry(0.30,0.30,0.24,10), mTur, 0.60,0.99,-0.30);  // 해치
  [-1.05,1.05].forEach(sx=>
    tadd(new THREE.CylinderGeometry(0.035,0.035,2.2,5), mMet, sx,1.60,-1.20)); // 안테나
  // 연막탄 발사기
  [-1.12,1.12].forEach(sx=>{ for(let k=0;k<3;k++)
    tadd(new THREE.CylinderGeometry(0.10,0.10,0.42,7), mAcc,
         sx,0.62,0.30+k*0.24, Math.PI/2.6); });

  // ── 주포 (앙각) ──
  const Gun=new THREE.Group(); Gun.position.set(0,0.46,1.35); T.add(Gun);
  const gadd=(p,mm,z,r)=>{const o=new THREE.Mesh(p,mm);
    o.rotation.x=Math.PI/2; o.position.set(0,0,z); o.castShadow=true;
    Gun.add(o); return o;};
  gadd(new THREE.SphereGeometry(0.62,12,10), mTur, 0.10);          // 방수포
  gadd(new THREE.CylinderGeometry(0.155,0.175,4.9,12), mMet, 2.55);// 포신
  gadd(new THREE.CylinderGeometry(0.235,0.235,2.0,12), mTur, 1.75);// 열차폐막
  gadd(new THREE.CylinderGeometry(0.255,0.255,0.62,12), mMet, 4.85);// 포구제퇴기

  // 포구 섬광
  const flash=new THREE.Mesh(new THREE.SphereGeometry(1.0,10,8),
    new THREE.MeshBasicMaterial({color:0xffd27a,transparent:true,opacity:0}));
  flash.position.set(0,0,5.3); Gun.add(flash);

  // 피격 점멸에 쓸 재질 목록. 원래 색을 기억해 둔다.
  const mats=[mHull,mTur,mMet,mAcc,mTrk];
  mats.forEach(m=>{ if(!m.emissive) m.emissive=new THREE.Color(0x000000);
    m.userData={emis:m.emissive.getHex(), inten:m.emissiveIntensity||0}; });

  g.userData={turret:T, gun:Gun, flash:flash, ring:fring, mats:mats};
  return g;
}

// ══════════════════════════════════════════
//  포탄 비행 · 폭발 효과
// ══════════════════════════════════════════
let SHELLS=[], BOOMS=[], HITFX=[];

/** 피격 점멸 — 맞은 전차가 하얗게 번쩍이며 흔들린다. */
function hitFlash(tank, base, mine){
  if(!tank) return;
  HITFX.push({tank, base:base.clone(), t:0, dur:1.25});
  document.body.classList.add('shk');
  setTimeout(()=>document.body.classList.remove('shk'),430);
  // 화면 가장자리 섬광 — 적 명중은 녹색, 아군 피격은 적색
  const v=$('#vig');
  v.classList.toggle('hitok', !mine);
  v.classList.add('on');
  setTimeout(()=>v.classList.remove('on'),90);
  setTimeout(()=>{v.classList.add('on');
    setTimeout(()=>v.classList.remove('on'),80);},190);
  // HP 바 점멸
  const b=$(mine?'#h-myb':'#h-enb');
  b.classList.remove('flash'); void b.offsetWidth; b.classList.add('flash');
  toast(mine?'⚠ 아군 피격':'✦ 적 명중');
}
function stepHit(dt){
  for(let i=HITFX.length-1;i>=0;i--){
    const h=HITFX[i]; h.t+=dt;
    const u=Math.min(1,h.t/h.dur);
    // 6 회 깜빡이며 점점 약해진다
    const on=Math.sin(u*Math.PI*12)>0 ? 1 : 0;
    const k=on*(1-u)*(1-u);
    h.tank.userData.mats.forEach(m=>{
      m.emissive.setHex(k>0.25 ? 0xffffff : m.userData.emis);
      m.emissiveIntensity=m.userData.inten + k*5.0;
    });
    // 충격으로 흔들림
    const sh=(1-u)*(1-u)*0.9;
    h.tank.position.set(h.base.x+(Math.random()-0.5)*sh,
                        h.base.y+Math.abs(Math.random()-0.5)*sh*0.7,
                        h.base.z+(Math.random()-0.5)*sh);
    const rg=h.tank.userData.ring;
    if(rg) rg.material.color.setHex(k>0.25 ? 0xffffff : rg.userData.base);
    if(u>=1){
      h.tank.userData.mats.forEach(m=>{
        m.emissive.setHex(m.userData.emis); m.emissiveIntensity=m.userData.inten;});
      if(rg) rg.material.color.setHex(rg.userData.base);
      HITFX.splice(i,1);
    }
  }
}
function glowTex(){
  if(glowTex._t) return glowTex._t;
  const s=64,c=document.createElement('canvas'); c.width=c.height=s;
  const g=c.getContext('2d'), gr=g.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
  gr.addColorStop(0,'rgba(255,240,190,1)'); gr.addColorStop(.35,'rgba(255,180,80,.55)');
  gr.addColorStop(1,'rgba(255,140,40,0)');
  g.fillStyle=gr; g.fillRect(0,0,s,s);
  return glowTex._t=new THREE.CanvasTexture(c);
}

/** 발사 → 탄착 포물선을 따라 날아가는 탄 하나를 만든다. */
function spawnShell(sh){
  if(!RN||!sh.fire||!sh.imp) return;
  const core=new THREE.Mesh(new THREE.SphereGeometry(0.5,10,8),
    new THREE.MeshBasicMaterial({color:0xfff0c0}));
  const halo=new THREE.Sprite(new THREE.SpriteMaterial(
    {map:glowTex(),color:0xffb347,transparent:true,opacity:.95,depthTest:false}));
  halo.scale.set(5,5,1); core.add(halo);
  const tail=new THREE.Line(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({color:0xffc98a,transparent:true,opacity:.75}));
  tail.frustumCulled=false;
  RN.fxG.add(core); RN.fxG.add(tail);
  const a=sh.fire, b=sh.imp;
  SHELLS.push({core, tail, a, b, pts:[], t:0,
    dur: Math.max(0.55, Math.min(2.6, (sh.tof||0.9)*1.7)),
    h0: hAt(a[0],a[1])*VS+2.7, h1: hAt(b[0],b[1])*VS+0.5,
    lift: Math.min(30, dist2(a,b)*0.22), hit: !!sh.hit});
  // 포구 섬광
  if(RN.me.userData.flash) RN.me.userData.flash.material.opacity=1.0;
}

/** 탄착 폭발 */
function boom(x,z,hit){
  if(!RN) return;
  const p=V(x,z,0.8);
  const core=new THREE.Mesh(new THREE.SphereGeometry(1,14,10),
    new THREE.MeshBasicMaterial({color:hit?0xfff4b0:0xffd79a,transparent:true,opacity:1}));
  core.position.copy(p);
  const ring=new THREE.Mesh(new THREE.RingGeometry(0.6,1.15,40),
    new THREE.MeshBasicMaterial({color:hit?0x7dfba0:0xffc46b,transparent:true,
      opacity:.95,side:THREE.DoubleSide,depthWrite:false}));
  ring.rotation.x=-Math.PI/2; ring.position.copy(p).setY(p.y+0.35);
  const smoke=new THREE.Sprite(new THREE.SpriteMaterial(
    {map:glowTex(),color:hit?0xc9f7d6:0x8d8578,transparent:true,opacity:.85,depthTest:false}));
  smoke.scale.set(7,7,1); smoke.position.copy(p).setY(p.y+2.5);
  RN.fxG.add(core); RN.fxG.add(ring); RN.fxG.add(smoke);
  BOOMS.push({core, ring, smoke, t:0, dur:1.15, hit});
}

/** 매 프레임 진행 */
function stepFX(dt){
  if(!RN) return;
  for(let i=SHELLS.length-1;i>=0;i--){
    const s=SHELLS[i]; s.t+=dt;
    const u=Math.min(1,s.t/s.dur);
    const x=s.a[0]+(s.b[0]-s.a[0])*u, z=s.a[1]+(s.b[1]-s.a[1])*u;
    const y=s.h0+(s.h1-s.h0)*u + s.lift*Math.sin(Math.PI*u);
    const w=W2S(x,z);
    s.core.position.set(w[0],y,w[1]);
    s.pts.push(new THREE.Vector3(w[0],y,w[1]));
    if(s.pts.length>26) s.pts.shift();
    s.tail.geometry.setFromPoints(s.pts);
    if(u>=1){
      boom(s.b[0],s.b[1],s.hit);
      // destoryObstaclesOnHit — 탄착 지점의 오브젝트를 무너뜨린다
      destroyAt(s.b[0], s.b[1]);
      // 명중이면 적 전차가 피격 점멸한다
      if(s.hit) hitFlash(RN.foe, RN.foe.position);
      RN.fxG.remove(s.core); RN.fxG.remove(s.tail);
      s.core.geometry.dispose(); s.tail.geometry.dispose();
      SHELLS.splice(i,1);
    }
  }
  for(let i=BOOMS.length-1;i>=0;i--){
    const b=BOOMS[i]; b.t+=dt;
    const u=Math.min(1,b.t/b.dur);
    b.core.scale.setScalar(1+u*5.5);
    b.core.material.opacity=Math.max(0,1-u*1.6);
    b.ring.scale.setScalar(1+u*13);
    b.ring.material.opacity=Math.max(0,0.95-u);
    b.smoke.scale.setScalar(7+u*16);
    b.smoke.position.y+=dt*5.5;
    b.smoke.material.opacity=Math.max(0,0.85-u*0.95);
    if(u>=1){[b.core,b.ring,b.smoke].forEach(o=>{RN.fxG.remove(o);
      if(o.geometry)o.geometry.dispose();}); BOOMS.splice(i,1);}
  }
}

// ══════════════════════════════════════════
//  씬 구성
// ══════════════════════════════════════════
let RN=null;
function boot(){
  if(RN||!ST||!window.THREE) return;
  TR=decodeTerrain(ST.terrain);
  if(!TR){ toast('지형 데이터를 읽지 못했다'); return; }
  $('#v-wl').textContent=waterLevel().toFixed(1)+' m';

  const sc=new THREE.Scene();
  sc.background=new THREE.Color(0x0a1018);
  sc.fog=new THREE.FogExp2(0x243a52,0.0021);      // 푸른 대기 산란
  sc.add(skyDome());

  const cam=new THREE.PerspectiveCamera(46,innerWidth/innerHeight,0.5,4000);
  const rd=new THREE.WebGLRenderer({antialias:true});
  rd.setPixelRatio(Math.min(2,devicePixelRatio));
  rd.setSize(innerWidth,innerHeight);
  rd.shadowMap.enabled=true; rd.shadowMap.type=THREE.PCFSoftShadowMap;
  rd.outputEncoding=THREE.sRGBEncoding;
  rd.toneMapping=THREE.ACESFilmicToneMapping; rd.toneMappingExposure=0.82;
  $('#gl').appendChild(rd.domElement);

  // ── 조명 ──
  // 지형 음영은 이미 정점색에 힐셰이드로 구워져 있다.
  // 그래서 3D 조명은 거의 평평하게 두고, 전차·나무만 입체로 살린다.
  sc.add(new THREE.HemisphereLight(0x93aac4,0x191710,1.05));
  const sun=new THREE.DirectionalLight(0xffeed2,0.62);
  sun.position.set(150,240,110); sun.castShadow=true;
  sun.shadow.mapSize.set(2048,2048);
  const sh=sun.shadow.camera; sh.near=1; sh.far=900;
  sh.left=sh.bottom=-190; sh.right=sh.top=190; sh.updateProjectionMatrix();
  sun.shadow.bias=-0.0012; sc.add(sun);
  const fill=new THREE.DirectionalLight(0x7ea8d8,0.30);
  fill.position.set(-160,90,-120); sc.add(fill);

  // ── 지형 ──
  const n=TR.n;
  const geo=new THREE.PlaneGeometry(SPAN,SPAN,n-1,n-1); geo.rotateX(-Math.PI/2);
  geo.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(n*n*3),3));
  const terrain=new THREE.Mesh(geo, new THREE.MeshStandardMaterial(
    {vertexColors:true, roughness:.95, metalness:.03, flatShading:false}));
  terrain.receiveShadow=true; terrain.castShadow=true; sc.add(terrain);

  // ── 수면 ──
  const water=new THREE.Mesh(new THREE.PlaneGeometry(SPAN,SPAN),
    new THREE.MeshStandardMaterial({color:0x1d4a5e,transparent:true,opacity:0.86,
      roughness:0.22,metalness:0.42}));
  water.rotation.x=-Math.PI/2; sc.add(water);

  // ── 격자 ──
  const grid=new THREE.GridHelper(SPAN,6,0x36506e,0x22354c);
  grid.material.transparent=true; grid.material.opacity=.5; sc.add(grid);

  // ── 수목 · 구조물 ──
  const mapG=new THREE.Group(); sc.add(mapG);
  const knownG=new THREE.Group(); sc.add(knownG);
  const detG=new THREE.Group(); sc.add(detG);
  const foeG=new THREE.Group(); sc.add(foeG);   // 추정 적 전차 표식

  // ── 전차 ──
  const me =buildTank({hull:0x2b5a91,turret:0x356ba6,track:0x121924,
                       metal:0x7a92ab,accent:0x1b3757,
                       ring:0x4b9cf5, emis:0x0b2140});
  const foe=buildTank({hull:0xb01410,turret:0xcc1a13,track:0x1d0d0b,
                       metal:0xd2837a,accent:0x6d0b07,
                       ring:0xff2b21, emis:0x460604});
  sc.add(me); sc.add(foe);

  // ── 선 요소 ──
  const mkL=(c,o,w)=>{const l=new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color:c,transparent:true,opacity:o===undefined?1:o}));
    l.frustumCulled=false; sc.add(l); return l;};
  const routeL=mkL(0xf5a524), losL=mkL(0x4ade80,.9),
        trailMe=mkL(0x4b9cf5,.55), trailFoe=mkL(0xef4444,.55),
        ringMin=mkL(0x38bdf8,.55), ringMax=mkL(0x38bdf8,.34),
        ringSug=mkL(0xa78bfa,.5);
  const shotG=new THREE.Group(); sc.add(shotG);
  // 2026-09-09  적이 쏜 탄도 · 탄착. 아군 것과 색으로 구분한다.
  const foeShotG=new THREE.Group(); sc.add(foeShotG);
  const fxG=new THREE.Group(); sc.add(fxG);          // 비행 탄 · 폭발 효과
  // 포신 지향선 — 주포가 실제로 겨누는 방향
  const aimRay=new THREE.Line(new THREE.BufferGeometry(),
    new THREE.LineDashedMaterial({color:0x9fe8ff,dashSize:3,gapSize:2.4,
      transparent:true,opacity:.65}));
  aimRay.frustumCulled=false; sc.add(aimRay);

  // ── 빌보드 ──
  const spr=(canvas,scale,order)=>{const s=new THREE.Sprite(new THREE.SpriteMaterial(
      {map:new THREE.CanvasTexture(canvas),depthTest:false,transparent:true}));
    s.scale.set(canvas.width*scale,canvas.height*scale,1);
    s.renderOrder=order||20; sc.add(s); return s;};
  const meTag=spr(tagCanvas('아군','#4b9cf5',1,1),0.052),
        foeTag=spr(tagCanvas('적','#ef4444',1,1),0.052);
  const aimSp=spr(aimCanvas(),0.10,21);

  RN={sc,cam,rd,sun,terrain,water,grid,mapG,knownG,detG,me,foe,
      routeL,losL,trailMe,trailFoe,ringMin,ringMax,ringSug,shotG,foeShotG,fxG,aimRay,foeG,
      meTag,foeTag,aimSp,geo,
      cam3:{yaw:.62,pit:.72,dis:340,tgt:new THREE.Vector3(0,0,0)}};

  paintTerrain(); buildMapObstacles(); applyLayers();

  // ── 조작 ──
  const el=rd.domElement, C=RN.cam3;
  el.oncontextmenu=e=>e.preventDefault();
  let drag=0,px=0,py=0;
  el.onmousedown=e=>{drag=e.button===2?2:1;px=e.clientX;py=e.clientY;};
  addEventListener('mouseup',()=>drag=0);
  addEventListener('mousemove',e=>{
    if(drag===1){C.yaw-=(e.clientX-px)*.0055;
      C.pit=Math.max(.10,Math.min(1.52,C.pit-(e.clientY-py)*.0045)); follow=false; syncBtn();}
    else if(drag===2){const s=C.dis*.0015;
      C.tgt.x-=Math.cos(C.yaw)*(e.clientX-px)*s-Math.sin(C.yaw)*(e.clientY-py)*s;
      C.tgt.z+=Math.sin(C.yaw)*(e.clientX-px)*s+Math.cos(C.yaw)*(e.clientY-py)*s;
      follow=false; syncBtn();}
    px=e.clientX;py=e.clientY;});
  el.onwheel=e=>{e.preventDefault();
    C.dis=Math.max(18,Math.min(1000,C.dis*(1+Math.sign(e.deltaY)*.1)));};

  const ray=new THREE.Raycaster(), m2=new THREE.Vector2();
  el.addEventListener('mousemove',e=>{
    m2.x=(e.clientX/innerWidth)*2-1; m2.y=-(e.clientY/innerHeight)*2+1;
    ray.setFromCamera(m2,cam);
    const h=ray.intersectObject(RN.terrain);
    if(h.length){const w=S2W(h[0].point.x,h[0].point.z);
      $('#t-xy').textContent=mgrs(w[0],w[1]);
      $('#t-el').textContent=hAt(w[0],w[1]).toFixed(0)+' m';}});

  addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;
    cam.updateProjectionMatrix(); rd.setSize(innerWidth,innerHeight); sizeMini();});

  let tt=0, prev=performance.now();
  (function loop(){requestAnimationFrame(loop);
    const now=performance.now(), dt=Math.min(0.05,(now-prev)/1000); prev=now; tt+=dt;
    if(follow&&S&&S.fire&&S.fire.tm&&S.fire.tm.my){
      const m=S.fire.tm.my, s=W2S(m[0],m[2]);
      C.tgt.lerp(new THREE.Vector3(s[0],hAt(m[0],m[2])*VS,s[1]),0.12);}
    cam.position.set(C.tgt.x+Math.sin(C.yaw)*Math.cos(C.pit)*C.dis,
                     C.tgt.y+Math.sin(C.pit)*C.dis,
                     C.tgt.z+Math.cos(C.yaw)*Math.cos(C.pit)*C.dis);
    cam.lookAt(C.tgt);
    RN.water.material.opacity=0.70+Math.sin(tt*1.1)*0.05;
    // 진영 링 맥동
    const pu=0.45+Math.sin(tt*2.2)*0.18;
    [me,foe].forEach(t=>{if(t.userData.ring) t.userData.ring.material.opacity=pu;});
    // 포구 섬광 감쇠
    [me,foe].forEach(t=>{const f=t.userData.flash;
      if(f.material.opacity>0) f.material.opacity=Math.max(0,f.material.opacity-dt*7);});
    stepFX(dt); stepHit(dt); stepFall(dt);
    rd.render(sc,cam); drawCompass(C.yaw);})();
  sizeMini();
  $('#mmh').addEventListener('click', toggleMini);   // 축소 지도 접기/펴기
}

function skyDome(){
  // 안쪽을 보는 큰 구. 정점 색으로 하늘 그라디언트를 만든다.
  const g=new THREE.SphereGeometry(1600,24,16);
  const pos=g.attributes.position, col=[];
  const top=new THREE.Color(0x14314f), mid=new THREE.Color(0x3d648a),
        bot=new THREE.Color(0x1b2b3c), c=new THREE.Color();
  for(let i=0;i<pos.count;i++){
    const u=Math.max(0,Math.min(1,(pos.getY(i)/1600+1)/2));   // 0 아래 ~ 1 위
    if(u>0.52) c.copy(mid).lerp(top,(u-0.52)/0.48);
    else       c.copy(bot).lerp(mid,u/0.52);
    col.push(c.r,c.g,c.b);
  }
  g.setAttribute('color',new THREE.Float32BufferAttribute(col,3));
  const m=new THREE.Mesh(g,new THREE.MeshBasicMaterial(
    {vertexColors:true,side:THREE.BackSide,fog:false,depthWrite:false}));
  m.renderOrder=-1;
  return m;
}
/** ATAK 방식 라벨판 — 회색 둥근 판에 흰 글씨, 아래로 지시선 */
function tagCanvas(text,color){
  const f=17, padX=13, padY=8, stem=15, R=7;
  const m=document.createElement('canvas').getContext('2d');
  m.font=`700 ${f}px "Malgun Gothic",sans-serif`;
  const w=Math.ceil(m.measureText(text).width)+padX*2;
  const hh=f+padY*2;
  const c=document.createElement('canvas');
  c.width=w+6; c.height=hh+stem+6;
  const g=c.getContext('2d');
  g.font=`700 ${f}px "Malgun Gothic",sans-serif`;
  const x0=3, y0=3;
  // 그림자
  g.shadowColor='rgba(0,0,0,.75)'; g.shadowBlur=6; g.shadowOffsetY=2;
  // 둥근 판
  g.beginPath();
  g.moveTo(x0+R,y0); g.lineTo(x0+w-R,y0); g.quadraticCurveTo(x0+w,y0,x0+w,y0+R);
  g.lineTo(x0+w,y0+hh-R); g.quadraticCurveTo(x0+w,y0+hh,x0+w-R,y0+hh);
  g.lineTo(x0+R,y0+hh); g.quadraticCurveTo(x0,y0+hh,x0,y0+hh-R);
  g.lineTo(x0,y0+R); g.quadraticCurveTo(x0,y0,x0+R,y0); g.closePath();
  g.fillStyle='rgba(58,64,74,.94)'; g.fill();
  g.shadowColor='transparent';
  g.strokeStyle=color; g.lineWidth=1.8; g.stroke();
  // 지시선
  g.beginPath(); g.moveTo(x0+w/2,y0+hh); g.lineTo(x0+w/2,y0+hh+stem);
  g.strokeStyle=color; g.lineWidth=1.6; g.stroke();
  // 글씨
  g.fillStyle='#f2f6fb'; g.textBaseline='middle'; g.textAlign='center';
  g.fillText(text,x0+w/2,y0+hh/2+1);
  return c;
}
function aimCanvas(){
  const s=64,c=document.createElement('canvas');c.width=c.height=s;
  const g=c.getContext('2d');
  g.strokeStyle='#fbbf24'; g.lineWidth=3;
  g.beginPath();g.arc(s/2,s/2,18,0,7);g.stroke();
  g.lineWidth=2.4;
  [[0,-26,0,-11],[0,11,0,26],[-26,0,-11,0],[11,0,26,0]].forEach(v=>{
    g.beginPath();g.moveTo(s/2+v[0],s/2+v[1]);g.lineTo(s/2+v[2],s/2+v[3]);g.stroke();});
  g.fillStyle='#fbbf24';g.beginPath();g.arc(s/2,s/2,3,0,7);g.fill();
  return c;
}

// ── 절차적 잡음 (위성사진의 얼룩덜룩한 질감을 만든다) ──
function hash2(i,j){
  let h=(i*374761393+j*668265263)|0;
  h=(h^(h>>>13))*1274126177|0;
  return ((h^(h>>>16))>>>0)/4294967295;
}
function noise2(x,y){
  const i=Math.floor(x), j=Math.floor(y), fx=x-i, fy=y-j;
  const u=fx*fx*(3-2*fx), v=fy*fy*(3-2*fy);
  const a=hash2(i,j), b=hash2(i+1,j), c=hash2(i,j+1), d=hash2(i+1,j+1);
  return (a*(1-u)+b*u)*(1-v)+(c*(1-u)+d*u)*v;
}
function fbm(x,y){
  let s=0,a=0.5,f=1;
  for(let k=0;k<4;k++){ s+=a*noise2(x*f,y*f); a*=0.5; f*=2.03; }
  return s;                      // 0 ~ 1 근처
}
const LERP3=(A,B,t)=>[A[0]+(B[0]-A[0])*t, A[1]+(B[1]-A[1])*t, A[2]+(B[2]-A[2])*t];

// ── 지형 채색 ──
function paintTerrain(){
  if(!RN||!TR) return;
  const n=TR.n, pos=RN.geo.attributes.position, col=RN.geo.attributes.color;
  const rng=(TR.hi-TR.lo)||1, wl=waterLevel();
  const C=new THREE.Color(), H=TR.H;

  // ── 위성사진 팔레트 ──────────────────────────────
  // 전부 어둡고 채도가 낮다. 실제 항공/위성 영상은 이 대역에 있다.
  // 실제 지형 데이터로 렌더 시험한 값. 평균 밝기 0.277 / 대비 0.091 —
  // 참조로 삼은 전술지도(ATAK) 화면과 같은 대역이다.
  const P_SHOAL=[0.026,0.062,0.082];   // 얕은 물
  const P_SHORE=[0.118,0.108,0.082];   // 물가 모래톱
  const P_FOR_D=[0.026,0.046,0.022];   // 짙은 침엽수림
  const P_FOR_L=[0.046,0.070,0.033];   // 활엽수림
  const P_SCRUB=[0.076,0.092,0.046];   // 관목지
  const P_GRASS=[0.104,0.104,0.060];   // 초지
  const P_DRY  =[0.142,0.128,0.082];   // 마른 풀 · 황토
  const P_SOIL =[0.156,0.132,0.096];   // 나지
  const P_ROCK =[0.132,0.128,0.118];   // 암반
  const P_RIDGE=[0.192,0.186,0.174];   // 능선 노출암

  // 힐셰이드용 태양 (북서 315°, 고도 45°) — 위성 영상 관례
  const SAZ=315*Math.PI/180, SEL=45*Math.PI/180;
  const SUNV=[Math.sin(SAZ)*Math.cos(SEL), Math.sin(SEL), -Math.cos(SAZ)*Math.cos(SEL)];
  const cell=SPAN/(n-1);

  for(let j=0;j<n;j++)for(let i=0;i<n;i++){
    const k=idx(i,j), h=H[k];
    pos.setY(k,h*VS);
    let r,g,b;

    if(SURF==='slope'){ const s=TR.sl?TR.sl[k]/255:0;
      C.setHSL(0.34-0.34*s,0.55,0.20+0.26*s); r=C.r;g=C.g;b=C.b; }
    else if(SURF==='expo'){ const e=TR.ex?TR.ex[k]/255:0;
      C.setHSL(0.62-0.62*e,0.62,0.17+0.30*e); r=C.r;g=C.g;b=C.b; }
    else if(SURF==='block'){ const q=TR.bl?TR.bl[k]:0;
      r=q?0.44:0.10; g=q?0.09:0.17; b=q?0.09:0.12; }
    // ── 2026-08-26  사격 진지 적합도 ─────────────────────────────
    //
    //   전차가 그 자리에 섰을 때 얼마나 기우는지를 색으로 보여준다.
    //   '점' 의 경사가 아니라 차체가 놓이는 5 m 반경 안의 최대 경사다.
    //   전차는 6.3 x 3.3 m 면이라 면 전체가 평평해야 안 기운다.
    //
    //   실측 근거 (8/26 기동사격 27발)
    //       평평 (0°)        9발 9명중  100 %
    //       6.3° / -10.8°    9발 8명중   89 %
    //      12.8° / -11.2°    9발 6명중   67 %
    //
    //   초록 = 여기 찍으면 된다.  붉은색 = 기운다.
    else if(SURF==='flat'){ const t=TR.ft?TR.ft[k]:255;
      if(t<=2)      { r=0.13; g=0.62; b=0.31; }   // 2° 이하 — 적합
      else if(t<=5) { r=0.42; g=0.55; b=0.20; }   // 5° 이하 — 무난
      else if(t<=10){ r=0.55; g=0.42; b=0.14; }   // 10° 이하 — 주의
      else          { r=0.45; g=0.13; b=0.13; }   // 그 이상 — 부적합
    }
    else {
      const u=(h-TR.lo)/rng;
      const s=TR.sl?TR.sl[k]/255:0;
      const i0=Math.max(1,Math.min(n-2,i)), j0=Math.max(1,Math.min(n-2,j));

      // ── 지형 기울기 벡터 (동쪽 · 남쪽 방향 성분) ──
      const dzdx=(H[idx(i0+1,j0)]-H[idx(i0-1,j0)])/(2*cell);
      const dzdy=(H[idx(i0,j0+1)]-H[idx(i0,j0-1)])/(2*cell);
      const nl=Math.hypot(dzdx,dzdy,1);
      const nrm=[-dzdx/nl, 1/nl, -dzdy/nl];
      // 사면 방향: +1 이면 남향(볕이 잘 듦) → 마른 황토, -1 이면 북향 → 짙은 숲
      const aspect=Math.max(-1,Math.min(1,dzdy/Math.max(0.05,Math.hypot(dzdx,dzdy))));

      // ── ① 고도 + 사면 방향으로 식생을 정한다 ──
      const dryness=Math.max(0,Math.min(1, u*0.95 + aspect*0.25 + s*0.35 - 0.10));
      let c1;
      if(dryness<0.24)      c1=LERP3(P_FOR_D,P_FOR_L, dryness/0.24);
      else if(dryness<0.44) c1=LERP3(P_FOR_L,P_SCRUB,(dryness-0.24)/0.20);
      else if(dryness<0.62) c1=LERP3(P_SCRUB,P_GRASS,(dryness-0.44)/0.18);
      else if(dryness<0.80) c1=LERP3(P_GRASS,P_DRY  ,(dryness-0.62)/0.18);
      else                  c1=LERP3(P_DRY  ,P_SOIL ,Math.min(1,(dryness-0.80)/0.20));

      // ── ② 급경사 · 능선은 암반이 드러난다 ──
      const rk=Math.max(0,Math.min(1,(s-0.20)/0.42));
      c1=LERP3(c1,P_ROCK,rk*0.80);
      if(u>0.86) c1=LERP3(c1,P_RIDGE,(u-0.86)/0.14*0.55);

      // ── ③ 물가 · 얕은 물 ──
      const sh=Math.max(0,Math.min(1,(h-wl)/1.8));
      c1=LERP3(P_SHORE,c1,sh);
      if(h<wl) c1=LERP3(P_SHOAL,P_SHORE,Math.max(0,Math.min(1,(h-wl+2.4)/2.4)));

      // ── ④ 수관 질감 — 숲일수록 오돌토돌하게 ──
      const nx=i*0.11, ny=j*0.11;
      const m1=fbm(nx,ny)-0.5;                      // 큰 임상 얼룩
      const m2=fbm(nx*5.1+31,ny*5.1-17)-0.5;        // 우듬지 잔결
      const m3=fbm(nx*13.0-5,ny*13.0+9)-0.5;        // 미세 노이즈
      const forest=Math.max(0,1-dryness*1.5)*Math.max(0,1-rk);
      const tex=1 + m1*0.30 + m2*(0.18+forest*0.34) + m3*0.12;
      c1=[c1[0]*tex, c1[1]*tex*(1+forest*m2*0.10), c1[2]*tex];

      // ── ⑤ 힐셰이드 — 음영을 색에 굽는다 (위성영상의 핵심) ──
      const dot=Math.max(0, nrm[0]*SUNV[0]+nrm[1]*SUNV[1]+nrm[2]*SUNV[2]);
      const hs=0.34+1.10*dot;                        // 0.34 ~ 1.44, 대비 강함
      c1=[c1[0]*hs, c1[1]*hs, c1[2]*hs];

      // ── ⑥ 침식골 — 오목한 곳을 어둡게 (물길이 드러난다) ──
      const mean=(H[idx(i0-1,j0)]+H[idx(i0+1,j0)]+
                  H[idx(i0,j0-1)]+H[idx(i0,j0+1)])*0.25;
      const curv=Math.max(-1,Math.min(1,(h-mean)*1.1));
      const ak=1+curv*0.30;
      c1=[c1[0]*ak,c1[1]*ak,c1[2]*ak];

      r=Math.max(0,c1[0]); g=Math.max(0,c1[1]); b=Math.max(0,c1[2]);
    }
    col.setXYZ(k,r,g,b);
  }
  pos.needsUpdate=true; col.needsUpdate=true;
  RN.geo.computeVertexNormals();
  RN.water.position.y=wl*VS;
  RN.grid.position.y=TR.hi*VS+20;
  drawMini._bg=null;             // 미니맵 배경도 다시 그린다
}

// ══════════════════════════════════════════
//  맵 오브젝트 — 시뮬레이터 실물에 맞춘 모형
// ══════════════════════════════════════════
const rnd=(i,a,b)=>a+((((i*2654435761)>>>0)%1000)/1000)*(b-a);

/** 세로 판자벽 텍스처 */
function plankTex(dark){
  const key='pk'+(dark?1:0); if(plankTex[key]) return plankTex[key];
  const w=128,h=128,c=document.createElement('canvas'); c.width=w;c.height=h;
  const g=c.getContext('2d');
  g.fillStyle=dark?'#38291d':'#8f7d5e'; g.fillRect(0,0,w,h);
  for(let i=0;i<16;i++){
    const x=i*8, k=Math.abs((Math.sin(i*12.9898)*43758.5453)%1);
    g.fillStyle=dark ? `rgb(${44+k*26|0},${32+k*20|0},${23+k*14|0})`
                     : `rgb(${132+k*40|0},${116+k*34|0},${88+k*26|0})`;
    g.fillRect(x,0,7,h);
    g.fillStyle='rgba(0,0,0,.30)'; g.fillRect(x+7,0,1,h);
  }
  const t=new THREE.CanvasTexture(c);
  t.wrapS=t.wrapT=THREE.RepeatWrapping;
  return plankTex[key]=t;
}

/** 수목 — 큰 키 활엽수.
 *
 *  2026-08-26  실제 숲 사진에 맞춰 되돌렸다.
 *    줄기 높이   3 m  ->  9~14 m
 *    줄기 굵기   가늘게 · 색 더 어둡게
 *    수관        원뿔 한 덩어리  ->  성긴 잎뭉치 10 개
 *    배치        줄기 위쪽 40 % 구간에만 (아래는 맨 줄기)
 *    색          자홍 + 황록을 한 나무 안에 섞는다
 */
const BLOBS=10;                       // 나무 한 그루당 잎뭉치 수
function addTrees(list){
  if(!list.length) return;
  const N=list.length;

  // 줄기 — 가늘고 어둡게. 높이 1 로 만들고 인스턴스마다 늘린다.
  const trunk=new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.10,0.20,1,5),
    mat(0x2b2018,.98,.01), N);

  // 수관 — 저폴리 잎뭉치. N x BLOBS 개를 한 InstancedMesh 로 그린다.
  const crown=new THREE.InstancedMesh(
    new THREE.IcosahedronGeometry(1,0),
    mat(0xffffff,.94,.02), N*BLOBS);

  trunk.castShadow=crown.castShadow=true; crown.receiveShadow=true;

  const d=new THREE.Object3D();
  const rec=[];                       // 파괴 처리에 쓸 나무별 정보
  const cMag=new THREE.Color(0xc2417f);   // 자홍
  const cLim=new THREE.Color(0xa8bf46);   // 황록
  const tmp=new THREE.Color();

  // 결정론적 난수 — 새로고침해도 숲 모양이 같아야 비교가 된다
  const rnd=(a,b)=>{ const t=Math.sin(a*127.1+b*311.7)*43758.5453;
                     return t-Math.floor(t); };

  list.forEach((o,i)=>{
    const s=W2S(o.x,o.z), y=hAt(o.x,o.z)*VS;
    const H=(9+rnd(i,1)*5)*VS;        // 줄기 높이 9~14 m
    const R=0.9+rnd(i,2)*0.5;         // 수관 반경 계수

    d.position.set(s[0], y+H/2, s[1]);
    d.scale.set(1, H, 1);
    d.rotation.set(0, i*1.1, 0);
    d.updateMatrix(); trunk.setMatrixAt(i, d.matrix);

    for(let b=0;b<BLOBS;b++){
      // 위쪽 40 % 구간에만 흩뿌린다
      const f=0.60+rnd(i,10+b)*0.40;
      const ang=rnd(i,30+b)*Math.PI*2;
      const rad=(0.4+rnd(i,50+b)*1.5)*R;
      const bs=(0.75+rnd(i,70+b)*0.95)*R;
      d.position.set(s[0]+Math.cos(ang)*rad, y+H*f, s[1]+Math.sin(ang)*rad);
      d.scale.set(bs, bs*0.8, bs);
      d.rotation.set(rnd(i,90+b)*3, rnd(i,110+b)*3, 0);
      d.updateMatrix(); crown.setMatrixAt(i*BLOBS+b, d.matrix);
      // 한 나무 안에서도 자홍과 황록이 섞이게
      tmp.copy(cMag).lerp(cLim, rnd(i,130+b));
      crown.setColorAt(i*BLOBS+b, tmp);
    }
    rec.push({x:o.x, z:o.z, i:i, sx:s[0], sz:s[1], y:y, sc:1, H:H, R:R,
              dead:false});
  });
  if(crown.instanceColor) crown.instanceColor.needsUpdate=true;
  RN.mapG.add(trunk); RN.mapG.add(crown);
  RN.trees={trunk, crown, rec, d:new THREE.Object3D()};
}

// ══════════════════════════════════════════
//  파괴 — destoryObstaclesOnHit 반영
//  시뮬레이터가 맞은 오브젝트를 없애므로 화면에서도 무너뜨린다.
// ══════════════════════════════════════════
const KILL_R={House:5.0, Tent:3.2, Car:3.0, Human:1.6, Tank:4.0, Tree:2.2};
let FALLING=[];

/** 탄착 지점 주변 오브젝트를 무너뜨린다. */
function destroyAt(wx,wz){
  if(!RN) return 0;
  let n=0;
  // 개별 오브젝트 (헛간 · 천막 · 차량 · 사람 · 전차 소품)
  (RN.mapObjs||[]).forEach(o=>{
    if(o.dead) return;
    if(Math.hypot(o.x-wx,o.z-wz) <= (KILL_R[o.kind]||2.5)){
      o.dead=true; n++;
      FALLING.push({obj:o.obj, t:0, dur:0.85,
        y0:o.obj.position.y, tilt:(Math.random()-0.5)*1.6});
      dustPuff(o.x,o.z, o.kind==='House'?2.2:1.3);
    }
  });
  // 수목 — 인스턴스를 0 으로 줄여 없앤다
  const T=RN.trees;
  if(T){
    T.rec.forEach(r=>{
      if(r.dead) return;
      if(Math.hypot(r.x-wx,r.z-wz) <= KILL_R.Tree){
        r.dead=true; n++;
        FALLING.push({tree:r, t:0, dur:0.7});
        dustPuff(r.x,r.z,1.0);
      }
    });
  }
  return n;
}

/** 먼지 · 파편 */
function dustPuff(wx,wz,scale){
  const p=V(wx,wz,1.0);
  const s=new THREE.Sprite(new THREE.SpriteMaterial(
    {map:glowTex(),color:0x9a8f7c,transparent:true,opacity:.8,depthTest:false}));
  s.scale.set(6*scale,6*scale,1); s.position.copy(p);
  RN.fxG.add(s);
  BOOMS.push({core:s, ring:s, smoke:s, t:0, dur:1.0, dustOnly:true});
}

/** 무너지는 중인 것들을 매 프레임 진행 */
function stepFall(dt){
  if(!RN) return;
  const T=RN.trees;
  for(let i=FALLING.length-1;i>=0;i--){
    const f=FALLING[i]; f.t+=dt;
    const u=Math.min(1,f.t/f.dur);
    if(f.obj){
      f.obj.rotation.z=f.tilt*u;
      f.obj.position.y=f.y0-u*u*3.2;
      f.obj.scale.setScalar(Math.max(0.01,1-u*0.55));
      f.obj.traverse(o=>{ if(o.material&&!o.material._t){
        o.material.transparent=true; o.material._t=1; }
        if(o.material) o.material.opacity=1-u; });
      if(u>=1){ RN.mapG.remove(f.obj); FALLING.splice(i,1); }
    }else if(f.tree&&T){
      // 2026-08-26  수관이 잎뭉치 BLOBS 개로 바뀌어 함께 줄인다.
      const r=f.tree, k=Math.max(0.001,1-u);
      T.d.position.set(r.sx, r.y+r.H*k/2, r.sz);
      T.d.scale.set(k, r.H*k, k); T.d.rotation.set(u*1.3, r.i*1.1, 0);
      T.d.updateMatrix(); T.trunk.setMatrixAt(r.i,T.d.matrix);
      for(let b=0;b<BLOBS;b++){
        T.d.position.set(r.sx, r.y+r.H*k*0.7, r.sz);
        T.d.scale.setScalar(0.001);
        T.d.updateMatrix(); T.crown.setMatrixAt(r.i*BLOBS+b, T.d.matrix);
      }
      T.trunk.instanceMatrix.needsUpdate=true;
      T.crown.instanceMatrix.needsUpdate=true;
      if(u>=1) FALLING.splice(i,1);
    }else FALLING.splice(i,1);
  }
}

/** 목조 헛간 — 어두운 판자벽 + 박공지붕 + 밝은 서까래 + 흰 문 */
function buildHouse(i){
  const g=new THREE.Group();
  const W=6.4, D=6.0, Hh=3.4, RH=3.0;
  const wallM=new THREE.MeshStandardMaterial(
    {map:plankTex(true),roughness:.96,metalness:.02});
  wallM.map.repeat.set(2,1.2);
  const beamM=mat(0x9c8a68,.92,.03);
  const body=new THREE.Mesh(new THREE.BoxGeometry(W,Hh,D), wallM);
  body.position.y=Hh/2; body.castShadow=body.receiveShadow=true; g.add(body);
  const gable=new THREE.Shape();
  gable.moveTo(-W/2,0); gable.lineTo(W/2,0); gable.lineTo(0,RH); gable.closePath();
  [D/2,-D/2].forEach(z=>{
    const m=new THREE.Mesh(new THREE.ShapeGeometry(gable),
      new THREE.MeshStandardMaterial({map:plankTex(true),roughness:.96,metalness:.02}));
    m.position.set(0,Hh,z); if(z<0) m.rotation.y=Math.PI;
    m.castShadow=true; g.add(m);
  });
  const slope=Math.atan2(RH,W/2), len=Math.hypot(W/2,RH);
  [1,-1].forEach(sx=>{
    const m=new THREE.Mesh(new THREE.BoxGeometry(len,0.14,D+0.5), beamM);
    m.position.set(sx*W/4, Hh+RH/2, 0); m.rotation.z=-sx*slope;
    m.castShadow=true; g.add(m);
  });
  for(let k=-2;k<=2;k++){
    const m=new THREE.Mesh(new THREE.BoxGeometry(W+0.3,0.13,0.13), beamM);
    m.position.set(0, Hh+0.15, k*D/5); m.castShadow=true; g.add(m);
  }
  const ridge=new THREE.Mesh(new THREE.BoxGeometry(0.18,0.18,D+0.6), beamM);
  ridge.position.set(0,Hh+RH,0); g.add(ridge);
  const white=mat(0xe8e4da,.85,.02);
  const door=new THREE.Mesh(new THREE.BoxGeometry(1.1,1.9,0.12), white);
  door.position.set(-0.9,0.95,D/2+0.02); g.add(door);
  const win=new THREE.Mesh(new THREE.BoxGeometry(0.9,0.8,0.12), white);
  win.position.set(1.2,2.1,D/2+0.02); g.add(win);
  // DemoMap.map 의 rotation 이 전부 (0,0,0,1) 이므로 회전을 주지 않는다.
  return g;
}

/** 개방형 A형 천막 골조 */
function buildTent(i){
  const g=new THREE.Group();
  const W=3.6, D=4.4, Hh=2.6;
  const woodM=mat(0x4a4238,.95,.03), barM=mat(0x6d6252,.94,.03);
  [1,-1].forEach(sx=>{
    const len=Math.hypot(W/2,Hh), slope=Math.atan2(Hh,W/2);
    for(let k=0;k<7;k++){
      const t=(k+0.5)/7;
      const m=new THREE.Mesh(new THREE.BoxGeometry(0.12,0.12,D), barM);
      m.position.set(sx*(W/2)*(1-t), Hh*t, 0);
      m.castShadow=true; g.add(m);
    }
    const p=new THREE.Mesh(new THREE.BoxGeometry(len,0.09,D), woodM);
    p.position.set(sx*W/4, Hh/2, 0); p.rotation.z=-sx*slope;
    p.castShadow=p.receiveShadow=true; g.add(p);
  });
  const rdg=new THREE.Mesh(new THREE.BoxGeometry(0.14,0.14,D+0.3), barM);
  rdg.position.y=Hh; g.add(rdg);
  [W/2,-W/2].forEach(x=>{
    const m=new THREE.Mesh(new THREE.BoxGeometry(0.16,0.16,D+0.3), barM);
    m.position.set(x,0.08,0); g.add(m);
  });
  return g;
}

/** 차량 — 승합 · 스포츠 · 픽업 · 세단 */
const CAR_COL=[{body:0x1f9c72,roof:0x1f9c72,kind:'van'},
               {body:0x2331b8,roof:0x14161f,kind:'coupe'},
               {body:0xe8b423,roof:0xf2efe6,kind:'pickup'},
               {body:0xc9b489,roof:0xc9b489,kind:'sedan'}];
function buildCar(i){
  const g=new THREE.Group();
  const C=CAR_COL[i%CAR_COL.length];
  const bodyM=mat(C.body,.42,.55), roofM=mat(C.roof,.42,.55);
  const glass=new THREE.MeshStandardMaterial(
    {color:0x1b2530,roughness:.12,metalness:.85});
  const tire=mat(0x14161a,.95,.05);
  const van=C.kind==='van', pick=C.kind==='pickup', coupe=C.kind==='coupe';
  const L=van?4.9:pick?5.2:4.4, W=1.95, Hb=van?1.5:coupe?0.62:0.78;
  const b=new THREE.Mesh(new THREE.BoxGeometry(W,Hb,L), bodyM);
  b.position.y=0.62+Hb/2; b.castShadow=b.receiveShadow=true; g.add(b);
  if(!van){
    const cl=pick?1.9:coupe?2.1:2.4;
    const cab=new THREE.Mesh(new THREE.BoxGeometry(W*0.92,coupe?0.62:0.78,cl), roofM);
    cab.position.set(0,0.62+Hb+(coupe?0.31:0.39), pick?0.9:0.15);
    cab.castShadow=true; g.add(cab);
    const win=new THREE.Mesh(
      new THREE.BoxGeometry(W*0.94,coupe?0.42:0.52,cl*0.94), glass);
    win.position.copy(cab.position); g.add(win);
    if(pick){
      const bed=new THREE.Mesh(new THREE.BoxGeometry(W*0.94,0.55,2.0), bodyM);
      bed.position.set(0,0.62+Hb+0.27,-1.5); g.add(bed);
    }
  }else{
    const win=new THREE.Mesh(new THREE.BoxGeometry(W*1.01,0.62,L*0.62), glass);
    win.position.set(0,0.62+Hb*0.72,0.35); g.add(win);
  }
  [[-1,1],[1,1],[-1,-1],[1,-1]].forEach(v=>{
    const w=new THREE.Mesh(new THREE.CylinderGeometry(0.36,0.36,0.28,14), tire);
    w.rotation.z=Math.PI/2;
    w.position.set(v[0]*(W/2-0.06), 0.36, v[1]*(L/2-1.05));
    w.castShadow=true; g.add(w);
  });
  return g;
}

/**
 * 사람 — 어두운 지형 위에서 형태가 읽히도록 밝은 카키로 올렸다.
 * 머리 · 몸통 · 팔 · 다리를 나눠 실루엣이 사람으로 보이게 한다.
 */
function buildHuman(i){
  const g=new THREE.Group();
  const skin =mat(0xd8b189,.88,.02);          // 밝은 피부
  const cloth=mat(0xb9ad84,.90,.03);          // 밝은 카키 상의
  const pants=mat(0x7d7458,.92,.03);          // 하의
  const add=(geo,m,x,y,z,rz)=>{const o=new THREE.Mesh(geo,m);
    o.position.set(x,y,z); if(rz)o.rotation.z=rz;
    o.castShadow=true; o.receiveShadow=true; g.add(o); return o;};

  add(new THREE.CylinderGeometry(0.19,0.23,0.62,10), cloth, 0,1.20,0);   // 몸통
  add(new THREE.SphereGeometry(0.135,12,10), skin, 0,1.42,0);            // 어깨선
  add(new THREE.SphereGeometry(0.155,12,10), skin, 0,1.66,0);            // 머리
  add(new THREE.CylinderGeometry(0.055,0.05,0.58,7), skin, -0.245,1.16,0, 0.13);  // 팔
  add(new THREE.CylinderGeometry(0.055,0.05,0.58,7), skin,  0.245,1.16,0,-0.13);
  [-0.095,0.095].forEach(x=>
    add(new THREE.CylinderGeometry(0.085,0.075,0.90,7), pants, x,0.45,0));       // 다리
  [-0.095,0.095].forEach(x=>
    add(new THREE.BoxGeometry(0.13,0.08,0.26), mat(0x3a3428,.95,.03), x,0.04,0.04));

  return g;
}

/** 맵에 놓인 정지 전차 (소품 — 살아 있는 적이 아니다) */
function buildPropTank(i){
  const g=buildTank({hull:0x3a4038,turret:0x434a40,track:0x14170f,
                     metal:0x6d7466,accent:0x252a20,
                     ring:0x8d9099, emis:0x000000});
  if(g.userData.ring) g.userData.ring.visible=false;
  g.userData.turret.rotation.y=rnd(i+59,0,6.28);
  return g;
}

function buildMapObstacles(){
  if(!RN||!ST.obstacles) return;
  RN.mapG.clear(); RN.mapObjs=[]; FALLING=[];
  addTrees(ST.obstacles.filter(o=>o.kind==='Tree'));
  const BUILD={House:buildHouse, Tent:buildTent, Car:buildCar,
               Human:buildHuman, Tank:buildPropTank};
  let idx=0;
  ST.obstacles.filter(o=>o.kind!=='Tree').forEach(o=>{
    const f=BUILD[o.kind];
    let m;
    if(f) m=f(idx++);
    else { m=new THREE.Mesh(new THREE.BoxGeometry(2,2,2), mat(0x5b6470,.9,.05));
           m.castShadow=true; }
    const s=W2S(o.x,o.z);
    m.position.set(s[0], hAt(o.x,o.z)*VS, s[1]);
    RN.mapG.add(m);
    RN.mapObjs.push({kind:o.kind, x:o.x, z:o.z, obj:m, dead:false});
  });
}

function applyLayers(){
  if(!RN) return;
  RN.mapG.visible=L.mapobs; RN.knownG.visible=L.known; RN.detG.visible=L.det;
  if(RN.foeG) RN.foeG.visible=L.foes;
  RN.routeL.visible=L.route; RN.losL.visible=L.los;
  RN.trailMe.visible=RN.trailFoe.visible=L.trail;
  RN.ringMin.visible=RN.ringMax.visible=RN.ringSug.visible=L.rings;
  RN.shotG.visible=L.shots; RN.grid.visible=L.grid; RN.water.visible=L.water;
  RN.foeShotG.visible=L.foeshots;
  RN.meTag.visible=RN.foeTag.visible=L.label; RN.aimSp.visible=L.aim;
  if(RN.aimRay) RN.aimRay.visible=L.ray;
}

// ── 컨트롤 ──
$('#vs').oninput=e=>{VS=+e.target.value; $('#v-vs').textContent=VS.toFixed(1)+'×';
  paintTerrain(); if(S)update3D();};
$('#wl').oninput=e=>{WATERP=+e.target.value;
  $('#v-wl').textContent=waterLevel().toFixed(1)+' m'; paintTerrain();};
$('#surf').onchange=e=>{SURF=e.target.value; paintTerrain();};
$('#b-reset').onclick=()=>{if(!RN)return; const C=RN.cam3;
  C.yaw=.62;C.pit=.72;C.dis=340;C.tgt.set(0,0,0);follow=false;syncBtn();};
$('#b-top').onclick=()=>{if(!RN)return; const C=RN.cam3;
  C.pit=1.50;C.yaw=0;C.dis=330;C.tgt.set(0,0,0);follow=false;syncBtn();};
$('#b-follow').onclick=()=>{follow=!follow; if(follow&&RN)RN.cam3.dis=Math.min(RN.cam3.dis,90);
  syncBtn(); toast(follow?'아군 추적 켬':'아군 추적 끔');};
function syncBtn(){$('#b-follow').classList.toggle('on',follow);}

// ══════════════════════════════════════════
//  데이터
// ══════════════════════════════════════════
fetch(API+'/static-data').then(r=>r.json()).then(d=>{ST=d;boot();});
async function poll(){
  try{const r=await fetch(API+'/state',{cache:'no-store'});
    S=await r.json(); paint();}catch(e){}
}
setInterval(poll,220); poll();

const T=(v,u='')=>(v===null||v===undefined)?'–':(v+u);
function gateRow(n,pass,note){
  const c=pass===null?'off':(pass?'pass':'fail');
  const t=pass===null?'꺼짐':(pass?'통과':'차단');
  return `<div class="row"><span>${n}</span><b><span style="color:var(--dim);
    margin-right:6px;font-size:10px">${note||''}</span>
    <span class="pill ${c}">${t}</span></b></div>`;
}

function paint(){
  if(!S||!S.fire) return;
  const f=S.fire, tm=f.tm||{}, g=f.gates||{}, tk=f.track||{},
        dv=S.drive||{}, dt=S.detect||{};

  $('#r-t').textContent = tm.t!=null?('t '+tm.t.toFixed(1)):'';
  $('#t-st').textContent= T(f.state);
  $('#t-sp').textContent= tm.my_speed!=null?tm.my_speed.toFixed(1)+' m/s':'–';

  const hp=(v)=>v==null?100:Math.max(0,Math.min(100,+v));
  $('#h-my').textContent=T(tm.my_hp); $('#h-myb').style.width=hp(tm.my_hp)+'%';
  $('#h-en').textContent=T(tm.enemy_hp); $('#h-enb').style.width=hp(tm.enemy_hp)+'%';
  // HP 가 줄면 피격 점멸 (적탄에 맞은 경우도 잡힌다)
  if(paint._mhp!=null && tm.my_hp!=null && tm.my_hp<paint._mhp && RN)
    hitFlash(RN.me, RN.me.position, true);
  if(paint._ehp!=null && tm.enemy_hp!=null && tm.enemy_hp<paint._ehp && RN
     && !HITFX.some(h=>h.tank===RN.foe))
    hitFlash(RN.foe, RN.foe.position, false);
  paint._mhp=tm.my_hp; paint._ehp=tm.enemy_hp;

  if(tm.my&&tm.enemy){
    const a=[tm.my[0],tm.my[2]], e=[tm.enemy[0],tm.enemy[2]];
    $('#t-rg').textContent=dist2(a,e).toFixed(0)+' m';
    $('#t-br').textContent=brg(a,e).toFixed(0)+'°';
  }
  $('#r-d').textContent =T(f.dist,' m');
  $('#r-sg').textContent=T(f.suggest,' m');
  $('#r-ev').textContent=f.envelope?f.envelope[0]+'–'+f.envelope[1]+' m':'–';
  $('#r-ph').textContent=T(f.p_hit);
  $('#r-tf').textContent=T(f.tof,' s');
  $('#r-rl').textContent=f.reload_left>0?f.reload_left+' s':'준비';
  $('#r-fh').textContent=T(f.fired)+' / '+T(f.hits);
  $('#r-hr').textContent=T(f.hit_rate,' %');
  // 2026-08-26  차체 자세 경고
  //
  //   8/26 기동사격 27발 실측 — 차체 기울기가 명중률을 그대로 가른다.
  //       평평 (0°)        9발 9명중  100 %   평균오차 0.60 m
  //       6.3° / -10.8°    9발 8명중   89 %   평균오차 0.73 m
  //      12.8° / -11.2°    9발 6명중   67 %   평균오차 0.89 m
  //   빗나간 4발이 전부 기운 상태였다. 알고리즘이 아니라 선 자리 문제다.
  //   그래서 값만 보여주지 않고 색으로 경고한다.
  {
    const py=Math.abs(+tm.body_y||0), rz=Math.abs(+tm.body_z||0);
    const tilt=Math.max(py,rz);
    const e=$('#r-hull');
    e.textContent=T(tm.body_y,'°')+' / '+T(tm.body_z,'°');
    e.style.color = tilt>=8 ? 'var(--no)' : (tilt>=4 ? 'var(--warn)' : '');
    e.title = tilt>=8 ? '차체가 많이 기울었다. 실측 명중률 67 %'
            : (tilt>=4 ? '차체가 기울었다. 실측 명중률 89 %'
                       : '평평하다. 실측 명중률 100 %');
  }
  $('#r-tur').textContent=T(tm.turret_x,'°')+' / '+T(tm.turret_y,'°');
  $('#r-ko').textContent=(dv.known||[]).length+' 개';
  $('#r-do').textContent=T(dt.count)+' 개';

  const pm=g.p_hit_min,br2=g.body_rate_max,tf=g.tof_max;
  // 비과시간 게이트는 표적이 0.6 m/s 넘게 움직일 때만 적용된다.
  //   fire_control.py:  if (tof_max is not None and tracker.speed > 0.6 and ...)
  const tofActive = (tf!=null && (tk.speed||0) > 0.6);
  $('#gates').innerHTML=
    gateRow('탄도해',f.in_envelope===undefined?null:!!f.in_envelope,'')
  + gateRow('명중확률',pm==null?null:(f.p_hit!=null&&f.p_hit>=pm),
            (f.p_hit!=null?f.p_hit:'–')+'≥'+(pm==null?'—':pm))
  + gateRow('차체각속도',br2==null?null:true,br2==null?'제한없음':'≤'+br2)
  + gateRow('비과시간', !tofActive ? null : (f.tof!=null&&f.tof<=tf),
            (f.tof!=null?f.tof:'–')+'≤'+(tf==null?'—':tf)
            + (tofActive?'':'  정지표적 → 미적용'));

  $('#diag').innerHTML=diagnose(f, dv);
  $('#track').innerHTML=trackDiag(f);
  $('#drive').innerHTML=driveDiag(dv, tm);
  reconcileObstacles(dv);
  buildMarkers(); update3D(); drawMini();
}

/**
 * 플래너가 아는 장애물 목록과 대조해 이미 사라진 것을 지운다.
 *
 * pid_controller 의 set_obstacles() 는 목록을 통째로 교체하므로,
 * planner.obstacle_rectangles 가 현재 남아 있는 장애물의 정답이다.
 * 다만 시뮬레이터가 일부만 보내는 경우 오판할 수 있어,
 * 전체의 절반 이상이 들어왔을 때만 보정한다.
 */
function reconcileObstacles(dv){
  if(!RN||!RN.mapObjs) return;
  const kn=dv.known||[];
  const total=(ST&&ST.obstacles)?ST.obstacles.length:0;
  if(!total || kn.length < total*0.5) return;      // 부분 수신이면 건너뛴다
  const covered=(x,z,pad)=>kn.some(r=>
    x>=r.x0-pad && x<=r.x1+pad && z>=r.z0-pad && z<=r.z1+pad);
  RN.mapObjs.forEach(o=>{
    if(o.dead) return;
    if(!covered(o.x,o.z,1.5)){ o.dead=true;
      FALLING.push({obj:o.obj,t:0,dur:0.6,y0:o.obj.position.y,
                    tilt:(Math.random()-0.5)*1.4}); }
  });
  const T=RN.trees;
  if(T) T.rec.forEach(r=>{
    if(r.dead) return;
    if(!covered(r.x,r.z,1.2)){ r.dead=true;
      FALLING.push({tree:r,t:0,dur:0.5}); }
  });
}

/**
 * 기동표적 진단 — 목표 2 의 핵심 지표.
 *
 * 예측오차 ≈ k_err × 비과시간²  이다.
 * 비과시간이 제곱으로 들어가므로 거리를 좁히는 것이 가장 강력한 대응이다.
 */
function trackDiag(f){
  const tk=f.track||{}, sol=f.sol||{};
  const T=(v,u='')=>(v==null?'–':v+u);
  const moving=(tk.speed||0)>0.6;

  let html=`<div class="row"><span style="color:var(--acc);font-weight:600">표적 추적</span>
    <b>${moving?'<span class="pill fail">기동 중</span>'
               :'<span class="pill off">정지</span>'}</b></div>`;
  html+=`<div class="row"><span>표적 속도</span><b>${T(tk.speed,' m/s')}</b></div>`;
  if(moving){
    html+=`<div class="row"><span>진행 방위</span><b>${T(tk.heading,'°')}</b></div>`;
    html+=`<div class="row"><span>선회 각속도</span><b>${T(tk.omega,' °/s')}</b></div>`;
    html+=`<div class="row"><span>접선 가속</span><b>${T(tk.a_long,' m/s²')}</b></div>`;
  }
  html+=`<div class="row"><span>예측 채점</span><b>${T(tk.n_scored,' 회')}</b></div>`;

  // 예측 품질 — 이것이 명중률을 좌우한다
  if(tk.k_err!=null){
    const tof=f.tof||0;
    const pe=tk.pred_err;
    html+=`<div class="row"><span>k_err</span><b>${T(tk.k_err,' m/s²')}</b></div>`;
    if(pe!=null){
      const bad=pe>1.8;   // 표적 반폭
      html+=`<div class="row"><span>예상 예측오차</span>
        <b style="color:${bad?'var(--no)':'var(--ok)'}">${pe} m</b></div>`;
      html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--dim);
        line-height:1.55">${tk.k_err} × ${tof}² = <b style="color:var(--fg)">${pe} m</b>
        &nbsp;·&nbsp; 표적 반폭 1.8 m<br>`;
      if(bad && tof>0.6){
        // 반폭 안에 들어오려면 비과시간이 얼마여야 하나
        const tofNeed=Math.sqrt(1.8/Math.max(0.001,tk.k_err));
        html+=`<b style="color:var(--warn)">비과시간을 ${tofNeed.toFixed(2)} s 이하로 줄여야
          반폭 안에 들어온다.</b><br>지금 ${tof} s 다. 예측오차는 비과시간의
          <b>제곱</b>에 비례하므로 거리를 좁히는 것이 가장 효과적이다`;
      }else if(!bad){
        html+=`반폭 안에 들어온다. 예측은 문제없다`;
      }
      html+=`</div>`;
    }
  }
  if(tk.sig_lat!=null)
    html+=`<div class="row"><span>1σ 횡 / 종</span>
      <b>${T(tk.sig_lat)} / ${T(tk.sig_lon)} m</b></div>`;

  // 추적기가 죽어 있으면 그것부터 알려준다
  if(moving && (tk.n_scored||0)===0)
    html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--no);
      line-height:1.55">⚠ 표적이 움직이는데 예측 채점이 0 이다.
      추적기가 동작하지 않는다 — 리드 계산이 안 되므로 기동표적은 못 맞춘다</div>`;
  return html;
}

/**
 * 주행 진단 — 왜 저 경로로 가는지 답한다.
 *
 * 핵심: 전차는 '적'이 아니라 '목적지'를 향해 간다.
 *       목적지는 시뮬레이터가 /set_destination 으로 정해준다.
 *       그래서 적과 엉뚱한 방향으로 가는 것처럼 보일 수 있다.
 */
function driveDiag(dv, tm){
  if(!dv||!dv.available) return '';
  const T=(v,u='')=>(v==null?'–':v+u);
  let html=`<div class="row"><span style="color:var(--acc);font-weight:600">주행</span>
    <b>${dv.arrived?'<span class="pill pass">도착</span>'
                   :(dv.stop_flag?'<span class="pill off">정지</span>':'이동중')}</b></div>`;
  html+=`<div class="row"><span>목적지</span><b>${
    dv.dest?`(${dv.dest[0]}, ${dv.dest[1]})`:'없음'}</b></div>`;
  // 2026-08-26  목적지가 '전차가 기울지 않는 자리' 인가.
  //   기울면 명중률이 100 % -> 67 % 로 떨어진다 (8/26 27발 실측).
  //   터미널 [평지보정] 메시지가 /info 로그에 묻혀 안 보인다는
  //   지적이 있어 같은 정보를 여기에도 띄운다.
  if(dv.dest_tilt!=null){
    const t=dv.dest_tilt;
    const g = t<=2 ? ['적합','pass'] : t<=5 ? ['무난','pass']
            : t<=10 ? ['주의','off'] : ['부적합','fail'];
    html+=`<div class="row"><span>진지 기울기</span>
      <b><span style="color:var(--dim);margin-right:6px;font-size:10px">
      ${t.toFixed(1)}°</span><span class="pill ${g[1]}">${g[0]}</span></b></div>`;
  }
  html+=`<div class="row"><span>남은 직선거리</span><b>${T(dv.straight,' m')}</b></div>`;
  html+=`<div class="row"><span>계획 경로 길이</span><b>${T(dv.path_len,' m')}</b></div>`;
  if(dv.detour!=null){
    const bad=dv.detour>1.6;
    html+=`<div class="row"><span>우회율</span>
      <b style="color:${bad?'var(--no)':'var(--fg)'}">${dv.detour} 배</b></div>`;
    if(bad) html+=`<div style="padding:2px 13px 6px;font-size:10.5px;color:var(--warn);
      line-height:1.5">직선의 ${dv.detour}배로 돌아간다. 장애물 여유폭
      (clearance_radius 8 m)이 넓어 좁은 길을 피하는 것이다</div>`;
  }
  html+=`<div class="row"><span>목적지 변경</span><b>${T(dv.dest_n,' 회')}</b></div>`;
  html+=`<div class="row"><span>경로 재계산</span><b>${T(dv.replan_n,' 회')}</b></div>`;

  // 목적지와 적이 다른 방향이면 그걸 명시한다 — 가장 흔한 오해다
  if(dv.dest && tm && tm.my && tm.enemy){
    const me=[tm.my[0],tm.my[2]], en=[tm.enemy[0],tm.enemy[2]];
    const bDest=brg(me,dv.dest), bEn=brg(me,en);
    const gap=Math.abs(((bDest-bEn+540)%360)-180);
    const dEn=dist2(me,en);
    html+=`<div class="row"><span>목적지 / 적 방위차</span>
      <b style="color:${gap>45?'var(--no)':'var(--fg)'}">${gap.toFixed(0)}°</b></div>`;
    if(gap>45)
      html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--warn);
        line-height:1.55">목적지(${bDest.toFixed(0)}°)와 적(${bEn.toFixed(0)}°, ${dEn.toFixed(0)} m)이
        <b>${gap.toFixed(0)}° 다른 방향</b>이다.<br>
        전차는 적이 아니라 목적지를 향해 간다 — 시뮬레이터가
        /set_destination 으로 준 좌표다. 사격은 그 이동 중에 곁다리로 이뤄진다.</div>`;
  }
  if(dv.dests && dv.dests.length>1){
    html+=`<div style="padding:2px 13px 7px;font-size:10px;color:var(--dim);
      line-height:1.5">최근 목적지: ${dv.dests.map(d=>`(${d[0]},${d[1]})`).join(' → ')}</div>`;
  }
  return html;
}

/**
 * 사격 진단 — 지금 무엇이 발사를 막고 있는지 한 줄로 답한다.
 * 판단 순서는 fire_control 의 게이트 순서를 그대로 따른다.
 */
function diagnose(f, dv){
  const g=f.gates||{}, tm=f.tm||{}, tk=f.track||{}, A=f.aim||{};

  // FireControl.state 가 곧 게이트 판정 결과다. 그대로 해석한다.
  const st=(f.state||'').toUpperCase();
  const yE=Math.abs(A.yaw_err||0),   yD=A.yaw_db||0;
  const pE=Math.abs(A.pitch_err||0), pD=A.pitch_db||0;

  const D={
    FIRE:    ['발사','pass','모든 게이트 통과'],
    RELOAD:  ['재장전 중','off', (f.reload_left||0)+' s 남음'],
    SLEW:    ['포탑 정렬 중','fail',
      `조준오차 yaw ${yE.toFixed(2)}° / 허용 ${yD.toFixed(2)}°` +
      (yE>yD?' <b style="color:var(--no)">← 초과</b>':'') +
      ` &nbsp;·&nbsp; pitch ${pE.toFixed(2)}° / 허용 ${pD.toFixed(2)}°` +
      (pE>pD?' <b style="color:var(--no)">← 초과</b>':'') +
      (yD<0.3||pD<0.2 ? '<br>데드밴드가 매우 좁다. 거리가 멀수록 더 좁아진다 — '
        +'db_safety 를 올리면 완화된다' : '')],
    TOFCAP:  ['비과시간 초과','fail',
      `비행 ${f.tof} s > 상한 ${g.tof_max} s. 표적이 ${tk.speed} m/s 로 움직일 때만 걸리는 게이트다. `
      +'거리를 좁히거나 tof_max 를 0.9 로 올려야 한다'],
    TURNING: ['차체 선회 중','fail',
      `차체 각속도가 ${g.body_rate_max} °/s 를 넘었다. 멎기를 기다린다`],
    SETTLE:  ['차체 정지 대기','off','moving_fire 가 꺼져 있다'],
    HOLD:    ['명중확률 미달','fail',
      `p_hit ${f.p_hit} < 기준 ${g.p_hit_min}`
      +(tk.speed>1.0?` — 표적이 ${tk.speed} m/s 로 움직여 예측이 흔들린다`:'')
      +`<br>${g.patience} s 기다리면 기준이 ${g.p_hit_floor} 까지 내려간다`],
    NOTRACK: ['포탑 추적 한계','fail',
      `표적 각속도가 포탑보다 빠르다 (duty ${A.track_duty} > ${A.duty_max}). 거리를 벌려야 한다`],
    INHIBIT: ['사격 억제','fail','외부에서 발사를 막고 있다'],
    IDLE:    ['대기','off','탄도해가 없거나 표적이 없다'],
  };

  let why,cls,fix;
  if(D[st]){ why=D[st][0]; cls=D[st][1]; fix=D[st][2]; }
  else if(!tm.enemy){ why='적 위치 없음'; cls='fail'; fix='텔레메트리에 enemyPos 가 안 온다'; }
  else if(f.dist!=null && f.envelope && f.dist>f.envelope[1]){
    why='최대 사거리 밖'; cls='fail';
    fix=`적까지 ${f.dist} m — ${f.envelope[1]} m 안으로 접근해야 한다`; }
  else { why=st||'—'; cls='off'; fix=''; }

  // ── 시각 어긋남 — 이게 있으면 다른 진단은 의미가 없다 ──
  let html='';
  if(f.desync){
    html+=`<div style="margin:2px 9px 8px;padding:8px 10px;border-radius:7px;
      background:rgba(248,113,113,.14);border:1px solid rgba(248,113,113,.45)">
      <div style="color:var(--no);font-weight:800;font-size:11px;margin-bottom:4px">
        ⚠ 사격 모듈 시각 어긋남</div>
      <div style="font-size:10.5px;color:var(--dim);line-height:1.55">
        마지막 발사 시각이 <b style="color:var(--fg)">${f.desync.last_fire} s</b> 인데
        현재는 <b style="color:var(--fg)">${f.desync.now} s</b> 다
        (${f.desync.gap} s 미래).<br>
        에피소드를 Restart 해서 sim_time 이 되감겼는데
        FireControl.last_fire_t 가 이전 판 값을 들고 있다.<br>
        <b style="color:var(--no)">이 상태로는 한 발도 못 쏜다.</b></div>
      <button class="btn" style="margin-top:7px;width:100%" onclick="fixDesync()">
        시각 상태 초기화</button></div>`;
  }
  html+=`<div class="row"><span>사격 진단</span>
    <b><span class="pill ${cls}">${why}</span></b></div>`;
  if(fix) html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--dim);
    line-height:1.55">${fix}</div>`;
  // 조준 정렬 현황은 항상 보여준다
  if(A.yaw_db!==undefined){
    const bar=(e,d)=>{const r=d>0?Math.min(1,e/d):1;
      const c=e<=d?'var(--ok)':'var(--no)';
      return `<span style="display:inline-block;width:52px;height:4px;border-radius:2px;
        background:rgba(255,255,255,.12);vertical-align:middle;overflow:hidden">
        <i style="display:block;height:100%;width:${(r*100).toFixed(0)}%;background:${c}"></i></span>`;};
    html+=`<div class="row"><span>방위 정렬</span><b>${bar(yE,yD)}
      <span style="margin-left:6px">${yE.toFixed(2)} / ${yD.toFixed(2)}°</span></b></div>
      <div class="row"><span>앙각 정렬</span><b>${bar(pE,pD)}
      <span style="margin-left:6px">${pE.toFixed(2)} / ${pD.toFixed(2)}°</span></b></div>`;
  }

  // ── 탄착오차 분해 진단 ────────────────────────────
  // 조준 오차가 작은데 탄착이 벌어지면 원인은 데드밴드가 아니다.
  // 오차를 사선 기준 앞뒤/좌우로 쪼개야 범인이 갈린다.
  const sh=(f.shots||[]).filter(s=>s.imp);
  if(sh.length>=2){
    const avgOf=k=>{const v=sh.map(s=>s[k]).filter(x=>x!=null);
      return v.length? v.reduce((a,b)=>a+b,0)/v.length : null;};
    const absOf=k=>{const v=sh.map(s=>s[k]).filter(x=>x!=null).map(Math.abs);
      return v.length? v.reduce((a,b)=>a+b,0)/v.length : null;};
    const mAll=absOf('miss'), R=avgOf('rng_err'), C=avgOf('crs_err');
    const aR=absOf('rng_err'), aC=absOf('crs_err');
    const hits=sh.filter(s=>s.hit).length;
    const terr=sh.filter(s=>s.kind==='terrain').length;

    const sgn=(v,pos,neg)=>v==null?'':(v>0?pos:neg);
    html+=`<div class="row"><span>평균 탄착오차</span>
      <b>${mAll!=null?mAll.toFixed(1)+' m':'–'}</b></div>`;
    if(R!=null) html+=`<div class="row"><span>앞뒤 오차</span>
      <b style="color:${aR>aC?'var(--no)':'var(--fg)'}">${R>0?'+':''}${R.toFixed(1)} m
      <span style="font-size:10px;color:var(--dim)">${sgn(R,'길게','짧게')}</span></b></div>`;
    if(C!=null) html+=`<div class="row"><span>좌우 오차</span>
      <b style="color:${aC>aR?'var(--no)':'var(--fg)'}">${C>0?'+':''}${C.toFixed(1)} m
      <span style="font-size:10px;color:var(--dim)">${sgn(C,'오른쪽','왼쪽')}</span></b></div>`;

    if(hits===0 && aR!=null && aC!=null){
      const d=f.dist||1, sol=f.sol||{}, drdt=sol.drdt||6.8;
      let note;
      const aimTight = yE<=yD*1.5 && pE<=pD*1.5;
      if(aR > aC*2){
        // 앞뒤가 지배 → 앙각 계열
        const degEq=(aR/drdt);
        note=`<b style="color:var(--fg)">앞뒤 오차가 좌우의 ${(aR/Math.max(0.01,aC)).toFixed(1)}배</b>다. `
          +`앙각 계열 문제다.<br>거리오차 ${aR.toFixed(1)} m ÷ drdt ${drdt} = 앙각 `
          +`<b style="color:var(--fg)">${degEq.toFixed(2)}°</b> 어긋난 셈이다.<br>`
          +(aimTight
            ? `조준 오차는 ${pE.toFixed(2)}° 뿐인데 탄착은 ${degEq.toFixed(2)}° 만큼 벌어졌다 — `
              +`<b style="color:var(--warn)">데드밴드가 아니라 탄도해 자체가 틀렸다.</b> `
              +`차체 피치 ${tm.body_y}° 가 앙각 기준에 반영되지 않았을 가능성이 크다`
            : `조준이 아직 안 잡혔다. 정렬을 먼저 확인할 것`)
          + (R<0 ? '<br>일관되게 <b>짧게</b> 떨어진다 — 앙각을 올려야 한다'
                 : '<br>일관되게 <b>길게</b> 떨어진다 — 앙각을 내려야 한다');
      }else if(aC > aR*2){
        const degEq=Math.atan2(aC,d)*180/Math.PI;
        note=`<b style="color:var(--fg)">좌우 오차가 앞뒤의 ${(aC/Math.max(0.01,aR)).toFixed(1)}배</b>다. `
          +`방위 계열 문제다.<br>${aC.toFixed(1)} m / ${d} m = 방위 `
          +`<b style="color:var(--fg)">${degEq.toFixed(2)}°</b> 어긋난 셈이다.<br>`
          +`차체 롤 ${tm.body_z}° 가 포탑 yaw 를 월드 기준에서 틀어놓고 있을 수 있다`;
      }else{
        note=`앞뒤 ${aR.toFixed(1)} m · 좌우 ${aC.toFixed(1)} m 로 비슷하다. `
          +`차체 자세 전반(피치 ${tm.body_y}° / 롤 ${tm.body_z}°)이 원인일 가능성이 크다`;
      }
      html+=`<div style="padding:3px 13px 7px;font-size:10.5px;color:var(--dim);
        line-height:1.55">${sh.length}발 중 명중 ${hits} · 지면착탄 ${terr}<br>${note}</div>`;
    }
  }
  return html;
}

// ── 마커 ──
let mkSig='';
function buildMarkers(){
  const f=S.fire||{},tm=f.tm||{},dv=S.drive||{},dt=S.detect||{};
  const it=[];
  const push=(c,n,s,x,z)=>it.push({c,n,s,x,z});
  if(tm.my)   push('#4b9cf5','아군 기갑',
    `(${tm.my[0].toFixed(0)}, ${tm.my[2].toFixed(0)})  ▲${hAt(tm.my[0],tm.my[2]).toFixed(0)}m`,
    tm.my[0],tm.my[2]);
  if(tm.enemy)push('#ef4444','적 기갑',
    `(${tm.enemy[0].toFixed(0)}, ${tm.enemy[2].toFixed(0)})  ${tm.my?
      dist2([tm.my[0],tm.my[2]],[tm.enemy[0],tm.enemy[2]]).toFixed(0)+'m':''}`,
    tm.enemy[0],tm.enemy[2]);
  if(dv.dest) push('#f5a524','목적지',
    `(${dv.dest[0].toFixed(0)}, ${dv.dest[1].toFixed(0)})  ${dv.arrived?'도착':'이동중'}`,
    dv.dest[0],dv.dest[1]);
  (dt.objects||[]).slice(0,10).forEach(o=>{
    const x=o.x!=null?o.x:o.posX, z=o.z!=null?o.z:o.posZ;
    if(x==null||z==null)return;
    push('#eab308',(o.className||o.name||'미상'),
      `(${(+x).toFixed(0)}, ${(+z).toFixed(0)})  탐지`,+x,+z);});
  const sig=it.map(i=>i.n+i.s).join('|'); if(sig===mkSig)return; mkSig=sig;
  $('#mk-n').textContent=it.length;
  const b=$('#markers'); b.innerHTML='';
  it.forEach(i=>{const d=document.createElement('div'); d.className='mk';
    d.innerHTML=`<u style="width:9px;height:9px;border-radius:2px;background:${i.c};
      display:inline-block;flex:none"></u><div><b>${i.n}</b><span>${i.s}</span></div>`;
    d.onclick=()=>{if(!RN)return; const s=W2S(i.x,i.z);
      RN.cam3.tgt.set(s[0],hAt(i.x,i.z)*VS,s[1]);
      RN.cam3.dis=Math.min(RN.cam3.dis,110); follow=false; syncBtn();};
    b.appendChild(d);});
}

// ── 사선 차폐 ──
function losBlocked(a,b){
  if(!TR)return false;
  const ha=hAt(a[0],a[1])+2.5,hb=hAt(b[0],b[1])+2.5,N=72;
  for(let i=1;i<N;i++){const t=i/N;
    if(hAt(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)>ha+(hb-ha)*t+0.7)return true;}
  return false;
}

// ── 3D 갱신 ──
let lastShot=-1;
function update3D(){
  if(!RN||!S||!S.fire)return;
  const f=S.fire,tm=f.tm||{},dv=S.drive||{};

  if(tm.my){
    const p=V(tm.my[0],tm.my[2],0);
    const shakeMe=HITFX.some(h=>h.tank===RN.me);
    if(shakeMe) HITFX.forEach(h=>{if(h.tank===RN.me) h.base.copy(p);});
    else RN.me.position.copy(p);
    RN.me.rotation.order='YXZ';
    RN.me.rotation.y=yawOf(tm.body_x);                     // 차체 방위
    RN.me.rotation.z=(tm.body_z||0)*Math.PI/180;           // 롤
    RN.me.rotation.x=(tm.body_y||0)*Math.PI/180;           // 피치
    // playerTurretX 는 월드 절대 방위다(차체각 포함).
    // 포탑은 차체의 자식이므로 차체 회전을 상쇄해야 실제 방위를 향한다.
    RN.me.userData.turret.rotation.y = yawOf(tm.turret_x) - yawOf(tm.body_x);
    RN.me.userData.gun.rotation.x = -(tm.turret_y||0)*Math.PI/180;   // 앙각(+가 위)
    RN.meTag.position.copy(p).setY(p.y+12);
    // 포신 지향선 — 실제로 어디를 겨누는지 눈으로 확인한다
    if(RN.aimRay){
      const th=(tm.turret_x||0)*Math.PI/180, el=(tm.turret_y||0)*Math.PI/180;
      const R0=dist2([tm.my[0],tm.my[2]],
        tm.enemy?[tm.enemy[0],tm.enemy[2]]:[tm.my[0],tm.my[2]+60])||60;
      const pts=[];
      for(let k=0;k<=16;k++){
        const d=R0*1.15*k/16;
        const x=tm.my[0]+Math.sin(th)*d, z=tm.my[2]+Math.cos(th)*d;
        const w=W2S(x,z);
        pts.push(new THREE.Vector3(w[0], p.y+3.0+Math.tan(el)*d*VS*0.35, w[1]));
      }
      RN.aimRay.geometry.setFromPoints(pts);
      RN.aimRay.computeLineDistances();     // 점선 재계산 (필수)
    }
  }
  if(tm.enemy){
    const p=V(tm.enemy[0],tm.enemy[2],0);
    // 피격 점멸 중에는 흔들림이 덮어쓰지 않도록 기준 위치만 갱신한다.
    const shaking=HITFX.some(h=>h.tank===RN.foe);
    if(shaking) HITFX.forEach(h=>{if(h.tank===RN.foe) h.base.copy(p);});
    else RN.foe.position.copy(p);
    RN.foe.rotation.y=yawOf(tm.enemy_body_x);
    // 적 포탑은 우리 쪽을 향하게 둔다 (적 포탑각은 텔레메트리에 없다)
    if(tm.my){
      const bk=brg([tm.enemy[0],tm.enemy[2]],[tm.my[0],tm.my[2]]);
      RN.foe.userData.turret.rotation.y=yawOf(bk)-yawOf(tm.enemy_body_x);
    }
    RN.foeTag.position.copy(p).setY(p.y+12);
  }
  // 조준점
  if(f.aim3){ RN.aimSp.visible=L.aim;
    RN.aimSp.position.copy(V(f.aim3[0],f.aim3[2],3.4)); }
  else RN.aimSp.visible=false;

  // 경로 — 지형에 밀착
  if(dv.path&&dv.path.length>1){
    const pts=[];
    for(let i=0;i<dv.path.length-1;i++){
      const a=dv.path[i],b=dv.path[i+1],seg=Math.max(2,Math.round(dist2(a,b)/3));
      for(let k=0;k<=seg;k++){const t=k/seg;
        pts.push(V(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,0.9));}}
    RN.routeL.geometry.setFromPoints(pts);
  }
  // 이동 궤적
  const tr=f.trail||{};
  [['my',RN.trailMe],['enemy',RN.trailFoe]].forEach(([k,ln])=>{
    const q=tr[k]||[]; if(q.length>1)
      ln.geometry.setFromPoints(q.map(p=>V(p[0],p[1],0.7)));});

  // 사선
  if(tm.my&&tm.enemy){
    const a=[tm.my[0],tm.my[2]],e=[tm.enemy[0],tm.enemy[2]];
    const bl=losBlocked(a,e);
    RN.losL.material.color.setHex(bl?0xf87171:0x4ade80);
    RN.losL.geometry.setFromPoints([V(a[0],a[1],2.6),V(e[0],e[1],2.6)]);
  }
  // 사거리 링
  if(tm.my&&f.envelope){
    const ring=r=>{const p=[];
      for(let d=0;d<=360;d+=3){const t=d*Math.PI/180;
        const x=tm.my[0]+Math.sin(t)*r,z=tm.my[2]+Math.cos(t)*r;
        if(x<0||x>SPAN||z<0||z>SPAN)continue; p.push(V(x,z,0.6));}
      return p;};
    RN.ringMin.geometry.setFromPoints(ring(f.envelope[0]));
    RN.ringMax.geometry.setFromPoints(ring(f.envelope[1]));
    if(f.suggest) RN.ringSug.geometry.setFromPoints(ring(f.suggest));
  }
  // ── 추정 적 전차 — 크기·맵대조로 찾아낸 것. 항상 표시한다. ──
  const foes=(dv.known||[]).filter(o=>o.g);
  if(RN.foeG.children.length!==foes.length){
    RN.foeG.clear();
    foes.forEach(()=>{
      const grp=new THREE.Group();
      const box=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),
        new THREE.MeshBasicMaterial({color:0xff2b21,wireframe:true,
          transparent:true,opacity:.95}));
      const ring=new THREE.Mesh(new THREE.RingGeometry(4.0,4.9,40),
        new THREE.MeshBasicMaterial({color:0xff2b21,transparent:true,
          opacity:.55,side:THREE.DoubleSide,depthWrite:false}));
      ring.rotation.x=-Math.PI/2;
      const sp=new THREE.Sprite(new THREE.SpriteMaterial(
        {map:new THREE.CanvasTexture(tagCanvas('적 전차 (추정)','#ff2b21')),
         depthTest:false,transparent:true}));
      const im=sp.material.map.image;
      sp.scale.set(im.width*0.052,im.height*0.052,1); sp.renderOrder=22;
      grp.add(box); grp.add(ring); grp.add(sp);
      grp.userData={box,ring,sp};
      RN.foeG.add(grp);
    });
  }
  foes.forEach((o,i)=>{
    const g2=RN.foeG.children[i]; if(!g2)return;
    const p=V(o.cx,o.cz,0);
    const w=Math.max(2,o.x1-o.x0), dd=Math.max(2,o.z1-o.z0);
    g2.userData.box.scale.set(w,3.0,dd);
    g2.userData.box.position.copy(p).setY(p.y+1.6);
    g2.userData.ring.position.copy(p).setY(p.y+0.25);
    g2.userData.sp.position.copy(p).setY(p.y+13);
    g2.userData.ring.material.opacity=0.35+Math.sin(performance.now()*0.004)*0.22;
  });

  // 인지 장애물
  const kn=dv.known||[];
  if(RN.knownG.children.length!==kn.length){
    RN.knownG.clear();
    const C={enemy_tank:0xef4444,enemy:0xf97316,team_tank:0x4b9cf5,
             team:0x60a5fa,unknown:0xeab308,nature:0x5b6b7d};
    // 자연물은 수가 많아 아주 흐리게, 위협 대상만 또렷하게 그린다.
    const OP={enemy_tank:.85,enemy:.75,unknown:.60,team_tank:.6,team:.5,nature:.14};
    kn.forEach(o=>{
      const w=Math.max(1,o.x1-o.x0),d2=Math.max(1,o.z1-o.z0);
      const m=new THREE.Mesh(new THREE.BoxGeometry(w,3.2,d2),
        new THREE.MeshBasicMaterial({color:C[o.t]||0x5b6b7d,wireframe:true,
          transparent:true,opacity:OP[o.t]!==undefined?OP[o.t]:.3,
          depthWrite:false}));
      const cx=(o.x0+o.x1)/2,cz=(o.z0+o.z1)/2;
      m.position.copy(V(cx,cz,1.8)); RN.knownG.add(m);});
  }
  // 탐지 객체
  const objs=(S.detect.objects||[]).filter(o=>o.x!=null||o.posX!=null);
  if(RN.detG.children.length!==objs.length){
    RN.detG.clear();
    objs.forEach(()=>{const m=new THREE.Mesh(new THREE.OctahedronGeometry(1.6),
      new THREE.MeshBasicMaterial({color:0xeab308,wireframe:true}));
      RN.detG.add(m);});
  }
  objs.forEach((o,i)=>{const x=+(o.x!=null?o.x:o.posX),z=+(o.z!=null?o.z:o.posZ);
    if(RN.detG.children[i]) RN.detG.children[i].position.copy(V(x,z,4));});

  // ── 탄도 · 탄착 ──
  const shots=(f.shots||[]).filter(s=>s.fire&&s.imp);
  if(RN.shotG.children.length!==shots.length*2){
    RN.shotG.clear();
    shots.forEach(s=>{
      const arc=new THREE.Line(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({color:s.hit?0x4ade80:0xa78bfa,
          transparent:true,opacity:.55}));
      const mk=new THREE.Mesh(new THREE.SphereGeometry(s.hit?1.5:1.0,10,8),
        new THREE.MeshBasicMaterial({color:s.hit?0x4ade80:0xfbbf24,
          transparent:true,opacity:.85}));
      RN.shotG.add(arc); RN.shotG.add(mk);
    });
  }
  shots.forEach((s,i)=>{
    const arc=RN.shotG.children[i*2], mk=RN.shotG.children[i*2+1];
    if(!arc||!mk)return;
    const a=s.fire,b=s.imp,N=28,pts=[];
    const h0=hAt(a[0],a[1])*VS+2.6, h1=hAt(b[0],b[1])*VS+0.5;
    const lift=Math.min(28,dist2(a,b)*0.22);
    for(let k=0;k<=N;k++){const t=k/N;
      const x=a[0]+(b[0]-a[0])*t, z=a[1]+(b[1]-a[1])*t;
      const y=h0+(h1-h0)*t + lift*Math.sin(Math.PI*t);
      const w=W2S(x,z);
      pts.push(new THREE.Vector3(w[0],y,w[1]));}
    arc.geometry.setFromPoints(pts);
    mk.position.copy(V(b[0],b[1],1.0));
  });
  // ── 적 사격 (2026-09-09) ──
  //   9/09 46번 판에서 적이 8발을 쐈는데 3D 화면에 하나도 안 나왔다.
  //   이 상황도는 아군 FireModule 만 읽는데, 적 사격 기록은 5100 번
  //   서버에만 있었기 때문이다. 이제 _foe_poll() 이 가져온다.
  //
  //   색으로 가른다.  아군 초록/보라  <->  적 주황
  const fs=((f.foe||{}).shots||[]).filter(s=>s.fire&&s.imp);
  if(RN.foeShotG.children.length!==fs.length*2){
    RN.foeShotG.clear();
    fs.forEach(s=>{
      const arc=new THREE.Line(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({color:s.hit?0xf97316:0x9a3412,
          transparent:true,opacity:.5}));
      const mk=new THREE.Mesh(new THREE.SphereGeometry(s.hit?1.5:1.0,10,8),
        new THREE.MeshBasicMaterial({color:s.hit?0xf97316:0xea580c,
          transparent:true,opacity:.8}));
      RN.foeShotG.add(arc); RN.foeShotG.add(mk);
    });
  }
  fs.forEach((s,i)=>{
    const arc=RN.foeShotG.children[i*2], mk=RN.foeShotG.children[i*2+1];
    if(!arc||!mk)return;
    const a=s.fire,b=s.imp,N=28,pts=[];
    const h0=hAt(a[0],a[1])*VS+2.6, h1=hAt(b[0],b[1])*VS+0.5;
    const lift=Math.min(28,dist2(a,b)*0.22);
    for(let k=0;k<=N;k++){const t=k/N;
      const x=a[0]+(b[0]-a[0])*t, z=a[1]+(b[1]-a[1])*t;
      const y=h0+(h1-h0)*t + lift*Math.sin(Math.PI*t);
      const w=W2S(x,z);
      pts.push(new THREE.Vector3(w[0],y,w[1]));}
    arc.geometry.setFromPoints(pts);
    mk.position.copy(V(b[0],b[1],1.0));
  });

  // 새 사격 → 포탄 발사 (비행 + 탄착 폭발)
  if(shots.length){
    if(lastShot<0){ lastShot=shots[0].id; }        // 첫 폴링은 과거분 무시
    else if(shots[0].id!==lastShot){
      // 마지막으로 본 것 이후의 새 사격을 오래된 것부터 재생
      const fresh=[];
      for(const s of shots){ if(s.id===lastShot) break; fresh.push(s); }
      fresh.reverse().forEach((s,k)=>setTimeout(()=>spawnShell(s),k*180));
      lastShot=shots[0].id;
    }
  }
}

// ── 나침반 ──
function drawCompass(yaw){
  const c=$('#comp'),g=c.getContext('2d'),S2=116,R=44;
  g.clearRect(0,0,S2,S2); g.save(); g.translate(S2/2,S2/2);
  g.fillStyle='rgba(11,16,24,.85)';g.beginPath();g.arc(0,0,R+7,0,7);g.fill();
  g.strokeStyle='rgba(120,160,210,.22)';g.lineWidth=1.5;g.stroke();
  g.rotate(yaw);
  for(let a=0;a<360;a+=15){const t=a*Math.PI/180,big=a%90===0;
    g.strokeStyle=big?'#38bdf8':'rgba(120,160,210,.3)';g.lineWidth=big?2.4:1.2;
    g.beginPath();g.moveTo(Math.sin(t)*(R-(big?12:6)),-Math.cos(t)*(R-(big?12:6)));
    g.lineTo(Math.sin(t)*R,-Math.cos(t)*R);g.stroke();}
  g.fillStyle='#ef4444';g.beginPath();
  g.moveTo(0,-R+1);g.lineTo(-7,-R+16);g.lineTo(7,-R+16);g.closePath();g.fill();
  g.restore();
  g.fillStyle='#e6edf7';g.font='800 12px Consolas';g.textAlign='center';
  g.fillText('N',S2/2,S2/2-R+30);
}

// ── 축소 지도 ──
// 2026-08-26  제목줄을 눌러 접었다 펼 수 있다. 기본은 접힘.
//   210 x 210 캔버스가 우측 사격 패널을 통째로 가려서 필요할 때만 편다.
function toggleMini(){
  const m=$('#mini'); m.classList.toggle('fold');
  $('#mmt').textContent = m.classList.contains('fold') ? '펴기' : '접기';
  if(!m.classList.contains('fold')) sizeMini();
}
function sizeMini(){const c=$('#mmc'),r=c.getBoundingClientRect();
  if(!r.width) return;
  c.width=r.width*devicePixelRatio;c.height=r.height*devicePixelRatio;drawMini();}
function drawMini(){
  const c=$('#mmc'); if(!c.width||!TR)return;
  if($('#mini').classList.contains('fold')) return;   // 접혀 있으면 그리지 않는다
  const g=c.getContext('2d'),P=Math.min(c.width,c.height);
  const X=v=>v/SPAN*P, Z=v=>P-(v/SPAN*P);
  g.clearRect(0,0,c.width,c.height);
  if(!drawMini._bg){          // 지형 배경은 한 번만 그린다
    const n=TR.n, off=document.createElement('canvas'); off.width=off.height=n;
    const og=off.getContext('2d'), im=og.createImageData(n,n);
    const rng=(TR.hi-TR.lo)||1, wl=waterLevel();
    for(let j=0;j<n;j++)for(let i=0;i<n;i++){
      const k=idx(i,j),h=TR.H[k],u=(h-TR.lo)/rng,q=(j*n+i)*4;
      let r,gg,b;
      if(h<wl+0.6){r=42;gg=92;b=124;}
      else if(u<0.4){r=64;gg=94;b=54;}
      else if(u<0.7){r=112;gg=116;b=68;}
      else {r=150;gg=140;b=118;}
      im.data[q]=r;im.data[q+1]=gg;im.data[q+2]=b;im.data[q+3]=255;}
    og.putImageData(im,0,0); drawMini._bg=off;
  }
  g.imageSmoothingEnabled=true;
  g.drawImage(drawMini._bg,0,0,P,P);
  const D=devicePixelRatio, f=(S&&S.fire)||{}, tm=f.tm||{}, dv=(S&&S.drive)||{};
  if(dv.path&&dv.path.length>1){g.strokeStyle='#f5a524';g.lineWidth=1.8*D;
    g.beginPath();dv.path.forEach((p,i)=>i?g.lineTo(X(p[0]),Z(p[1])):g.moveTo(X(p[0]),Z(p[1])));
    g.stroke();}
  const tr=f.trail||{};
  [['my','rgba(75,156,245,.55)'],['enemy','rgba(239,68,68,.55)']].forEach(([k,c2])=>{
    const q=tr[k]||[]; if(q.length<2)return; g.strokeStyle=c2;g.lineWidth=1.2*D;
    g.beginPath();q.forEach((p,i)=>i?g.lineTo(X(p[0]),Z(p[1])):g.moveTo(X(p[0]),Z(p[1])));
    g.stroke();});
  if(tm.my&&f.envelope){g.setLineDash([4*D,4*D]);g.strokeStyle='rgba(56,189,248,.5)';
    g.lineWidth=1*D;g.beginPath();g.arc(X(tm.my[0]),Z(tm.my[2]),f.envelope[1]/SPAN*P,0,7);
    g.stroke();g.setLineDash([]);}
  const dot=(x,z,c2,r)=>{g.fillStyle=c2;g.beginPath();g.arc(X(x),Z(z),r*D,0,7);g.fill();
    g.strokeStyle='rgba(0,0,0,.6)';g.lineWidth=1*D;g.stroke();};
  if(dv.dest) dot(dv.dest[0],dv.dest[1],'#f5a524',3);
  if(tm.enemy)dot(tm.enemy[0],tm.enemy[2],'#ef4444',4);
  if(tm.my){dot(tm.my[0],tm.my[2],'#4b9cf5',4);
    // 차체 방위(가는 선) + 포신 방위(굵은 선)
    const bx=(tm.body_x||0)*Math.PI/180, tx=(tm.turret_x||0)*Math.PI/180;
    const ray=(ang,len,col,w)=>{g.strokeStyle=col;g.lineWidth=w*D;g.beginPath();
      g.moveTo(X(tm.my[0]),Z(tm.my[2]));
      g.lineTo(X(tm.my[0])+Math.sin(ang)*len*D,Z(tm.my[2])-Math.cos(ang)*len*D);g.stroke();};
    ray(bx,11,'rgba(142,197,255,.55)',1.2);
    ray(tx,18,'#9fe8ff',2.0);}
}
</script></body></html>
"""
