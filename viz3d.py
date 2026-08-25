# -*- coding: utf-8 -*-
"""
viz3d.py - 실시간 3D 전장 뷰 (통합 데모용)

설계 원칙: 각 팀 코드에 손대지 않는다.
    이 모듈은 세 팀의 객체를 '읽기만' 한다.
    사격 판정, 경로 계획, 객체 인식에 어떤 영향도 주지 않는다.
    상태 수집은 브라우저가 요청할 때만 일어나므로
    /get_action 주기에 부하를 얹지도 않는다.

읽는 대상
    사격팀  fm.tm / fm.fc / fm.log          위치·포탑·조준점·탄착
    경로팀  drive_controller.path           D* Lite 계획 경로
    인식팀  tskijun.DETECTED_OBJECTS_INFO   탐지 객체와 위협도

붙이는 방법 (ally-controller.py 맨 아래 두 줄)
    from viz3d import attach_viz
    attach_viz(app, fm=fm, drive=drive_controller, detect=tskijun)

    그 다음 브라우저에서  http://<서버IP>:5000/view3d

지형 데이터
    move/risk_layers.npz 를 시작 시 한 번 읽어 압축 후 페이지에 심는다.
    경로팀이 이미 쓰고 있는 파일이므로 별도 준비가 필요 없다.
"""
import base64
import json
import math
import os
import zlib

from flask import Response, jsonify

# ══════════════════════════════════════════════════════════
RISK_NPZ = "move/risk_layers.npz"    # 경로팀이 쓰는 파일 그대로
HEIGHTMAP = "move/heightmap_filled.npy"   # 없으면 npz 의 height 사용
CELL = 2.0                 # 고도맵 셀 크기 [m]
MAP_SPAN = 300.0           # 맵 한 변 [m]
TRAIL_MAX = 400
SHOT_MAX = 40


TANK_SPOTS = [(120.0, 180.0), (210.0, 95.0)]  # 지정 좌표에서 5 m 안에 있으면서 전차 크기 범위에 맞는 장애물을 찾아 그립니다. 
# 두 조건을 함께 걸어야 파괴 후 옆 바위에 달라붙지 않아요.


# /info 의 playerPos 가 아군, enemyPos 가 적이라는 것이 기본 전제다.
# 시뮬레이터 설정이나 서버 배치에 따라 반대로 들어오는 경우가 있으므로
# 화면에서 뒤바뀌어 보이면 이 값을 True 로 바꾼다.
SWAP_SIDES = False

# 지형 배열의 축 순서.
# risk_layers.npz 의 height 는 [x][z] 순서로 저장되어 있고,
# 경로팀 플래너도 flipud + rot90(-1) (= 전치) 을 적용해 쓴다.
# 이 변환을 하지 않으면 지형만 90도 돌아가 전차 위치가 어긋나 보인다.
# 검증 결과 risk_layers.npz 의 height 는 arr[z][x] 순서다.
#   미니맵에서 강은 오른쪽(좁은 x)에 세로로(넓은 z) 흐른다.
#   arr[z][x] 로 조회 시 강 구역 수면비율 18%, arr[x][z] 는 1%.
# 따라서 전치하지 않는다.
TERRAIN_TRANSPOSE = False

# 축 방향. 화면이 거울처럼 뒤집혀 보이면 해당 축을 True 로 바꾼다.
# 맵 모서리 표식(/view3d 의 '좌표 표식')으로 확인할 수 있다.
TERRAIN_FLIP_X = True
TERRAIN_FLIP_Z = False

# 물 표현. 이 고도 이하를 수면으로 그린다.
WATER_LEVEL = 5.5          # [m]  

# ── 위협 반경 ───────────────────────────────────────────
# 적 전차의 사거리를 붉은 원으로 표시한다.
# 실측 교전 포락 상한(약 130 m)을 기본으로 하되,
# 안쪽일수록 진하고 가장자리로 갈수록 투명해지도록 그린다.
THREAT_RADIUS = 130.0      # 위협 반경 [m]
THREAT_CORE = 30.0         # 이 거리 안은 최대 농도 (근거리 고명중 구간)

# 장애물 중 적 전차로 분류된 것만 3D 에 전차로 그린다.
# ── 장애물 전차 지정 ────────────────────────────────────
# ObstacleRect.from_min_max 는 미리 등록된 좌표 목록과 대조해 타입을 정한다.
# 목록에 없는 전차는 전부 'nature' 로 떨어져 타입으로는 구분할 수 없다.
# 크기로 거르는 방법도 시험했으나 회전한 전차의 바운딩박스가
# 중형 바위와 겹쳐(6.8 x 6.8) 오탐이 많았다.
#
# 그래서 좌표를 직접 지정한다. 맵에 배치한 전차 위치를 넣으면 된다.
# 가장 가까운 장애물을 찾아 그 크기로 그리므로 대략의 좌표면 충분하다.
#
#   TANK_SPOTS = [(120.0, 180.0), (210.0, 95.0)]
TANK_SPOTS = []
TANK_SNAP = 5.0            # 지정 좌표에서 이 거리 안의 장애물을 전차로 본다 [m]
                           # 너무 크게 잡으면 파괴 후 옆 바위에 달라붙는다.
TANK_SPOT_FALLBACK = False # True 면 장애물이 없어도 지정 좌표에 그린다.
                           # 파괴돼도 사라지지 않으므로 기본은 False.

# 좌표를 모를 때: 타입이 아래에 해당하면 크기와 무관하게 전차로 본다.
# 경로팀이 from_min_max 목록에 전차를 등록하면 이쪽으로 잡힌다.
ENEMY_TANK_TYPES = ("enemy_tank",)
# 파괴 판정: 장애물 목록에서 사라지면 제거된 것으로 본다.
# ══════════════════════════════════════════════════════════

_TERRAIN = {"ok": False}


def _pack(a, dtype):
    return base64.b64encode(zlib.compress(a.astype(dtype).tobytes(), 9)).decode()


def _load_terrain():
    """시작 시 1회. 실패해도 뷰어는 평지로 동작한다."""
    out = {"ok": False, "w": 150, "h": 150, "cell": CELL,
           "lo": 0.0, "hi": 1.0, "hm": "", "sl": "", "ex": "", "bl": "",
           "water": None, "fx": TERRAIN_FLIP_X, "fz": TERRAIN_FLIP_Z}
    try:
        import numpy as np
    except ImportError:
        print("[viz3d] numpy 없음 - 지형 없이 동작")
        return out

    hm = None
    if os.path.exists(HEIGHTMAP):
        hm = np.load(HEIGHTMAP).astype("float32")

    if not os.path.exists(RISK_NPZ):
        print(f"[viz3d] {RISK_NPZ} 없음 - 지형 없이 동작")
        return out

    try:
        d = np.load(RISK_NPZ)
        if hm is None and "height" in d:
            hm = d["height"].astype("float32")
        if hm is None:
            print("[viz3d] 고도 데이터 없음")
            return out

        # 위험 레이어는 300x300 으로 업샘플되어 저장되어 있다.
        # 고도맵 격자에 맞춰 되돌린다.
        H, W = hm.shape

        def fit(x, how="mean"):
            if x is None or x.shape == (H, W):
                return x
            f = x.shape[0] // H
            if f >= 1 and x.shape[0] == H * f:
                r = x.reshape(H, f, W, f)
                return r.any(axis=(1, 3)) if how == "any" else r.mean(axis=(1, 3))
            return None

        if hm.shape[0] > 200:      # height 자체가 업샘플된 경우
            f = hm.shape[0] // 150
            if f > 1:
                hm = hm.reshape(150, f, 150, f).mean(axis=(1, 3))
                H, W = hm.shape

        # 축 정렬
        if TERRAIN_TRANSPOSE:
            hm = hm.T
        if TERRAIN_FLIP_Z:
            hm = hm[::-1, :]
        if TERRAIN_FLIP_X:
            hm = hm[:, ::-1]

        lo, hi = float(np.nanmin(hm)), float(np.nanmax(hm))
        hm = np.nan_to_num(hm, nan=lo)
        out.update({"ok": True, "h": H, "w": W, "lo": lo, "hi": hi,
                    "water": WATER_LEVEL,
                    "fx": TERRAIN_FLIP_X, "fz": TERRAIN_FLIP_Z,
                    "hm": _pack((hm - lo) / max(1e-6, hi - lo) * 65535, "<u2")})

        def tr(a):
            if TERRAIN_TRANSPOSE:
                a = a.T
            if TERRAIN_FLIP_Z:
                a = a[::-1, :]
            if TERRAIN_FLIP_X:
                a = a[:, ::-1]
            return a
        sl = fit(tr(d["slope_cost"])) if "slope_cost" in d else None
        ex = fit(tr(d["exposure"])) if "exposure" in d else None
        bl = fit(tr(d["blocked"]), "any") if "blocked" in d else None
        if sl is not None:
            out["sl"] = _pack(np.clip(sl, 0, 1) * 255, "u1")
        if ex is not None:
            out["ex"] = _pack(np.clip(ex, 0, 1) * 255, "u1")
        if bl is not None:
            out["bl"] = _pack(bl, "u1")
        print(f"[viz3d] 지형 {H}x{W} {lo:.1f}~{hi:.1f} m + 위험 레이어 적재")
    except Exception as e:
        print(f"[viz3d] 지형 적재 실패: {e}")
    return out


class _Trail:
    def __init__(self):
        self.my, self.enemy, self._t = [], [], None

    def push(self, t, my, enemy):
        if t is None or (self._t is not None and t - self._t < 0.25):
            return
        self._t = t
        if my:
            self.my.append([round(my[0], 1), round(my[2], 1)])
            del self.my[:-TRAIL_MAX]
        if enemy:
            self.enemy.append([round(enemy[0], 1), round(enemy[2], 1)])
            del self.enemy[:-TRAIL_MAX]


_trail = _Trail()


def _shots(fm):
    """최근 사격. 발사점과 착탄점을 넘겨 브라우저가 포물선을 그린다."""
    out = []
    try:
        for r in fm.log.records[:SHOT_MAX]:
            f, i = r.get("fire_pos"), r.get("impact")
            if not f or not i:
                continue
            out.append({"f": [round(v, 1) for v in f],
                        "i": [round(v, 1) for v in i],
                        "hit": r.get("kind") == "tank",
                        "d": r.get("dist"), "zone": r.get("zone")})
    except Exception:
        pass
    return out


def _path(drive):
    """D* Lite 계획 경로. 경로팀이 관리한다."""
    try:
        p = getattr(drive, "path", None) or []
        return [[round(float(w[0]), 1), round(float(w[1]), 1)] for w in p][:600]
    except Exception:
        return []


def _obstacle_tanks(drive):
    """
    /update_obstacle 로 들어온 장애물 중 적 전차만 뽑는다.

    .map 파일을 읽을 필요가 없다. 시뮬레이터가 장애물 목록을 보내주고
    경로팀 플래너가 obstacle_rectangles 에 이미 보관하고 있다.
    파괴되면 시뮬레이터가 목록에서 빼주므로 자연히 화면에서도 사라진다.
    """
    out = []
    try:
        planner = (getattr(drive, "planner", None)
                   or getattr(drive, "path_planner", None))
        rects = getattr(planner, "obstacle_rectangles", None) or []

        def info(r):
            w = abs(float(r.x_max) - float(r.x_min))
            l = abs(float(r.z_max) - float(r.z_min))
            return {"x": round((float(r.x_min) + float(r.x_max)) * 0.5, 1),
                    "z": round((float(r.z_min) + float(r.z_max)) * 0.5, 1),
                    "w": round(w, 1), "l": round(l, 1),
                    "type": getattr(r, "type", "nature")}

        seen = set()

        # 1) 타입으로 잡히는 것
        for r in rects:
            if getattr(r, "type", "nature") in ENEMY_TANK_TYPES:
                d = info(r)
                seen.add((d["x"], d["z"]))
                out.append(d)

        # 2) 지정 좌표에 가장 가까운 장애물
        #    거리뿐 아니라 크기도 전차 범위여야 한다.
        #    거리만 보면 전차가 파괴된 뒤 옆 바위에 달라붙는다.
        for sx, sz in TANK_SPOTS:
            best, bd = None, TANK_SNAP
            for r in rects:
                d = info(r)
                lo_d, hi_d = min(d["w"], d["l"]), max(d["w"], d["l"])
                if not (2.6 <= lo_d <= 7.4 and 5.2 <= hi_d <= 7.6):
                    continue
                dd = math.hypot(d["x"] - sx, d["z"] - sz)
                if dd < bd:
                    best, bd = d, dd
            if best is not None and (best["x"], best["z"]) not in seen:
                seen.add((best["x"], best["z"]))
                out.append(best)
            elif best is None and TANK_SPOT_FALLBACK:
                # 장애물 목록에 없어도 지정 좌표에 그린다 (파괴돼도 남는다)
                out.append({"x": round(sx, 1), "z": round(sz, 1),
                            "w": 3.3, "l": 6.3, "type": "spot"})
    except Exception:
        pass
    return out


def _obstacle_types(drive):
    """장애물 분류 집계. 적 전차가 안 보일 때 원인을 찾기 위한 진단."""
    try:
        planner = (getattr(drive, "planner", None)
                   or getattr(drive, "path_planner", None))
        rects = getattr(planner, "obstacle_rectangles", None) or []
        c = {}
        for r in rects:
            c[getattr(r, "type", "?")] = c.get(getattr(r, "type", "?"), 0) + 1
        return c
    except Exception:
        return {}


def _obstacle_sizes(drive, top=14):
    """
    장애물 크기 분포. 전차 식별 임계값을 정하기 위한 진단이다.
    (짧은변 x 긴변) 을 0.5 m 단위로 묶어 개수를 센다.
    """
    try:
        planner = (getattr(drive, "planner", None)
                   or getattr(drive, "path_planner", None))
        rects = getattr(planner, "obstacle_rectangles", None) or []
        c = {}
        for r in rects:
            w = abs(float(r.x_max) - float(r.x_min))
            l = abs(float(r.z_max) - float(r.z_min))
            k = "%.1f x %.1f" % (round(min(w, l) * 2) / 2,
                                 round(max(w, l) * 2) / 2)
            c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1])[:top])
    except Exception:
        return {}


def _near_spots(drive, radius=20.0):
    """
    TANK_SPOTS 주변 장애물 목록. 좌표를 정할 때 참고용이다.
    지정 좌표가 비어 있으면 큰 장애물 상위 몇 개를 보여준다.
    """
    try:
        planner = (getattr(drive, "planner", None)
                   or getattr(drive, "path_planner", None))
        rects = getattr(planner, "obstacle_rectangles", None) or []
        items = []
        for r in rects:
            w = abs(float(r.x_max) - float(r.x_min))
            l = abs(float(r.z_max) - float(r.z_min))
            items.append({"x": round((float(r.x_min) + float(r.x_max)) / 2, 1),
                          "z": round((float(r.z_min) + float(r.z_max)) / 2, 1),
                          "w": round(w, 1), "l": round(l, 1)})
        if TANK_SPOTS:
            out = []
            for sx, sz in TANK_SPOTS:
                near = sorted(items,
                              key=lambda d: math.hypot(d["x"] - sx, d["z"] - sz))
                out.append({"spot": [sx, sz], "near": near[:4]})
            return out
        # 지정이 없으면 큰 것 위주로 보여준다 (전차 후보 찾기)
        big = sorted(items, key=lambda d: -(d["w"] * d["l"]))
        return big[:12]
    except Exception:
        return []


def _detected(detect_mod):
    """
    인식팀 탐지 객체.
    위협도 순으로 정렬되어 있으면 그대로, 아니면 원본 순서.
    """
    if detect_mod is None:
        return []
    out = []
    try:
        for o in (getattr(detect_mod, "DETECTED_OBJECTS_INFO", None) or [])[:40]:
            # (x, y, z, class_name) 튜플 또는 dict 두 형태를 모두 받는다
            if isinstance(o, dict):
                wp = o.get("world_pos") or {}
                out.append({"x": round(float(wp.get("x", 0)), 1),
                            "y": round(float(wp.get("y", 0)), 1),
                            "z": round(float(wp.get("z", 0)), 1),
                            "c": o.get("class_name", "?"),
                            "th": round(float(o.get("threat_score", 0)), 3),
                            "d": o.get("distance")})
            elif isinstance(o, (list, tuple)) and len(o) >= 4:
                out.append({"x": round(float(o[0]), 1),
                            "y": round(float(o[1]), 1),
                            "z": round(float(o[2]), 1),
                            "c": str(o[3]), "th": 0.0, "d": None})
    except Exception:
        pass
    return out


def _snapshot(fm, drive, detect_mod):
    g = lambda o, k, d=None: getattr(o, k, d) if o is not None else d
    tm = g(fm, "tm")
    my, en = g(tm, "my"), g(tm, "enemy")
    mhp, ehp = g(tm, "my_hp"), g(tm, "enemy_hp")
    mbody, ebody = g(tm, "body_x", 0.0), g(tm, "enemy_body_x", 0.0)
    mspd, espd = g(tm, "my_speed", 0.0), g(tm, "enemy_speed", 0.0)
    if SWAP_SIDES:
        my, en = en, my
        mhp, ehp = ehp, mhp
        mbody, ebody = ebody, mbody
        mspd, espd = espd, mspd
    _trail.push(g(tm, "t"), my, en)

    sol = g(g(fm, "fc"), "last_solution")
    aim = None
    if sol is not None and getattr(sol, "valid", False):
        ap = getattr(sol, "aim_point", None)
        if ap:
            aim = [round(v, 1) for v in ap]

    st = {}
    try:
        st = fm.status()
    except Exception:
        pass

    dest = g(drive, "destination")
    return {
        "t": g(tm, "t"),
        "my": [round(v, 2) for v in my] if my else None,
        "enemy": [round(v, 2) for v in en] if en else None,
        "body": round(mbody, 1),
        "turret": round(g(tm, "turret_x", 0.0), 1),
        "pitch": round(g(tm, "turret_y", 0.0), 2),
        "hull": [round(g(tm, "body_y", 0.0), 1), round(g(tm, "body_z", 0.0), 1)],
        "enemy_body": round(ebody, 1),
        "my_hp": mhp, "enemy_hp": ehp,
        "my_spd": round(mspd, 1),
        "en_spd": round(espd, 1),
        "swapped": SWAP_SIDES,
        "dist": st.get("dist"),
        "fc": st.get("state", "-"),
        "aim": aim,
        "p_hit": st.get("p_hit"),
        "reload": st.get("reload_left"),
        "fired": st.get("fired", 0), "hits": st.get("hits", 0),
        "rate": st.get("hit_rate", 0.0),
        "envelope": st.get("envelope"),
        "suggest": (fm.suggest_range() if hasattr(fm, "suggest_range") else None),
        "path": _path(drive),
        "dest": ([round(float(dest[0]), 1), round(float(dest[2]), 1)]
                 if dest else None),
        "objects": _detected(detect_mod),
        "threats": _obstacle_tanks(drive),
        "obs_types": _obstacle_types(drive),
        "obs_sizes": _obstacle_sizes(drive),
        "near_spots": _near_spots(drive),
        "threat_r": THREAT_RADIUS, "threat_core": THREAT_CORE,
        "trail": {"my": _trail.my[-200:], "enemy": _trail.enemy[-200:]},
        "shots": _shots(fm),
    }


def attach_viz(app, fm, drive=None, detect=None,
               route="/view3d", state_route="/state3d"):
    """
    통합 서버에 3D 뷰를 붙인다.

    fm      FireModule
    drive   TankDriveController  (경로 표시용, 없어도 동작)
    detect  인식팀 모듈           (탐지 객체 표시용, 없어도 동작)
    """
    global _TERRAIN
    _TERRAIN = _load_terrain()
    page = (_PAGE.replace("__TERRAIN__", json.dumps(_TERRAIN))
                 .replace("__SPAN__", str(MAP_SPAN))
                 .replace("__STATE__", state_route))

    @app.route(route)
    def _view3d():
        return Response(page, mimetype="text/html")

    @app.route(state_route)
    def _state3d():
        try:
            return jsonify(_snapshot(fm, drive, detect))
        except Exception as e:
            return jsonify({"error": str(e)})

    print(f"[viz3d] 3D 뷰 준비 완료 -> http://localhost:5000{route}")


_PAGE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>전장 3D 실시간</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;overflow:hidden;
font-family:-apple-system,'Segoe UI',Roboto,'Malgun Gothic',sans-serif}
#cv{display:block;width:100vw;height:100vh}
.panel{position:fixed;background:rgba(22,27,34,.93);border:1px solid #30363d;
border-radius:10px;padding:12px 15px;backdrop-filter:blur(6px)}
#ui{top:12px;left:12px;width:238px}
#hud{top:12px;right:12px;min-width:196px;font-size:12px;
font-variant-numeric:tabular-nums}
#leg{bottom:12px;left:12px;font-size:11px}
h1{font-size:14px;margin-bottom:3px}
.sub{font-size:11px;color:#8b949e;margin-bottom:10px}
label{font-size:12px;color:#8b949e;display:block;margin:8px 0 3px}
select,input[type=range]{width:100%}
select{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;
border-radius:6px;padding:5px 7px;font-size:12px;font-family:inherit}
input[type=range]{accent-color:#1f6feb}
.val{float:right;color:#58a6ff}
.chk{display:flex;align-items:center;gap:7px;font-size:12px;margin:6px 0;
cursor:pointer;color:#c9d1d9}
.r{display:flex;justify-content:space-between;padding:2px 0}
.r b{color:#58a6ff;font-weight:600}
.hp{height:5px;border-radius:3px;background:#30363d;margin:3px 0 7px}
.hp>i{display:block;height:100%;border-radius:3px;transition:width .3s}
#dot{display:inline-block;width:7px;height:7px;border-radius:50%;
background:#3fb950;margin-right:5px}
#dot.off{background:#f85149}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;
margin-right:5px;vertical-align:-1px}
hr{border:0;border-top:1px solid #30363d;margin:7px 0}
</style></head><body>
<canvas id="cv"></canvas>

<div class="panel" id="ui">
  <h1><span id="dot"></span>전장 3D 실시간</h1>
  <div class="sub" id="conn">연결 중…</div>
  <div class="sub" id="swapmsg" style="color:#d29922;display:none">
    진영 교체 적용 중 (SWAP_SIDES)</div>
  <label>표면</label>
  <select id="mode">
    <option value="elev">고도</option>
    <option value="expo">피탐도</option>
    <option value="slope">경사 저항</option>
    <option value="block">통행 가능</option>
  </select>
  <label>수직 과장 <span class="val" id="vexV">3.0×</span></label>
  <input type="range" id="vex" min="1" max="8" step="0.5" value="3">
  <label>갱신 주기 <span class="val" id="hzV">0.5s</span></label>
  <input type="range" id="hz" min="2" max="20" step="1" value="5">
  <label class="chk"><input type="checkbox" id="follow"> 아군 추적</label>
  <button id="reset" style="width:100%;margin-top:8px;background:#21262d;
    color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px;
    font-size:12px;font-family:inherit;cursor:pointer">시점 초기화</button>
  <label class="chk"><input type="checkbox" id="pathC" checked> 계획 경로</label>
  <label class="chk"><input type="checkbox" id="trail" checked> 이동 궤적</label>
  <label class="chk"><input type="checkbox" id="shots" checked> 탄도 궤적</label>
  <label class="chk"><input type="checkbox" id="objs" checked> 탐지 객체</label>
  <label class="chk"><input type="checkbox" id="threat" checked> 위협 반경</label>
  <label class="chk"><input type="checkbox" id="water" checked> 수면</label>
  <label class="chk"><input type="checkbox" id="axis"> 좌표 표식</label>
  <label class="chk"><input type="checkbox" id="los" checked> 조준선</label>
</div>

<div class="panel" id="hud">
  <div class="r"><span>아군 HP</span><b id="mhp">-</b></div>
  <div class="hp"><i id="mhpb" style="width:100%;background:#3fb950"></i></div>
  <div class="r"><span>적 HP</span><b id="ehp">-</b></div>
  <div class="hp"><i id="ehpb" style="width:100%;background:#f85149"></i></div>
  <div class="r"><span>거리</span><b id="dist">-</b></div>
  <div class="r"><span>권장 거리</span><b id="sug">-</b></div>
  <div class="r"><span>사격통제</span><b id="fc">-</b></div>
  <div class="r"><span>명중확률</span><b id="ph">-</b></div>
  <div class="r"><span>재장전</span><b id="rl">-</b></div>
  <hr>
  <div class="r"><span>사격 / 명중</span><b id="sc">-</b></div>
  <div class="r"><span>명중률</span><b id="hr">-</b></div>
  <div class="r"><span>경로 점</span><b id="pl">-</b></div>
  <div class="r"><span>탐지 객체</span><b id="ob">-</b></div>
  <div class="r"><span>적 전차(장애물)</span><b id="tk">-</b></div>
  <div class="r"><span>sim_t</span><b id="st">-</b></div>
</div>

<div class="panel" id="leg">
  <span class="sw" style="background:#58a6ff"></span>아군 &nbsp;
  <span class="sw" style="background:#f85149"></span>적 &nbsp;
  <span class="sw" style="background:#a371f7"></span>계획경로 &nbsp;
  <span class="sw" style="background:#d1495b"></span>적전차(장애물) &nbsp;
  <span class="sw" style="background:#2f6f9f"></span>수면 &nbsp;
  <span class="sw" style="background:#3fb950"></span>명중 &nbsp;
  <span class="sw" style="background:#8b949e"></span>빗나감<br>
  <span style="color:#6e7681">드래그 회전 · 휠 확대 · 우클릭 이동<br>
  시점은 맵 중앙 고정. 전차를 따라가려면 '아군 추적'</span>
</div>

<script type="importmap">
{"imports":{"three":"https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js"}}
</script>
<script type="module">
import * as THREE from 'three';
const T = __TERRAIN__, SPAN = __SPAN__, SURL = "__STATE__";

function inflate(b64, Out){
  if(!b64) return Promise.resolve(null);
  const bin=atob(b64), buf=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) buf[i]=bin.charCodeAt(i);
  return new Response(new Blob([buf]).stream()
    .pipeThrough(new DecompressionStream('deflate')))
    .arrayBuffer().then(ab=>new Out(ab));
}
let W=T.w,H=T.h,CELL=T.cell,height=null,expo=null,slope=null,blocked=null;
Promise.all([inflate(T.hm,Uint16Array),inflate(T.ex,Uint8Array),
             inflate(T.sl,Uint8Array),inflate(T.bl,Uint8Array)])
.then(([h,e,s,b])=>{
  height=new Float32Array(W*H);
  if(h) for(let i=0;i<W*H;i++) height[i]=T.lo+(h[i]/65535)*(T.hi-T.lo);
  expo=e;slope=s;blocked=b; init(); poll();
});

const RAMPS={
 elev:[[0,[38,70,52]],[.35,[92,120,64]],[.6,[168,152,96]],[.82,[150,120,92]],[1,[236,236,240]]],
 slope:[[0,[40,70,110]],[.5,[210,190,80]],[1,[200,60,50]]],
 expo:[[0,[24,60,48]],[.4,[70,150,110]],[.75,[230,190,70]],[1,[220,60,55]]]};
const L=(a,b,t)=>a+(b-a)*t;
function ramp(st,t){t=Math.max(0,Math.min(1,t));
 for(let i=0;i<st.length-1;i++){const[p0,c0]=st[i],[p1,c1]=st[i+1];
  if(t>=p0&&t<=p1){const k=(t-p0)/(p1-p0||1);
   return[L(c0[0],c1[0],k),L(c0[1],c1[1],k),L(c0[2],c1[2],k)];}}
 return st[st.length-1][1];}

let scene,cam,rnd,geo,mesh,gMy,gEn,gAim,gLos,gTrailMy,gTrailEn,gShots,gPath,gDest,gObjs;
let gThreat,gObsTank;
let threatKey='';        // 위협 원은 대상이 바뀔 때만 다시 만든다 (매 프레임 재생성 방지)
// 회전 중심을 전차에 묶으면 시점이 계속 흔들려 지형 파악이 어렵다.
// 기본은 맵 중앙 고정이고, 필요할 때만 '아군 추적'을 켠다.
let vex=3,mode='elev',yaw=-0.7,pitch=.6,dist=380,panX=0,panZ=0,follow=false,last=null;
const idx=(x,z)=>z*W+x;

// 지형 배열을 뒤집었다면 개체(전차·경로·탄도)의 좌표도 같은 방식으로
// 뒤집어야 지형 위 올바른 자리에 놓인다. 한쪽만 뒤집으면 어긋난다.
function fx(wx){ return T.fx ? (SPAN - wx) : wx; }
function fz(wz){ return T.fz ? (SPAN - wz) : wz; }

function hAt(wx,wz){ if(!height) return 0;
  const gx=Math.max(0,Math.min(W-1,Math.round(fx(wx)/CELL)));
  const gz=Math.max(0,Math.min(H-1,Math.round(fz(wz)/CELL)));
  return height[idx(gx,gz)]; }

const toScene=(x,y,z)=>new THREE.Vector3(fx(x)-SPAN/2, y*vex, fz(z)-SPAN/2);

function label(text,color){
  // 어느 쪽이 아군인지 화면에서 바로 확인할 수 있게 표찰을 붙인다.
  const c=document.createElement('canvas'); c.width=256; c.height=64;
  const x=c.getContext('2d');
  x.fillStyle='rgba(13,17,23,.75)'; x.fillRect(0,0,256,64);
  x.strokeStyle=color; x.lineWidth=4; x.strokeRect(2,2,252,60);
  x.fillStyle=color; x.font='bold 38px sans-serif';
  x.textAlign='center'; x.textBaseline='middle'; x.fillText(text,128,34);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({
    map:new THREE.CanvasTexture(c),depthTest:false,transparent:true}));
  sp.scale.set(16,4,1); sp.position.y=9; sp.renderOrder=999;
  return sp;
}

function tank(color,name){
  const g=new THREE.Group();
  const b=new THREE.Mesh(new THREE.BoxGeometry(3.6,1.6,7.5),
    new THREE.MeshStandardMaterial({color,roughness:.7}));
  b.position.y=1.0; g.add(b);
  const tur=new THREE.Group(); tur.name='tur';
  const t=new THREE.Mesh(new THREE.BoxGeometry(2.6,1.1,3.4),
    new THREE.MeshStandardMaterial({color,roughness:.6}));
  t.position.y=2.4; tur.add(t);
  const bar=new THREE.Mesh(new THREE.CylinderGeometry(.16,.16,5.5,8),
    new THREE.MeshStandardMaterial({color:0x8b949e}));
  bar.rotation.x=Math.PI/2; bar.position.set(0,2.5,2.6); tur.add(bar);
  g.add(tur);
  const ring=new THREE.Mesh(new THREE.RingGeometry(4,4.6,32),
    new THREE.MeshBasicMaterial({color,transparent:true,opacity:.45,side:THREE.DoubleSide}));
  ring.rotation.x=-Math.PI/2; ring.position.y=.1; g.add(ring);
  if(name) g.add(label(name,'#'+new THREE.Color(color).getHexString()));
  return g;
}
function mkLine(c,o){ const g=new THREE.Line(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({color:c,transparent:true,opacity:o||.55}));
  scene.add(g); return g; }

function init(){
  rnd=new THREE.WebGLRenderer({canvas:document.getElementById('cv'),antialias:true});
  rnd.setPixelRatio(Math.min(devicePixelRatio,2)); rnd.setSize(innerWidth,innerHeight);
  scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d1117);
  scene.fog=new THREE.Fog(0x0d1117,520,1200);
  cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,1,3000);
  geo=new THREE.PlaneGeometry(SPAN,SPAN,W-1,H-1); geo.rotateX(-Math.PI/2);
  geo.setAttribute('color',new THREE.BufferAttribute(
    new Float32Array(geo.attributes.position.count*3),3));
  applyH(); applyC();
  mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({
    vertexColors:true,roughness:.95,metalness:.02})); scene.add(mesh);
  buildWater();
  scene.add(new THREE.AmbientLight(0xffffff,.5));
  const sun=new THREE.DirectionalLight(0xfff2e0,1.1);
  sun.position.set(200,300,150); scene.add(sun);
  gMy=tank(0x58a6ff,'아군'); gEn=tank(0xf85149,'적');
  scene.add(gMy); scene.add(gEn);
  gAim=new THREE.Mesh(new THREE.SphereGeometry(1.6,12,12),
    new THREE.MeshBasicMaterial({color:0xffd33d})); scene.add(gAim);
  gLos=new THREE.Line(new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(),new THREE.Vector3()]),
    new THREE.LineDashedMaterial({color:0xffd33d,dashSize:3,gapSize:2,
    transparent:true,opacity:.7})); scene.add(gLos);
  gTrailMy=mkLine(0x58a6ff); gTrailEn=mkLine(0xf85149);
  gPath=mkLine(0xa371f7,.9);
  gDest=new THREE.Mesh(new THREE.ConeGeometry(2.4,6,8),
    new THREE.MeshBasicMaterial({color:0xa371f7})); gDest.rotation.x=Math.PI;
  scene.add(gDest);
  gShots=new THREE.Group(); scene.add(gShots);
  gObjs=new THREE.Group(); scene.add(gObjs);
  gThreat=new THREE.Group(); scene.add(gThreat);
  buildAxis();
  gObsTank=new THREE.Group(); scene.add(gObsTank);
  bind(); animate();
}
let gWater=null;
function buildWater(){
  // 수면 고도 이하 지역에만 반투명 수면을 깐다.
  // 전체를 덮는 평면 대신 해당 셀만 사각형으로 만들어 지형을 가리지 않는다.
  if(gWater){scene.remove(gWater);gWater=null;}
  const lvl=T.water; if(lvl==null||!height) return;
  const pos=[], idxs=[]; let n=0;
  const half=CELL*0.5;
  for(let z=0;z<H;z++) for(let x=0;x<W;x++){
    if(height[idx(x,z)]>lvl) continue;
    // 격자 인덱스는 이미 뒤집힌 배열의 것이므로 그대로 화면 좌표로 쓴다.
    // toScene 을 거치면 두 번 뒤집혀 어긋난다.
    const wx=x*CELL, wz=z*CELL, y=lvl*vex;
    const a=[wx-half,wz-half],b=[wx+half,wz-half],
          c=[wx+half,wz+half],d=[wx-half,wz+half];
    [a,b,c,d].forEach(q=>pos.push(q[0]-SPAN/2, y, q[1]-SPAN/2));
    idxs.push(n,n+2,n+1, n,n+3,n+2); n+=4;
  }
  if(!pos.length) return;
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  g.setIndex(idxs); g.computeVertexNormals();
  gWater=new THREE.Mesh(g,new THREE.MeshStandardMaterial({
    color:0x2f6f9f, transparent:true, opacity:0.62,
    roughness:0.15, metalness:0.35, depthWrite:false,
    side:THREE.DoubleSide}));
  scene.add(gWater);
}

function applyH(){ const p=geo.attributes.position;
  for(let z=0;z<H;z++) for(let x=0;x<W;x++) p.setY(z*W+x,(height?height[idx(x,z)]:0)*vex);
  p.needsUpdate=true; geo.computeVertexNormals(); }
function applyC(){ const c=geo.attributes.color;
  for(let z=0;z<H;z++) for(let x=0;x<W;x++){ const i=idx(x,z); let v;
    if(mode==='elev'){
      const hv=height?height[i]:0;
      v = (T.water!=null && hv<=T.water) ? [64,86,96]      // 수면 아래 바닥
        : ramp(RAMPS.elev,height?(hv-T.lo)/(T.hi-T.lo):0);
    }
    else if(mode==='expo') v=expo?ramp(RAMPS.expo,Math.min(1,(expo[i]/255)/.55)):[60,80,70];
    else if(mode==='slope') v=slope?ramp(RAMPS.slope,slope[i]/255):[60,80,70];
    else v=(blocked&&blocked[i])?[200,60,50]:[70,110,80];
    c.setXYZ(i,v[0]/255,v[1]/255,v[2]/255);} c.needsUpdate=true; }
function setLine(o,pts,on,lift){
  if(!on||!pts||pts.length<2){o.visible=false;return;}
  o.visible=true;
  o.geometry.setFromPoints(pts.map(p=>toScene(p[0],hAt(p[0],p[1])+(lift||1.2),p[1])));
}
function arc(f,i,hit){
  const pts=[],n=24,d=Math.hypot(i[0]-f[0],i[2]-f[2]),rise=Math.max(2,d*0.10);
  for(let k=0;k<=n;k++){const t=k/n;
    pts.push(toScene(L(f[0],i[0],t),L(f[1]+2.6,i[1],t)+Math.sin(Math.PI*t)*rise,L(f[2],i[2],t)));}
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({color:hit?0x3fb950:0x8b949e,
    transparent:true,opacity:hit?.85:.35}));
}
function threatRing(cx,cz,R,core){
  // 채워진 원은 경사면에서 무늬가 생기고 지형을 가린다.
  // 가장자리 테두리만 그리고, 안쪽은 아주 옅은 띠 몇 개로 표현한다.
  // 맵 밖으로 나가는 각도는 잘라낸다.
  const g=new THREE.Group();
  const bands=[[1.0,0.55],[0.66,0.20],[0.33,0.12]];   // [반경비율, 불투명도]
  bands.forEach(([k,op])=>{
    const rr=R*k, pts=[];
    let run=[];
    for(let a=0;a<=180;a++){
      const th=2*Math.PI*a/180;
      const wx=cx+Math.cos(th)*rr, wz=cz+Math.sin(th)*rr;
      if(wx<0||wx>SPAN||wz<0||wz>SPAN){          // 맵 밖 - 선을 끊는다
        if(run.length>1) pts.push(run);
        run=[]; continue;
      }
      run.push(toScene(wx,hAt(wx,wz)+0.5,wz));
    }
    if(run.length>1) pts.push(run);
    pts.forEach(seg=>{
      g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(seg),
        new THREE.LineBasicMaterial({color:0xf85149,
          transparent:true, opacity:op, depthWrite:false})));
    });
  });
  // 유효 사격권(core) 은 실선으로 한 번 더 강조
  const seg=[];
  for(let a=0;a<=180;a++){
    const th=2*Math.PI*a/180;
    const wx=cx+Math.cos(th)*core, wz=cz+Math.sin(th)*core;
    if(wx<0||wx>SPAN||wz<0||wz>SPAN) continue;
    seg.push(toScene(wx,hAt(wx,wz)+0.5,wz));
  }
  if(seg.length>1)
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(seg),
      new THREE.LineBasicMaterial({color:0xff7b72,transparent:true,
        opacity:0.85,depthWrite:false})));
  return g;
}

function obsTank(o){
  // 장애물로 등록된 적 전차. 실제 치수(w x l)로 그린다.
  const g=new THREE.Group();
  const b=new THREE.Mesh(new THREE.BoxGeometry(o.w||3.3,1.8,o.l||6.3),
    new THREE.MeshStandardMaterial({color:0xd1495b,roughness:.75}));
  b.position.y=1.1; g.add(b);
  const t=new THREE.Mesh(new THREE.BoxGeometry(2.4,1.0,3.0),
    new THREE.MeshStandardMaterial({color:0xb03a4b,roughness:.7}));
  t.position.y=2.5; g.add(t);
  g.add(label('적 전차','#f85149'));
  g.position.copy(toScene(o.x,hAt(o.x,o.z),o.z));
  return g;
}

function marker(o){
  // 위협도가 높을수록 붉고 크게
  const th=Math.max(0,Math.min(1,o.th||0));
  const col=new THREE.Color().setHSL(L(.33,0,th),.75,.55);
  const g=new THREE.Group();
  const m=new THREE.Mesh(new THREE.OctahedronGeometry(1.4+th*2.2),
    new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.8}));
  m.position.copy(toScene(o.x,hAt(o.x,o.z)+3.0,o.z)); g.add(m);
  const pole=new THREE.Line(new THREE.BufferGeometry().setFromPoints(
    [toScene(o.x,hAt(o.x,o.z),o.z),toScene(o.x,hAt(o.x,o.z)+3.0,o.z)]),
    new THREE.LineBasicMaterial({color:col,transparent:true,opacity:.5}));
  g.add(pole); return g;
}
let gAxis=null;
function buildAxis(){
  // 좌표 방향을 눈으로 확인하기 위한 표식.
  // 시뮬레이터 미니맵과 대조해 축이 맞는지 판단할 수 있다.
  if(gAxis){scene.remove(gAxis);gAxis=null;}
  gAxis=new THREE.Group();
  const pts=[[0,0,'x0 z0'],[SPAN,0,'x300 z0'],
             [0,SPAN,'x0 z300'],[SPAN,SPAN,'x300 z300']];
  pts.forEach(([x,z,txt])=>{
    const p=toScene(x,hAt(Math.min(x,SPAN-1),Math.min(z,SPAN-1))+10,z);
    const pole=new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      [toScene(x,hAt(Math.min(x,SPAN-1),Math.min(z,SPAN-1)),z),p]),
      new THREE.LineBasicMaterial({color:0x7d8590}));
    gAxis.add(pole);
    const l=label(txt,'#8b949e'); l.position.copy(p); l.scale.set(22,5.5,1);
    gAxis.add(l);
  });
  // 아군 스폰 (60, 27.23) - /init 설정값
  const sp=toScene(60,hAt(60,27)+6,27.23);
  const m=new THREE.Mesh(new THREE.ConeGeometry(2.2,6,4),
    new THREE.MeshBasicMaterial({color:0x58a6ff,transparent:true,opacity:.7}));
  m.position.copy(sp); gAxis.add(m);
  const sl=label('아군 스폰','#58a6ff'); sl.position.copy(sp);
  sl.position.y+=6; sl.scale.set(20,5,1); gAxis.add(sl);
  // 최고봉
  if(height){
    let bi=0; for(let i=1;i<W*H;i++) if(height[i]>height[bi]) bi=i;
    // 격자 인덱스 -> world 좌표. 배열이 뒤집혀 있으면 되돌려야
    // toScene 을 거쳤을 때 제자리에 온다.
    let bx=(bi%W)*CELL, bz=Math.floor(bi/W)*CELL;
    if(T.fx) bx=SPAN-bx;
    if(T.fz) bz=SPAN-bz;
    const pk=toScene(bx,height[bi]+8,bz);
    const pl=label('최고봉 '+height[bi].toFixed(0)+'m','#e3b341');
    pl.position.copy(pk); pl.scale.set(26,6,1); gAxis.add(pl);
  }
  gAxis.visible=false; scene.add(gAxis);
}

function bind(){
  const cv=document.getElementById('cv'); let drag=null,px=0,py=0;
  cv.addEventListener('pointerdown',e=>{drag=e.button===2?'pan':'rot';
    px=e.clientX;py=e.clientY;cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointerup',()=>drag=null);
  cv.addEventListener('contextmenu',e=>e.preventDefault());
  cv.addEventListener('pointermove',e=>{
    if(drag==='rot'){yaw-=(e.clientX-px)*.005;
      pitch=Math.max(.08,Math.min(1.5,pitch+(e.clientY-py)*.004));}
    else if(drag==='pan'){document.getElementById('follow').checked=false;follow=false;
      panX-=(e.clientX-px)*dist*.0016*Math.cos(yaw);
      panZ-=(e.clientX-px)*dist*.0016*Math.sin(yaw);
      panX+=(e.clientY-py)*dist*.0016*Math.sin(yaw);
      panZ-=(e.clientY-py)*dist*.0016*Math.cos(yaw);}
    px=e.clientX;py=e.clientY;});
  cv.addEventListener('wheel',e=>{e.preventDefault();
    dist=Math.max(60,Math.min(1400,dist*(1+Math.sign(e.deltaY)*.1)));},{passive:false});
  document.getElementById('mode').onchange=e=>{mode=e.target.value;applyC();};
  document.getElementById('vex').oninput=e=>{vex=+e.target.value;
    document.getElementById('vexV').textContent=vex.toFixed(1)+'×';
    applyH(); buildWater(); buildAxis();
    gAxis.visible=document.getElementById('axis').checked;
    if(last) draw(last);};
  document.getElementById('hz').oninput=e=>{
    document.getElementById('hzV').textContent=(e.target.value/10).toFixed(1)+'s';};
  document.getElementById('axis').onchange=e=>{ if(gAxis) gAxis.visible=e.target.checked; };
  document.getElementById('water').onchange=e=>{ if(gWater) gWater.visible=e.target.checked; };
  document.getElementById('follow').onchange=e=>follow=e.target.checked;
  document.getElementById('reset').onclick=()=>{
    document.getElementById('follow').checked=false; follow=false;
    panX=0; panZ=0; yaw=-0.7; pitch=.6; dist=380;
  };
  addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;
    cam.updateProjectionMatrix();rnd.setSize(innerWidth,innerHeight);});
}
async function poll(){
  const hz=+document.getElementById('hz').value;
  try{ const r=await fetch(SURL,{cache:'no-store'}); const s=await r.json();
    if(!s.error){last=s;draw(s);ok(true);} else ok(false,s.error);
  }catch(e){ ok(false,'서버 응답 없음'); }
  setTimeout(poll,hz*100);
}
function ok(good,msg){
  document.getElementById('dot').className=good?'':'off';
  document.getElementById('conn').textContent=good
    ?('갱신 중'+(last&&last.t!=null?(' · sim_t '+last.t.toFixed(1)):''))
    :(msg||'끊김');
}
// 축을 뒤집으면 방위각도 그에 맞춰 반사시켜야 포신이 옳은 쪽을 향한다.
//   x 반전: yaw -> -yaw        (동서가 바뀜)
//   z 반전: yaw -> 180 - yaw   (남북이 바뀜)
function fyaw(deg){
  let a=deg;
  if(T.fx) a=-a;
  if(T.fz) a=180-a;
  return a;
}

function draw(s){
  const g=id=>document.getElementById(id);
  if(s.my){ gMy.position.copy(toScene(s.my[0],hAt(s.my[0],s.my[2]),s.my[2]));
    gMy.visible=true; gMy.rotation.y=-fyaw(s.body)*Math.PI/180;
    gMy.getObjectByName('tur').rotation.y=
      -(fyaw(s.turret)-fyaw(s.body))*Math.PI/180;
    // Sprite 는 항상 카메라를 향하므로 차체 회전과 무관하다
  } else gMy.visible=false;
  if(s.enemy){ gEn.position.copy(toScene(s.enemy[0],hAt(s.enemy[0],s.enemy[2]),s.enemy[2]));
    gEn.visible=true; gEn.rotation.y=-fyaw(s.enemy_body)*Math.PI/180;
  } else gEn.visible=false;

  const showAim=g('los').checked&&s.aim&&s.my;
  gAim.visible=!!showAim; gLos.visible=!!showAim;
  if(showAim){
    const a=toScene(s.aim[0],hAt(s.aim[0],s.aim[2])+1.4,s.aim[2]);
    gAim.position.copy(a);
    const o=toScene(s.my[0],hAt(s.my[0],s.my[2])+2.6,s.my[2]);
    gLos.geometry.setFromPoints([o,a]); gLos.computeLineDistances();
  }
  setLine(gPath,s.path,g('pathC').checked,1.6);
  setLine(gTrailMy,s.trail&&s.trail.my,g('trail').checked);
  setLine(gTrailEn,s.trail&&s.trail.enemy,g('trail').checked);
  if(s.dest&&g('pathC').checked){ gDest.visible=true;
    gDest.position.copy(toScene(s.dest[0],hAt(s.dest[0],s.dest[1])+7,s.dest[1]));
  } else gDest.visible=false;

  gShots.clear();
  if(g('shots').checked&&s.shots) s.shots.forEach(x=>gShots.add(arc(x.f,x.i,x.hit)));
  gObjs.clear();
  if(g('objs').checked&&s.objects) s.objects.forEach(o=>gObjs.add(marker(o)));

  // 장애물로 등록된 적 전차. 목록에서 사라지면(파괴) 자동으로 없어진다.
  gObsTank.clear();
  (s.threats||[]).forEach(o=>gObsTank.add(obsTank(o)));

  // 위협 반경 - 5100 포트 적 전차 + 장애물 적 전차 모두
  const src=[];
  if(s.enemy) src.push({x:s.enemy[0], z:s.enemy[2]});
  (s.threats||[]).forEach(o=>src.push({x:o.x, z:o.z}));
  const key=g('threat').checked
    ? src.map(p=>p.x.toFixed(0)+','+p.z.toFixed(0)).join('|')+'@'+vex : 'off';
  if(key!==threatKey){
    threatKey=key; gThreat.clear();
    if(g('threat').checked)
      src.forEach(p=>gThreat.add(
        threatRing(p.x,p.z,s.threat_r||130,s.threat_core||30)));
  }

  const hp=(v,el,bar,col)=>{ g(el).textContent=v==null?'-':v;
    g(bar).style.width=(v==null?100:Math.max(0,Math.min(100,v)))+'%';
    g(bar).style.background=col;};
  hp(s.my_hp,'mhp','mhpb','#3fb950'); hp(s.enemy_hp,'ehp','ehpb','#f85149');
  g('dist').textContent=s.dist!=null?s.dist+' m':'-';
  g('sug').textContent=s.suggest!=null?s.suggest+' m':'-';
  g('fc').textContent=s.fc;
  g('ph').textContent=s.p_hit!=null?s.p_hit:'-';
  g('rl').textContent=s.reload!=null?s.reload+' s':'-';
  g('sc').textContent=`${s.fired} / ${s.hits}`;
  g('hr').textContent=s.rate!=null?s.rate+' %':'-';
  g('pl').textContent=(s.path||[]).length;
  g('ob').textContent=(s.objects||[]).length;
  g('tk').textContent=(s.threats||[]).length;
  g('st').textContent=s.t!=null?s.t.toFixed(1):'-';
  g('swapmsg').style.display = s.swapped ? 'block' : 'none';
}
let camX=0, camZ=0;          // 실제 주시점. 목표값을 향해 서서히 따라간다
function animate(){
  requestAnimationFrame(animate);
  let tx=panX, tz=panZ;
  if(follow&&last&&last.my){ tx=last.my[0]-SPAN/2; tz=last.my[2]-SPAN/2; }
  // 목표점이 바뀌어도 시점이 튀지 않도록 감쇠를 준다
  camX += (tx-camX)*0.08;
  camZ += (tz-camZ)*0.08;
  cam.position.set(camX+Math.sin(yaw)*Math.cos(pitch)*dist,
    Math.sin(pitch)*dist+30, camZ+Math.cos(yaw)*Math.cos(pitch)*dist);
  cam.lookAt(camX,20,camZ); rnd.render(scene,cam);
}
</script></body></html>"""
