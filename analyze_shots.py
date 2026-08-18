# -*- coding: utf-8 -*-
# ── 버전 ────────────────────────────────────────────────
#   파일   analyze_shots.py
#   버전   v8   (2026-08-11)   A13 지형·장애물 절 추가
#   역할   사격 로그 분석기.
#   변경 이력은 같은 폴더의  변경이력.md  를 볼 것.
#   ※ 파일명은 바꾸지 않는다 (import 가 이름으로 걸려 있다).
#      버전 구분은 이 배너 + 날짜 폴더(260806/260807/…) 로 한다.
# ────────────────────────────────────────────────────────
"""
analyze_shots.py (v8) - 실사격 로그 분석 · 튜닝 근거 생성

무엇을 하는가
    auto_aim_bot_v6.py 가 /save?tag=... 로 내보낸
        shots_v6_<tag>.csv   발사 1발 = 1행 (34 컬럼)
        ticks_v6_<tag>.csv   매 틱 1행 (54 컬럼)
    을 읽어, "왜 빗나갔는가"를 원인별로 분해하고 튜닝 방향을 제시한다.

핵심 질문 하나
    빗나간 발이
      (a) 조준이 덜 끝났는데 쏴서 빗나갔나   -> 데드밴드/정렬을 조인다
      (b) 조준은 맞았는데 예측이 틀렸나       -> 예측/교전거리를 손본다
      (c) 조준점 자체가 잘못 놓였나           -> lon_gain / 바이어스를 손본다
    셋은 처방이 정반대다. 이 스크립트는 셋을 구분하는 것이 목적이다.

실행
    python analyze_shots.py                 # 폴더의 shots_v6_*.csv 전부
    python analyze_shots.py static evade    # 태그 지정
    python analyze_shots.py --file a.csv    # 파일 직접 지정
"""
import csv
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BAR = "=" * 66
SUB = "-" * 66


# ══════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════
def f(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def wilson(hits, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간. 표본이 적을 때 정규근사보다 정확하다."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def fmt(v, n=2, w=0, unit=""):
    if v is None:
        return f"{'-':>{w}}"
    return f"{v:{w}.{n}f}{unit}"


# ══════════════════════════════════════════════════════════
# 1. 사격 로그 파싱
# ══════════════════════════════════════════════════════════
def parse_shots(rows):
    """유효한 사격만 추린다 (적탄 피격·비행중 제외)"""
    out = []
    for r in rows:
        kind = (r.get("kind") or "").strip()
        if kind in ("incoming", "pending", ""):
            continue
        rec = {
            "id": r.get("id"), "time": f(r.get("time")),
            "cond": (r.get("cond") or "").strip(),
            "engage": (r.get("engage") or "").strip(),      # 정지-정지 등
            "sweep_range": f(r.get("sweep_range")),          # 사거리 시험 목표
            "dist": f(r.get("dist")),
            "aim_elev": f(r.get("aim_elev")),
            "hit": kind == "tank",
            "kind": kind,
            "range_err": f(r.get("range_err")),
            "cross_err": f(r.get("cross_err")),
            "p_hit": f(r.get("p_hit")),
            "sig_lat": f(r.get("sig_lat")), "sig_lon": f(r.get("sig_lon")),
            "half_lat": f(r.get("half_lat")), "half_lon": f(r.get("half_lon")),
            "lon_short": f(r.get("lon_short")), "lon_long": f(r.get("lon_long")),
            "lon_shift": f(r.get("lon_shift")),
            "tof": f(r.get("tof")), "lead": f(r.get("lead")),
            "own_speed": f(r.get("own_speed")),
            "enemy_speed": f(r.get("enemy_speed")),
            "zone": (r.get("zone") or "").strip(),
        }
        out.append(rec)
    return out


def classify_miss(s):
    """
    빗나간 발의 원인을 분류한다.

    조준점 기준 허용 범위는 비대칭이다.
        짧은 쪽 여유 = lon_short + lon_shift   (조준점을 길게 밀어둔 만큼 더 여유)
        긴 쪽 여유   = lon_long  - lon_shift
        횡방향 여유  = half_lat
    """
    re_, ce = s["range_err"], s["cross_err"]
    if re_ is None or ce is None:
        return "판정불가"
    ls, ll = s["lon_short"], s["lon_long"]
    sh = s["lon_shift"] or 0.0
    hl = s["half_lat"]
    if None in (ls, ll, hl):
        return "판정불가"
    room_short = ls + sh
    room_long = ll - sh
    over_lat = abs(ce) > hl
    over_long = re_ > room_long
    over_short = re_ < -room_short
    if over_lat and (over_long or over_short):
        return "복합"
    if over_lat:
        return "횡방향"
    if over_long:
        return "종방향(길게)"
    if over_short:
        return "종방향(짧게)"
    return "창 안인데 빗나감"


# ══════════════════════════════════════════════════════════
# 2. 조건별 요약
# ══════════════════════════════════════════════════════════
def report_summary(tag, shots):
    n = len(shots)
    hits = sum(1 for s in shots if s["hit"])
    p, lo, hi = wilson(hits, n)
    print(BAR)
    print(f"[{tag}]  사격 {n}발 · 명중 {hits}발 · 명중률 {p*100:.1f}%")
    print(f"        95% 신뢰구간 {lo*100:.1f} ~ {hi*100:.1f}%", end="")
    if n < 30:
        print(f"   ! 표본 {n}발 - 30발 이상 권장")
    else:
        print()
    if lo >= 0.90:
        print("        판정: 90% 초과를 통계적으로 주장할 수 있다")
    elif p >= 0.90:
        print(f"        판정: 점추정은 90% 초과지만 하한이 {lo*100:.1f}% 라 표본이 부족하다")
    else:
        print("        판정: 90% 미달")
    print(BAR)

    kinds = {}
    for s in shots:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    print("  착탄 종류:", ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    return hits, n


def report_errors(shots):
    print("\n  ── 오차 분해 (조준점 기준, 시선 좌표계) " + "─" * 24)
    for name, key in (("종방향(사거리)", "range_err"), ("횡방향(방위)", "cross_err")):
        allv = [s[key] for s in shots if s[key] is not None]
        hv = [s[key] for s in shots if s["hit"] and s[key] is not None]
        mv = [s[key] for s in shots if not s["hit"] and s[key] is not None]
        if not allv:
            continue
        print(f"    {name}")
        for lbl, v in (("전체", allv), ("명중", hv), ("빗나감", mv)):
            if not v:
                continue
            print(f"      {lbl:6s} n={len(v):3d}  평균 {mean(v):+6.2f}  "
                  f"표준편차 {fmt(sd(v), 2):>5}  "
                  f"[{pct(v,0.05):+6.2f} .. {pct(v,0.95):+6.2f}] m")
    # ── A12 (v7): 전차 명중탄만 있을 때의 해석 주의 ──────────
    #
    #   /update_bullet 이 주는 착탄 좌표는 '탄이 멈춘 지점'이다.
    #   전차에 맞으면 그 지점은 **차체 표면**이므로, 조준점보다
    #   계통적으로 1~2 m 짧게 기록된다. 탄도 오차가 아니라 기하다.
    #   (v6.1 에서 이걸 탄도 오차로 오해해 자기보정이 -6 m 까지 폭주했다)
    #
    #   따라서 명중탄만 모인 데이터의 '평균 종오차'를 편향으로 읽으면 안 된다.
    #   편향은 **지면 착탄(terrain)** 에서만 편향 없이 잰다.
    n_tank = sum(1 for s in shots if s["kind"] == "tank")
    n_terr = sum(1 for s in shots if s["kind"] == "terrain")
    if n_tank and n_terr == 0:
        print()
        print("    [주의] 이 데이터는 전부 전차 명중탄이다 (지면 착탄 0발).")
        print("           착탄 좌표가 차체 '표면'이라 조준점보다 1~2 m 짧게 찍힌다.")
        print("           아래 '평균 종오차'는 탄도 편향이 아니라 기하학적 산물이다.")
        print("           편향을 재려면 지면 착탄이 필요하다 "
              "(measure_harness 의 P2/P6).")
        sh_ = [s["lon_shift"] for s in shots if s["lon_shift"] is not None]
        rr_ = [s["range_err"] for s in shots if s["range_err"] is not None]
        if sh_ and rr_:
            print(f"           표적 중심 기준 착탄 위치 "
                  f"= 종오차 {mean(rr_):+.2f} + lon_shift {mean(sh_):+.2f} "
                  f"= {mean(rr_) + mean(sh_):+.2f} m")

    # 계통 vs 산포 판정
    allr = [s["range_err"] for s in shots if s["range_err"] is not None]
    if len(allr) >= 5:
        m, s_ = mean(allr), sd(allr) or 0.0
        print()
        if s_ > 1e-9 and abs(m) > 2 * s_:
            print(f"    -> 종방향은 계통 편향이 지배적이다 (평균 {m:+.2f} m, "
                  f"표준편차 {s_:.2f} m).")
            print("       조준점 위치(lon_gain) 또는 탄도 상수를 고쳐야 한다.")
            print("       데드밴드를 좁혀도 평균은 안 움직인다.")
        else:
            print(f"    -> 종방향은 산포가 지배적이다 (평균 {m:+.2f} m, "
                  f"표준편차 {s_:.2f} m).")
            print("       예측 오차 또는 조준 수렴 문제다. 교전 거리·데드밴드를 본다.")


def report_terrain(shots):
    """
    A13 (2026-08-11) — 지형·장애물 절.

    8/11 까지의 전 로그 914발에서 kind 는  tank 807 · terrain 107 ·
    **obstacle 0** 이었다. Simple Flat 에만 쐈다는 증거다.
    평가 맵 Forest and River 에서는 나무가 사격선을 막을 수 있으므로
    obstacle 비율을 사격당 지표로 따로 본다.

    이 비율이 8/13 을 B17(속도 정합) 에 쓸지 B18(LOS 차폐) 에 쓸지 가른다.
    """
    n = len(shots)
    if not n:
        return 0.0
    kinds = {}
    for s in shots:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    n_obs = kinds.get("obstacle", 0)
    n_terr = kinds.get("terrain", 0)
    r_obs = 100.0 * n_obs / n

    print("\n  ── 지형 · 장애물 " + "─" * 46)
    for k in sorted(kinds):
        lbl = {"tank": "전차 명중", "terrain": "지면",
               "obstacle": "장애물"}.get(k, k)
        print(f"    {lbl:10s} {kinds[k]:4d}발  {100.0*kinds[k]/n:5.1f}%")

    if n_obs == 0:
        print("\n    장애물 명중 0발 — 사격선을 막는 지물이 없는 맵으로 보인다.")
        print("    (Simple Flat 기준. Forest 로그라면 나무가 실제로 안 막는다는 뜻이다)")
        return 0.0

    # 장애물에 막힌 사격은 거리와 상관이 있는지 본다.
    print(f"\n    장애물 차폐율 {r_obs:.1f}%  ({n_obs}/{n})")
    bands = [(0, 40), (40, 70), (70, 100), (100, 999)]
    print("      거리대별")
    for lo, hi in bands:
        grp = [s for s in shots
               if s["dist"] is not None and lo <= s["dist"] < hi]
        if not grp:
            continue
        o = sum(1 for s in grp if s["kind"] == "obstacle")
        h = sum(1 for s in grp if s["hit"])
        tag = f"{lo}~{hi} m" if hi < 999 else f"{lo} m+"
        print(f"        {tag:10s} n={len(grp):3d}  장애물 {o:3d} "
              f"({100.0*o/len(grp):4.1f}%)  명중 {h:3d} ({100.0*h/len(grp):5.1f}%)")

    # 8/13 판단 기준 (Forest맵_측정계획_0812.md 와 같은 표)
    hits = sum(1 for s in shots if s["hit"])
    hr = 100.0 * hits / n
    print()
    if hr < 90.0:
        verdict = ("명중률이 90% 밑이다 -> 차폐율과 무관하게 "
                   "**B18(LOS 차폐 판정) 우선**")
    elif r_obs > 15.0:
        verdict = "차폐율 15% 초과 -> 8/13 하루 전부 **B18(LOS 차폐 판정)**"
    elif r_obs >= 5.0:
        verdict = "차폐율 5~15% -> 8/13 오전 **B18**, 오후 B17"
    else:
        verdict = "차폐율 5% 미만 -> 원래대로 **B17(속도 정합)**"
    print(f"    -> {verdict}")
    print(f"       (명중률 {hr:.1f}%, 지면 {n_terr}발)")
    return r_obs


def report_miss_causes(shots):
    misses = [s for s in shots if not s["hit"]]
    if not misses:
        print("\n  ── 실패 원인 " + "─" * 50)
        print("    빗나간 발 없음")
        return {}
    print("\n  ── 실패 원인 분류 " + "─" * 45)
    cause = {}
    for s in misses:
        c = classify_miss(s)
        cause.setdefault(c, []).append(s)
    for c, v in sorted(cause.items(), key=lambda x: -len(x[1])):
        rs = [x["range_err"] for x in v if x["range_err"] is not None]
        cs = [x["cross_err"] for x in v if x["cross_err"] is not None]
        print(f"    {c:16s} {len(v):3d}발 ({len(v)/len(misses)*100:4.0f}%)  "
              f"종 {fmt(mean(rs), 2, 6)}  횡 {fmt(mean(cs), 2, 6)} m")
    print("\n    빗나간 발 상세 (최대 8발)")
    print(f"      {'id':>4s} {'거리':>7s} {'종오차':>8s} {'횡오차':>8s} "
          f"{'허용 짧/긴':>13s} {'횡허용':>7s} {'p_hit':>6s} {'표적속도':>8s}")
    for s in misses[:8]:
        sh = s["lon_shift"] or 0.0
        rs = (s["lon_short"] + sh) if s["lon_short"] is not None else None
        rl = (s["lon_long"] - sh) if s["lon_long"] is not None else None
        print(f"      {str(s['id']):>4s} {fmt(s['dist'],1,7,'m')} "
              f"{fmt(s['range_err'],2,8)} {fmt(s['cross_err'],2,8)} "
              f"{fmt(rs,1,6)}/{fmt(rl,1,6)} {fmt(s['half_lat'],1,7)} "
              f"{fmt(s['p_hit'],2,6)} {fmt(s['enemy_speed'],1,8)}")
    return cause


def report_breakdown(shots):
    print("\n  ── 구간별 명중률 " + "─" * 46)
    for name, keyf, bins in (
        ("거리", lambda s: s["dist"], [(0, 30), (30, 40), (40, 55), (55, 75), (75, 999)]),
        ("표적 속도", lambda s: s["enemy_speed"],
         [(0, 0.6), (0.6, 8), (8, 15), (15, 99)]),
        ("비행시간", lambda s: s["tof"], [(0, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 9)]),
    ):
        print(f"    [{name}]")
        for lo, hi in bins:
            v = [s for s in shots
                 if keyf(s) is not None and lo <= keyf(s) < hi]
            if not v:
                continue
            h = sum(1 for s in v if s["hit"])
            p, wl, wh = wilson(h, len(v))
            mark = "" if wl >= 0.90 else ("  <- 약점" if p < 0.90 else "")
            print(f"      {lo:5.1f} ~ {hi:5.1f}  n={len(v):3d}  "
                  f"{p*100:5.1f}%  [{wl*100:4.1f} .. {wh*100:5.1f}]{mark}")


def report_calibration(shots):
    """모델이 예측한 p_hit 과 실제 명중률의 일치도"""
    v = [s for s in shots if s["p_hit"] is not None]
    if len(v) < 10:
        return
    print("\n  ── P(hit) 캘리브레이션 (모델 예측 vs 실제) " + "─" * 21)
    print(f"      {'예측구간':>12s} {'n':>4s} {'예측평균':>9s} {'실제':>8s} {'차이':>8s}")
    tot_gap = []
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        b = [s for s in v if lo <= s["p_hit"] < hi]
        if not b:
            continue
        pred = mean([s["p_hit"] for s in b])
        act = sum(1 for s in b if s["hit"]) / len(b)
        tot_gap.append((pred - act) * len(b))
        print(f"      {lo:.2f}~{hi:.2f}   {len(b):4d} {pred*100:8.1f}% "
              f"{act*100:7.1f}% {(act-pred)*100:+7.1f}%p")
    g = sum(tot_gap) / len(v)
    print()
    if g > 0.05:
        print(f"    -> 모델이 명중률을 {g*100:.1f}%p 과신하고 있다.")
        print("       허용창(half_lat / lon_long) 이 실제보다 넓게 계산되고 있거나,")
        print("       예측 불확실도(sigma)를 과소평가하고 있다.")
    elif g < -0.05:
        print(f"    -> 모델이 명중률을 {-g*100:.1f}%p 과소평가하고 있다.")
        print("       p_hit_min 을 낮춰 사격 기회를 늘릴 여지가 있다.")
    else:
        print(f"    -> 예측과 실제가 잘 맞는다 (편차 {g*100:+.1f}%p).")


def report_engage(shots):
    """A11 (v7): 교전 형태별 - 누가 움직이고 있었나"""
    have = [s for s in shots if s["engage"]]
    if not have:
        return
    print("\n  ── 교전 형태별 (아군-적) " + "─" * 38)
    print(f"      {'형태':<12s} {'n':>4s} {'명중률':>8s} {'95% 구간':>16s} "
          f"{'종오차 평균':>11s} {'종 표준편차':>11s}")
    for key in ("정지-정지", "정지-이동", "이동-정지", "이동-이동"):
        v = [s for s in have if s["engage"] == key]
        if not v:
            continue
        h = sum(1 for s in v if s["hit"])
        p, lo, hi = wilson(h, len(v))
        rs = [s["range_err"] for s in v if s["range_err"] is not None]
        print(f"      {key:<12s} {len(v):4d} {p*100:7.1f}% "
              f"{lo*100:7.1f}~{hi*100:5.1f}% {fmt(mean(rs),2,11)} {fmt(sd(rs),2,11)}")
    other = [s for s in have if s["engage"] not in
             ("정지-정지", "정지-이동", "이동-정지", "이동-이동")]
    if other:
        print(f"      (기타 {len(other)}발)")


def report_sweep(shots):
    """
    A10 (v7): 사거리 스윕 분석.

    사거리별로 (1) 명중률 (2) 계통 편향 (3) 산포 를 따로 본다.
      계통 편향이 거리에 따라 기울어지면  -> 탄도 상수(v, 포구오프셋)가 틀렸다
      편향이 일정하면                      -> 조준점 오프셋(lon_gain) 문제다
      산포가 거리에 비례해 커지면          -> 조준 분해능(데드밴드) 한계다
    """
    sw = [s for s in shots if s["sweep_range"] is not None]
    if len(sw) < 4:
        return
    print("\n  ── 사거리 스윕 (정지-정지 시험) " + "─" * 31)
    grp = {}
    for s in sw:
        grp.setdefault(s["sweep_range"], []).append(s)
    print(f"      {'목표':>6s} {'n':>3s} {'실거리':>8s} {'명중률':>7s} "
          f"{'종편향':>8s} {'종산포':>8s} {'횡편향':>8s} {'앙각':>7s} {'비행':>6s}")
    rows = []
    for r in sorted(grp):
        v = grp[r]
        h = sum(1 for s in v if s["hit"])
        rs = [s["range_err"] for s in v if s["range_err"] is not None]
        cs = [s["cross_err"] for s in v if s["cross_err"] is not None]
        dd = [s["dist"] for s in v if s["dist"] is not None]
        el = [s["aim_elev"] for s in v if s.get("aim_elev") is not None]
        tf = [s["tof"] for s in v if s["tof"] is not None]
        m = mean(rs)
        rows.append((r, m))
        print(f"      {r:6.0f} {len(v):3d} {fmt(mean(dd),1,8)} "
              f"{h/len(v)*100:6.1f}% {fmt(m,2,8)} {fmt(sd(rs),2,8)} "
              f"{fmt(mean(cs),2,8)} {fmt(mean(el),2,7)} {fmt(mean(tf),2,6)}")

    # 편향의 거리 의존성 - 탄도 상수 문제인지 조준점 문제인지 가른다
    pts = [(r, m) for r, m in rows if m is not None]
    if len(pts) >= 3:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs)
        mx, my = mean(xs), mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx > 1e-9 else 0.0
        print(f"\n      종편향의 거리 기울기 = {slope*100:+.2f} m / 100 m  "
              f"(절편 {my - slope*mx:+.2f} m)")
        n_terr = sum(1 for s in sw if s["kind"] == "terrain")
        if n_terr == 0:
            print("      (전부 전차 명중탄 - 절편은 차체 표면 기하가 섞여 있다.")
            print("       그러나 '기울기'는 그 영향을 받지 않으므로 그대로 유효하다)")
        if abs(slope) * 100 > 1.5:
            print("      -> 편향이 거리에 따라 변한다. 탄도 모델 문제다.")
            print("         analyze_measure.py 의 M1 재적합(포구 오프셋 포함)을 "
                  "다시 돌리고 v/h/L 을 갱신할 것.")
        else:
            print("      -> **편향이 거리와 무관하게 일정하다. 탄도 모델이 정확하다.**")
            print("         (거리 20~125 m 에서 계통 잔차가 안 생긴다는 뜻이다.")
            print("          포구속도·포구오프셋을 다시 잴 필요가 없다)")


# ══════════════════════════════════════════════════════════
# 3. 틱 로그 - 조준 수렴 진단
# ══════════════════════════════════════════════════════════
def report_ticks(tag, ticks, shots):
    if not ticks:
        return None
    fire_rows = [t for t in ticks if (t.get("fire") or "").strip() == "True"]
    print("\n  ── 조준 수렴 진단 (틱 로그 " + f"{len(ticks)}행) " + "─" * 26)
    if not fire_rows:
        print("    발사 틱 없음")
        return None

    # 발사 시점의 조준 오차 / 데드밴드 비율
    ry, rp = [], []
    for t in fire_rows:
        ye, yd = f(t.get("yaw_err")), f(t.get("yaw_db"))
        pe, pd = f(t.get("pitch_err")), f(t.get("pitch_db"))
        if ye is not None and yd:
            ry.append(abs(ye) / yd)
        if pe is not None and pd:
            rp.append(abs(pe) / pd)
    print(f"      발사 시점 |조준오차| / 데드밴드   (1.0 = 데드밴드 경계에서 발사)")
    for name, v in (("방위", ry), ("앙각", rp)):
        if not v:
            continue
        print(f"        {name}  평균 {mean(v):.2f}  중앙 {pct(v,0.5):.2f}  "
              f"90분위 {pct(v,0.9):.2f}  최대 {max(v):.2f}")

    # ── A9 (v7): 데드밴드 ↔ 실측 산포 교차검증 ─────────────
    #
    #   조준 오차가 데드밴드 안에서 대략 균등하게 분포한다고 보면
    #   그 표준편차는 (슬랙 / sqrt(3)) 이다.
    #   이 값이 실측 산포에 가까우면, 산포의 원인은 '예측'이 아니라
    #   '조준을 덜 하고 쏜 것'이다. 처방이 정반대이므로 반드시 갈라야 한다.
    pds = [f(t.get("pitch_db")) for t in fire_rows]
    drs = [f(t.get("drdt")) for t in fire_rows]
    yds = [f(t.get("yaw_db")) for t in fire_rows]
    dis = [f(t.get("dist")) for t in fire_rows]
    lon_pair = [(a, b) for a, b in zip(pds, drs) if a and b]
    lat_pair = [(a, b) for a, b in zip(yds, dis) if a and b]

    share_out = {}

    def _cross(name, slack, obs, unit="m"):
        if slack is None or obs is None or obs < 1e-9:
            return
        pred = slack / math.sqrt(3)
        share = min(1.0, (pred / obs) ** 2)
        share_out[name[:2]] = share
        print(f"      {name}")
        print(f"        데드밴드 슬랙 ±{slack:.2f} {unit}"
              f"  -> 예상 표준편차 {pred:.2f} {unit}")
        print(f"        실측 표준편차  {obs:.2f} {unit}"
              f"   -> 산포의 {share*100:.0f}% 를 데드밴드가 설명")
        if share > 0.5:
            print("        ** 데드밴드가 산포의 주원인이다. "
                  "조이면 바로 좋아진다. **")
        elif share > 0.25:
            print("        데드밴드가 절반 가까이 기여한다. 조일 여지가 있다.")
        else:
            print("        데드밴드 기여가 작다. 예측 오차 쪽을 봐야 한다.")

    print("\n      [데드밴드 ↔ 실측 산포 교차검증]")
    obs_lon = sd([s["range_err"] for s in shots if s["range_err"] is not None])
    obs_lat = sd([s["cross_err"] for s in shots if s["cross_err"] is not None])
    if lon_pair:
        _cross("종방향(앙각)", mean([a * b for a, b in lon_pair]), obs_lon)
    if lat_pair:
        _cross("횡방향(방위)",
               mean([math.radians(a) * b for a, b in lat_pair]), obs_lat)

    # 조일 수 있는 여유가 있는가 - 포탑 최소 이동량과 비교
    dts = [f(t.get("ctrl_dt")) for t in fire_rows]
    dts = [d for d in dts if d]
    if dts and lon_pair:
        dt = pct(dts, 0.5)
        min_pitch = 5.0 * 0.02 * dt      # pitch_rate * w_min * dt
        cur_db = mean(pds) if pds else None
        if cur_db:
            print(f"\n      앙각 최소 이동량 = {min_pitch:.4f} deg "
                  f"(dt={dt:.3f}s, w_min=0.02)")
            print(f"      현재 앙각 데드밴드 = {cur_db:.3f} deg "
                  f"= 최소 이동량의 {cur_db/min_pitch:.0f} 배")
            if cur_db / min_pitch > 8:
                sug = max(min_pitch * 4, cur_db * 0.3)
                share_out["suggest_pitch_db"] = round(sug, 2)
                share_out["cur_pitch_db"] = round(cur_db, 3)
                print(f"      -> 분해능에 여유가 크다. 데드밴드를 "
                      f"{sug:.2f} deg 수준까지 조일 수 있다.")
                print("         CFG['pitch_db_max'] 를 낮출 것.")

    # 사격 임계값에 밀려 쐈는가 (인내 로직)
    forced = 0
    for t in fire_rows:
        ph, th = f(t.get("p_hit")), f(t.get("p_threshold"))
        if ph is not None and th is not None and th < 0.89:
            forced += 1
    print(f"      인내 로직으로 임계값이 낮아진 상태에서 쏜 발 "
          f"{forced}/{len(fire_rows)} ({forced/len(fire_rows)*100:.0f}%)")
    if forced / len(fire_rows) > 0.3:
        print("        -> 사격 기회가 부족하다. 교전 거리 밴드나 track_duty_max 를 본다.")

    # 상태 분포
    ph_cnt = {}
    for t in ticks:
        k = (t.get("phase") or "").strip()
        ph_cnt[k] = ph_cnt.get(k, 0) + 1
    tot = sum(ph_cnt.values()) or 1
    print("      상태 분포: " + "  ".join(
        f"{k} {v/tot*100:.0f}%" for k, v in sorted(ph_cnt.items(), key=lambda x: -x[1])))

    # 제어 주기
    dts = [f(t.get("ctrl_dt")) for t in ticks]
    dts = [d for d in dts if d]
    if dts:
        med = pct(dts, 0.5)
        print(f"      제어 주기 실측 중앙값 = {med:.4f} s")
        if abs(med - 0.41) > 0.15:
            print(f"        ! 0.41 s 가정과 다르다. TurretParams(dt={med:.3f}) 로 "
                  "맞추고 재튜닝이 필요하다.")
    out = {"ry": ry, "rp": rp, "forced": forced / len(fire_rows)}
    out.update(share_out)
    if dts:
        out["ctrl_dt"] = pct(dts, 0.5)
    return out


# ══════════════════════════════════════════════════════════
# 4. 튜닝 권고
# ══════════════════════════════════════════════════════════
def recommend(tag, shots, cause, tick_stat):
    print("\n  ── 튜닝 권고 " + "─" * 50)
    n = len(shots)
    hits = sum(1 for s in shots if s["hit"])
    p, lo, _ = wilson(hits, n)
    recs = []

    misses = [s for s in shots if not s["hit"]]
    nm = len(misses) or 1
    c_long = len(cause.get("종방향(길게)", []))
    c_short = len(cause.get("종방향(짧게)", []))
    c_lat = len(cause.get("횡방향", []))
    c_in = len(cause.get("창 안인데 빗나감", []))

    allr = [s["range_err"] for s in shots if s["range_err"] is not None]
    m_r = mean(allr) if allr else None
    s_r = sd(allr) if len(allr) > 1 else None
    shifts = [s["lon_shift"] for s in shots if s["lon_shift"] is not None]
    m_shift = mean(shifts) if shifts else None

    # (1) 종방향 계통 편향
    if c_long / nm > 0.4 and m_r is not None and m_r > 1.0:
        cur = "현재 lon_gain 값"
        recs.append((
            "조준점이 계통적으로 길다",
            f"빗나간 발의 {c_long/nm*100:.0f}%가 '종방향(길게)'이고 "
            f"전체 평균 종오차가 {m_r:+.2f} m 다."
            + (f" 의도한 길게 밀기(lon_shift)가 평균 {m_shift:+.2f} m 다."
               if m_shift else ""),
            f"CFG['lon_gain'] 을 낮춘다. 목표: 평균 종오차를 ±1 m 안으로. "
            f"{cur}에서 약 {max(0.0, 1 - 1.0/max(0.5, m_r)):.0%} 만큼 줄여 보고 재측정."))
    if c_short / nm > 0.3:
        recs.append((
            "조준점이 짧다",
            f"빗나간 발의 {c_short/nm*100:.0f}%가 '종방향(짧게)'이다. "
            "탄이 표적 앞 지면에 박히면 무조건 빗나간다.",
            "CFG['lon_gain'] 을 올리거나, 탄도 상수(v, 포구오프셋)를 재측정한다."))

    # (1-b) 데드밴드가 산포의 주원인인가  <- 교차검증 결과
    if tick_stat and tick_stat.get("suggest_pitch_db"):
        sh_lon = tick_stat.get("종방", 0.0)
        if sh_lon > 0.4:
            recs.insert(0, (
                "앙각 데드밴드가 종방향 산포를 만들고 있다",
                f"데드밴드 슬랙이 만드는 예상 표준편차가 실측 산포의 "
                f"{sh_lon*100:.0f}% 를 설명한다. 현재 데드밴드 "
                f"{tick_stat['cur_pitch_db']:.3f} deg 는 포탑 최소 이동량의 수십 배다.",
                f"CFG['pitch_db_max'] 를 {tick_stat['suggest_pitch_db']:.2f} 로 낮춘다. "
                "사격 횟수가 크게 줄면 조금 되올린다."))

    # (2) 횡방향
    if c_lat / nm > 0.3:
        recs.append((
            "방위 조준이 부족하다",
            f"빗나간 발의 {c_lat/nm*100:.0f}%가 횡방향 초과다.",
            "CFG['db_safety'] 를 낮추거나(현재보다 0.05 씩), "
            "TurretParams(w_min) 이 실측값(0.01~0.02)인지 확인한다."))

    # (3) 조준이 데드밴드 경계에서 발사
    if tick_stat:
        rp, ry = tick_stat["rp"], tick_stat["ry"]
        if rp and pct(rp, 0.5) > 0.7:
            recs.append((
                "앙각이 데드밴드 경계에서 발사되고 있다",
                f"발사 시점 |앙각오차|/데드밴드 중앙값이 {pct(rp,0.5):.2f} 다. "
                "조준이 수렴해서가 아니라 '허용 범위에 겨우 들어와서' 쏘고 있다.",
                "FireControl(pitch_db_max) 를 낮추거나 CFG['db_safety'] 를 조인다. "
                "단 사격 횟수가 줄어드는지 함께 본다."))
        if ry and pct(ry, 0.5) > 0.7:
            recs.append((
                "방위가 데드밴드 경계에서 발사되고 있다",
                f"발사 시점 |방위오차|/데드밴드 중앙값이 {pct(ry,0.5):.2f} 다.",
                "CFG['db_safety'] 를 낮춘다."))
        if tick_stat["forced"] > 0.3:
            recs.append((
                "사격 기회가 부족하다",
                f"발사의 {tick_stat['forced']*100:.0f}%가 인내 로직으로 "
                "임계값이 낮아진 뒤 나갔다. 즉 '좋은 기회'가 잘 안 온다.",
                "CFG['band_near'/'band_far'] 로 교전 거리를 넓히거나 "
                "CFG['track_duty_max'] 를 올린다."))

    # (4) 창 안인데 빗나감 = 모델과 실제 히트박스 불일치
    if c_in / nm > 0.25:
        recs.append((
            "허용창 안인데 빗나갔다",
            f"빗나간 발의 {c_in/nm*100:.0f}%가 모델상 허용 범위 안이었다. "
            "허용창 계산이 실제 히트박스보다 낙관적이다.",
            "TargetSize(width/length/height) 를 실측값으로 확인하고, "
            "longitudinal_window 의 높이 항이 과대하지 않은지 본다."))

    # (5) 산포 지배
    if m_r is not None and s_r and abs(m_r) < 2 * s_r and s_r > 1.5:
        recs.append((
            "종방향 산포가 크다 (예측 오차)",
            f"평균 {m_r:+.2f} m 대비 표준편차 {s_r:.2f} m. 계통 편향이 아니라 "
            "표적 예측이 흔들리고 있다.",
            "교전 거리를 줄여 비행시간을 낮춘다 (오차는 비행시간의 제곱에 비례). "
            "CFG['band_far'] 를 낮추거나 CFG['p_hit_min'] 을 올린다."))

    if tick_stat and tick_stat.get("ctrl_dt"):
        cd = tick_stat["ctrl_dt"]
        if abs(cd - 0.41) > 0.15:
            recs.insert(0, (
                "제어 주기가 튜닝 기준과 다르다",
                f"실측 {cd:.3f} s. 이 프로젝트의 상수는 0.41 s 를 가정해 정했다. "
                "제어 주기는 폐루프 게인의 분모이자 조준 분해능의 결정 요인이다.",
                f"fire_control.TurretParams(dt={cd:.3f}) 와 "
                f"offline_sim.TruePhysics(dt={cd:.3f}) 로 맞추고 "
                "파라미터를 다시 스윕할 것. (이걸 안 맞추면 아래 권고도 근거가 약하다)"))

    if not recs:
        if lo >= 0.90:
            print("    현재 설정으로 목표를 만족한다. 변경할 이유가 없다.")
        else:
            print("    뚜렷한 단일 원인이 없다. 표본을 늘려 재측정할 것을 권한다.")
        return

    for i, (title, evidence, action) in enumerate(recs, 1):
        print(f"\n    [{i}] {title}")
        print(f"        근거 : {evidence}")
        print(f"        조치 : {action}")

    print("\n    ! 한 번에 하나만 바꾸고 재측정할 것. 동시에 바꾸면 "
          "어느 것이 효과였는지 알 수 없다.")


# ══════════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════════
def analyze_one(tag, shot_path, tick_path):
    raw = load(shot_path)
    if not raw:
        print(f"[{tag}] {os.path.basename(shot_path)} 를 읽을 수 없거나 비어 있다")
        return None
    shots = parse_shots(raw)
    if not shots:
        print(f"[{tag}] 유효한 사격 기록이 없다 (전부 '비행중' 또는 '적탄')")
        return None
    hits, n = report_summary(tag, shots)
    print_session(shot_path)
    report_errors(shots)
    report_terrain(shots)
    cause = report_miss_causes(shots)
    report_breakdown(shots)
    report_engage(shots)
    report_sweep(shots)
    report_calibration(shots)
    ticks = load(tick_path) if tick_path else []
    ts = report_ticks(tag, ticks, shots)
    recommend(tag, shots, cause, ts)
    print()
    return hits, n


def discover():
    """
    logs/<날짜>_<태그>/shots.csv 를 찾는다.
    구형(코드 폴더에 흩어진 shots_v6_*.csv)도 함께 지원한다.
    """
    found = []
    roots = [os.path.join(HERE, "logs"),
             os.path.abspath(os.path.join(HERE, os.pardir, "logs"))]
    seen = set()
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "*", "shots.csv"))):
            key = os.path.basename(os.path.dirname(p))
            if key in seen:
                continue
            seen.add(key)
            found.append((key, p))
    for p in sorted(glob.glob(os.path.join(HERE, "shots_v6*.csv"))):
        b = os.path.basename(p)[:-4]
        found.append((b.replace("shots_v6", "").lstrip("_") or "(무태그)", p))
    return found


def tick_path_for(shot_path):
    d = os.path.dirname(shot_path)
    cand = os.path.join(d, "ticks.csv")                 # 신형
    if os.path.exists(cand):
        return cand
    cand = shot_path.replace("shots_v6", "ticks_v6")    # 구형
    return cand if os.path.exists(cand) else None


def print_session(shot_path):
    """같은 폴더의 session.json 이 있으면 측정 조건을 먼저 보여준다"""
    import json
    p = os.path.join(os.path.dirname(shot_path), "session.json")
    if not os.path.exists(p):
        return
    try:
        m = json.load(open(p, encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return
    cfg = m.get("cfg") or {}
    print(f"  측정 조건: 적 행동 {m.get('enemy_behavior')} · "
          f"제어주기 {m.get('ctrl_dt_measured')} s · "
          f"pitch_db_max {cfg.get('pitch_db_max')} · "
          f"lon_gain {cfg.get('lon_gain')} · "
          f"p_hit_min {cfg.get('p_hit_min')} · 저장 {m.get('saved_at')}")
    sw = m.get("sweep")
    if sw and sw.get("ranges"):
        print(f"  사거리 스윕: {sw['ranges']} · 구간당 {sw.get('shots_per')}발")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--file" in sys.argv:
        i = sys.argv.index("--file")
        files = [(os.path.basename(p), p) for p in sys.argv[i + 1:]]
    elif args:
        allf = discover()
        files = [(t, p) for t, p in allf if any(a in t for a in args)]
        if not files:
            files = [(t, os.path.join(HERE, f"shots_v6_{t}.csv")) for t in args]
    else:
        files = discover()

    if not files:
        print("분석할 로그가 없다.")
        print("봇 화면에서 [세 개 한번에 저장] 을 누르면 "
              "logs/<날짜>_<태그>/ 에 생긴다.")
        return

    print(BAR)
    print("실사격 로그 분석  (analyze_shots.py v7)")
    print(BAR)
    total_h = total_n = 0
    per = []
    for tag, p in files:
        r = analyze_one(tag, p, tick_path_for(p))
        if r:
            per.append((tag, r[0], r[1]))
            total_h += r[0]
            total_n += r[1]

    if len(per) > 1:
        print(BAR)
        print("전체 요약")
        print(BAR)
        print(f"  {'조건':<14s} {'사격':>5s} {'명중':>5s} {'명중률':>8s} "
              f"{'95% 신뢰구간':>18s}  판정")
        for tag, h, n in per:
            p, lo, hi = wilson(h, n)
            v = "달성" if lo >= 0.90 else ("표본부족" if p >= 0.90 else "미달")
            print(f"  {tag:<14s} {n:5d} {h:5d} {p*100:7.1f}% "
                  f"{lo*100:8.1f} ~ {hi*100:5.1f}%  {v}")
        p, lo, hi = wilson(total_h, total_n)
        print(SUB)
        print(f"  {'합계':<14s} {total_n:5d} {total_h:5d} {p*100:7.1f}% "
              f"{lo*100:8.1f} ~ {hi*100:5.1f}%")


if __name__ == "__main__":
    main()
