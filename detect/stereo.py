import detect.detection_server as ts
import cv2
import numpy as np

VERTICAL_FOV = 28.0  # deg, 기존과 동일 가정
HORIZONTAL_FOV_STEREO = 47.81061
LATEST_INFO = {}

# 좌표 오차 원인 파악용 디버그 로그 스위치.
# True로 두면 compute_stereo_for_pair()의 중간 계산값(baseline/disparity/
# depth/bearing 등)과 raw/smoothed world 좌표가 전부 콘솔에 찍힌다.
# [PERF] 매 프레임, 매 pair마다 여러 줄짜리 f-string을 만들고 print()로
# stdout에 쓰는 것 자체가 threaded Flask 서버에서는 블로킹 비용이 크다.
# 좌표 디버깅이 필요할 때만 True로 켜고, 평소 주행/전투 중에는 False로 둘 것.
DEBUG_STEREO = False

# 감지된 오브젝트의 이름, 좌표값을 전역변수로 list 저장
DETECTED_OBJECTS_INFO = []

# 클래스별 기본 위협 가중치 (우린 화력으로 나눴지만 결국 클래스 별로 나눔)
FIREPOWER_TABLE = {
    "Human1": 20,   # 소총병
    "Human2": 50,   # 바주카병
    "Tank1": 100,   # 전차
}

MAX_FIREPOWER = max(FIREPOWER_TABLE.values())

MAX_RELEVANT_DISTANCE = 100.0   # 이 거리를 넘으면 위협도 0에 수렴

THREAT_PER = 0                  # 현재 눈(카메라)에 보이는 위협도 (위협도 관련)

# 오차값 줄이기 위한 변수 및 라이브러리
from collections import defaultdict, deque
import statistics as st

POSITION_HISTORY = defaultdict(lambda: deque(maxlen=5))   # 클래스별 최근 5개 관측 저장
OUTLIER_THRESHOLD = 15.0   # 중앙값에서 이 거리(m) 이상 벗어나면 이상치로 간주하고 제외
MIN_SAMPLES_BEFORE_OUTPUT = 3   # 이 개수만큼 쌓이기 전엔 값을 내보내지 않음

# 기능(함수) 모음 cell

# [PERF] Flask로 올라온 이미지를 디스크에 저장했다가 다시 읽는 대신,
# 메모리에서 바로 디코딩한다.
# 기존 방식(save -> YOLO가 다시 파일 열어서 read+decode -> os.remove)은
# 좌/우 이미지 한 장당 write 1회 + read 1회 + delete 1회, 총 6회의
# 디스크 I/O가 프레임마다 반복됐음. 이게 스테레오 카메라를 켰을 때만
# 버벅이던 원인 중 하나.
def decode_image_from_filestorage(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


# [PERF] 좌/우 이미지를 따로따로 두 번 추론하지 않고, 하나의 배치로 묶어서
# YOLO를 한 번만 호출한다. ultralytics YOLO는 numpy 배열 리스트를 넣으면
# 내부적으로 배치 추론을 수행하므로, 모델 forward pass 자체가
# 2번 -> 1번으로 줄어들고 전/후처리 오버헤드도 절반 수준으로 줄어든다.
# (이게 가장 큰 병목이었음 - 자세한 설명은 채팅 답변 참고)
def run_inference_batch(left_img, right_img):
    results = ts.model([left_img, right_img], verbose=False)
    left_result, right_result = results[0], results[1]
    img_h, img_w = left_result.orig_shape

    left_detections = left_result.boxes.data.cpu().numpy()
    right_detections = right_result.boxes.data.cpu().numpy()
    return left_detections, right_detections, img_w, img_h


# 이미 뽑아둔 추론 결과(detections)에서 원하는 클래스만 골라내기 (재추론 없음)
def filter_boxes_by_class(detections, target_class_id):
    return [[float(c) for c in box[:4]] for box in detections if int(box[5]) == target_class_id]

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def match_stereo_boxes(left_boxes, right_boxes, y_tolerance=20):
    # 세로(y) 위치가 가장 비슷한 것끼리 좌/우 bbox 짝짓기
    pairs = []
    used_right = set()

    for lb in left_boxes:
        _, ly = bbox_center(lb)
        best_match, best_diff = None, y_tolerance
        for i, rb in enumerate(right_boxes):
            if i in used_right:
                continue
            _, ry = bbox_center(rb)
            diff = abs(ly - ry)
            if diff < best_diff:
                best_match, best_diff = i, diff

        if best_match is not None:
            pairs.append((lb, right_boxes[best_match]))
            used_right.add(best_match)

    return pairs

def pixel_offset_to_angle(pixel_offset_ratio, fov_deg):
    half_fov_rad = ts.math.radians(fov_deg / 2)
    return ts.math.degrees(ts.math.atan(pixel_offset_ratio * ts.math.tan(half_fov_rad)))


def get_focal_px(img_w, fov_deg=HORIZONTAL_FOV_STEREO):
    return (img_w / 2) / ts.math.tan(ts.math.radians(fov_deg / 2))

# 오차값 줄이기 위한 함수
def smooth_position(class_name, world_pos):
    # 최근 관측치들의 중앙값을 기준으로, 너무 크게 벗어난(이상치) 값은 제외하고
    # 나머지를 평균 내서 안정화된 좌표를 반환.
    hist = POSITION_HISTORY[class_name]
    hist.append(world_pos)

    if len(hist) < MIN_SAMPLES_BEFORE_OUTPUT:
        return None   # 초기엔 값 자체를 안 줌 (원본 그대로 내보내지 않음)

    # 각 축의 중앙값으로 "대략적인 중심"을 잡음
    med = {
        "x": st.median(p["x"] for p in hist),
        "y": st.median(p["y"] for p in hist),
        "z": st.median(p["z"] for p in hist),
    }

    # 중앙값에서 OUTLIER_THRESHOLD 이상 벗어난 관측은 제외
    def dist_to_med(p):
        return ts.math.sqrt((p["x"]-med["x"])**2 + (p["y"]-med["y"])**2 + (p["z"]-med["z"])**2)

    filtered = [p for p in hist if dist_to_med(p) <= OUTLIER_THRESHOLD]
    if not filtered:
        filtered = list(hist)   # 전부 걸러졌으면(극단적 경우) 원본 그대로 사용

    n = len(filtered)
    return {
        "x": sum(p["x"] for p in filtered) / n,
        "y": sum(p["y"] for p in filtered) / n,
        "z": sum(p["z"] for p in filtered) / n,
    }

def compute_stereo_for_pair(left_bbox, right_bbox, img_w, img_h):
    # 짝지어진 bbox 하나로 거리/월드좌표 계산 (기존 stereo_estimate_position 핵심 로직)
    left_pos = LATEST_INFO.get("stereoCameraLeftPos")
    left_rot = LATEST_INFO.get("stereoCameraLeftRot")
    right_pos = LATEST_INFO.get("stereoCameraRightPos")
    right_rot = LATEST_INFO.get("stereoCameraRightRot")
    if not left_pos or not right_pos or not left_rot:
        if DEBUG_STEREO:
            print(
                "[STEREO DEBUG] LATEST_INFO에 카메라 좌표/회전이 없어서 계산 스킵 "
                f"(left_pos={left_pos}, right_pos={right_pos}, left_rot={left_rot})"
            )
        return None

    baseline = ts.math.sqrt(
        (left_pos["x"] - right_pos["x"]) ** 2 +
        (left_pos["y"] - right_pos["y"]) ** 2 +
        (left_pos["z"] - right_pos["z"]) ** 2
    )

    cxL, cyL = bbox_center(left_bbox)
    cxR, cyR = bbox_center(right_bbox)
    disparity = abs(cxL - cxR)
    if disparity < 1e-3:
        if DEBUG_STEREO:
            print(f"[STEREO DEBUG] disparity가 0에 가까워서 계산 스킵 (cxL={cxL}, cxR={cxR})")
        return None

    focal_px = get_focal_px(img_w)
    depth = baseline * focal_px / disparity

    h_offset = pixel_offset_to_angle((cxL - img_w / 2) / (img_w / 2), HORIZONTAL_FOV_STEREO)
    v_offset = pixel_offset_to_angle((cyL - img_h / 2) / (img_h / 2), VERTICAL_FOV)

    bearing = (left_rot["y"] + h_offset) % 360
    vertical = left_rot["x"] - v_offset   # 지난번 검증한 부호

    rad_h, rad_v = ts.math.radians(bearing), ts.math.radians(vertical)
    dx = depth * ts.math.cos(rad_v) * ts.math.sin(rad_h)
    dz = depth * ts.math.cos(rad_v) * ts.math.cos(rad_h)
    dy = depth * ts.math.sin(rad_v)

    world_pos = {"x": left_pos["x"] + dx, "y": left_pos["y"] + dy, "z": left_pos["z"] + dz}

    player_pos = LATEST_INFO.get("playerPos")
    distance_3d = None
    if player_pos:
        distance_3d = ts.math.sqrt(
            (player_pos["x"] - world_pos["x"]) ** 2 +
            (player_pos["y"] - world_pos["y"]) ** 2 +
            (player_pos["z"] - world_pos["z"]) ** 2
        )

    if DEBUG_STEREO:
        print(
            "[STEREO DEBUG] ---- 좌표 계산 중간값 ----\n"
            f"  left_pos={left_pos}  left_rot={left_rot}\n"
            f"  right_pos={right_pos}  right_rot={right_rot}\n"
            f"  playerPos={player_pos}\n"
            f"  bbox: left_center=({cxL:.1f},{cyL:.1f}) right_center=({cxR:.1f},{cyR:.1f}) "
            f"disparity={disparity:.2f}px img_w={img_w} img_h={img_h}\n"
            f"  baseline={baseline:.4f}m  focal_px={focal_px:.2f}  depth={depth:.3f}m\n"
            f"  h_offset={h_offset:.2f}deg  v_offset={v_offset:.2f}deg  "
            f"bearing={bearing:.2f}deg  vertical={vertical:.2f}deg\n"
            f"  dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}\n"
            f"  -> raw world_pos={world_pos}  distance_to_player={distance_3d}"
        )

    return {"world_pos": world_pos, "distance": distance_3d, "bearing": bearing}

def scan_all_objects(target_classes, left_img, right_img):
    # target_classes: {class_id: class_name, ...}
    all_objects = []

    # [PERF] 좌/우 이미지를 한 번의 배치 추론으로 처리 (기존: run_inference() 2번 호출)
    left_detections, right_detections, img_w, img_h = run_inference_batch(left_img, right_img)

    for class_id, class_name in target_classes.items():
        left_boxes = filter_boxes_by_class(left_detections, class_id)     # 재추론 없이 필터링만
        right_boxes = filter_boxes_by_class(right_detections, class_id)

        if not left_boxes or not right_boxes:
            continue

        pairs = match_stereo_boxes(left_boxes, right_boxes)

        for left_bbox, right_bbox in pairs:
            result = compute_stereo_for_pair(left_bbox, right_bbox, img_w, img_h)
            if result is None:
                continue
            result["class_name"] = class_name

            # 스무딩 적용
            smoothed_pos = smooth_position(class_name, result["world_pos"])
            if smoothed_pos is None:                      #아직 안정화 안 됐으면 이번 프레임은 건너뜀
                if DEBUG_STEREO:
                    print(
                        f"[STEREO DEBUG] {class_name}: 스무딩 샘플 부족으로 이번 프레임 스킵 "
                        f"(raw_world_pos={result['world_pos']})"
                    )
                continue
            result["raw_world_pos"] = result["world_pos"]   # 원본값도 참고용으로 남겨둠
            result["world_pos"] = smoothed_pos

            if DEBUG_STEREO:
                print(
                    f"[STEREO DEBUG] {class_name}: raw={result['raw_world_pos']} "
                    f"-> smoothed={smoothed_pos}"
                )

            player_pos = LATEST_INFO.get("playerPos")
            if player_pos:
                result["distance"] = ts.math.sqrt(
                    (player_pos["x"] - smoothed_pos["x"]) ** 2 +
                    (player_pos["y"] - smoothed_pos["y"]) ** 2 +
                    (player_pos["z"] - smoothed_pos["z"]) ** 2
                )

            all_objects.append(result)
    return all_objects

# 여기는 위험도 계산 함수
def distance_score(distance, max_relevant_distance=MAX_RELEVANT_DISTANCE):
    # 가까울수록 1에 가깝고, max_relevant_distance 이상이면 0.
    if distance is None:
        return 0.0
    if distance <= 0:
        return 1.0
    score = 1 - (distance / max_relevant_distance)
    return max(0.0, min(1.0, score))

def firepower_score(class_name):
    # 화력을 0~1로 정규화. 값이 클수록 위험.
    fp = FIREPOWER_TABLE.get(class_name, 0)
    return fp / MAX_FIREPOWER if MAX_FIREPOWER > 0 else 0.0

def compute_threat_score(class_name, distance, w_class=0.5, w_distance=0.5):
    # 화력 등급 × 거리 점수
    return firepower_score(class_name) * distance_score(distance)

def rank_objects_by_threat(objects):
    # objects: scan_all_objects()가 리턴한 리스트
    #          [{"class_name":.., "world_pos":.., "distance":.., ...}, ...]
    # 각 객체에 threat_score를 채워넣고, 위험도 높은 순으로 정렬해서 반환
    for obj in objects:
        obj["threat_score"] = compute_threat_score(obj["class_name"], obj["distance"])

    return sorted(objects, key=lambda o: o["threat_score"], reverse=True)

def total_threat_score(ranked_objects):
    # 감지된 모든 객체의 위험도 합
    return sum(obj["threat_score"] for obj in ranked_objects)

def save_detected_object_info(objects):
    objects_info = []

    print(f'탐지된 오브젝트 개수 : {len(objects)}')
    for obj in objects:
        object_info_pos = (obj['world_pos']['x'], obj['world_pos']['y'], obj['world_pos']['z'], obj['class_name'])
        objects_info.append(object_info_pos)

    return objects_info


# 여기부터 서버 통신 함수 (ally-controller.py가 tskijun.stereo_image() /
# tskijun.info() / tskijun.DETECTED_OBJECTS_INFO 형태로 직접 호출하는 부분)
def stereo_image():                             # 오브젝트 좌표, 위협도, 거리 계산은 다 여기서 실시.
    global THREAT_PER                           # (위협도 관련)
    global DETECTED_OBJECTS_INFO

    left_image = ts.request.files.get('left_image')
    right_image = ts.request.files.get('right_image')

    if not left_image or not right_image:
        return ts.jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    # [PERF] 디스크에 저장하지 않고 메모리에서 바로 디코딩 (아래 채팅 설명 참고)
    left_img = decode_image_from_filestorage(left_image)
    right_img = decode_image_from_filestorage(right_image)

    if left_img is None or right_img is None:
        return ts.jsonify({"result": "error", "message": "Failed to decode image"}), 400

    if DEBUG_STEREO:
        print(
            "[STEREO DEBUG] ==== /stereo_image 프레임 시작 ====\n"
            f"  playerPos={LATEST_INFO.get('playerPos')}  "
            f"stereoCameraLeftPos={LATEST_INFO.get('stereoCameraLeftPos')}  "
            f"stereoCameraLeftRot={LATEST_INFO.get('stereoCameraLeftRot')}"
        )

    objects = scan_all_objects(ts.target_classes, left_img, right_img)
    ranked = rank_objects_by_threat(objects)
    DETECTED_OBJECTS_INFO = save_detected_object_info(objects)
    total = total_threat_score(ranked)          # 눈(카메라)에 보이는 위협도의 총합 (위협도 관련)
    THREAT_PER = total                          # 이 end point에서 나온 위협도를 전역변수에 저장 (위협도 관련)
    print(DETECTED_OBJECTS_INFO)

    return ts.jsonify({"result": "success"})

def info():              # 내 위치값, 회전값등을 가져와야하기 때문에 여기서 LATEST_INFO에 로그데이터를 저장.
    # info는 Log Mode를 켜야만 작동이 되는 함수.
    global LATEST_INFO
    data = ts.request.get_json(force=True)
    if not data:
        return ts.jsonify({"error": "No JSON received"}), 400

    LATEST_INFO = data   # <- 추가   lidarRotation

    return ts.jsonify({"status": "success", "control": ""})