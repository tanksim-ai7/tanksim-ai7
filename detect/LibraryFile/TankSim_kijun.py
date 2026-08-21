# library선언 및 변수 선언
from flask import Flask, request, jsonify
from ultralytics import YOLO
from pathlib import Path
from astar_planner import AStarPlanner, ObstacleRect

import os
import torch
import math
import uuid
import cv2
import numpy as np

app = Flask(__name__)
MODEL_PATH = Path("./models/best.yolov11s.pt")
model = YOLO(str(MODEL_PATH))

VERTICAL_FOV = 28.0  # deg, 기존과 동일 가정
HORIZONTAL_FOV_STEREO = 47.81061
LATEST_INFO = {}

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

#app = Flask(__name__)
#model = YOLO('best.pt')
#print(model.names)

# 오차값 줄이기 위한 변수 및 라이브러리
from collections import defaultdict, deque
import statistics as st

POSITION_HISTORY = defaultdict(lambda: deque(maxlen=5))   # 클래스별 최근 5개 관측 저장
OUTLIER_THRESHOLD = 15.0   # 중앙값에서 이 거리(m) 이상 벗어나면 이상치로 간주하고 제외
MIN_SAMPLES_BEFORE_OUTPUT = 3   # 이 개수만큼 쌓이기 전엔 값을 내보내지 않음

# 기능(함수) 모음 cell
# [NEW] 이미지 하나당 YOLO 추론을 딱 1번만 실행 (클래스별로 재추론하지 않음)
def run_inference(image_path):
    results = model(image_path, verbose=False)
    img_h, img_w = results[0].orig_shape                # 이미지의 가로값, 세로 값을 도출
                                                        # YOLO는 자체적으로 640x640으로 리사이징해서 처리함.
                                                        # offset을 구하는공식에서 640을 그대로 써버리면 값이 error
    detections = results[0].boxes.data.cpu().numpy()    # 이렇게 쓰면 YOLO가 도출한 값에 접근할수 있음
    return detections, img_w, img_h


# [NEW] 이미 뽑아둔 추론 결과(detections)에서 원하는 클래스만 골라내기 (재추론 없음)
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
    half_fov_rad = math.radians(fov_deg / 2)
    return math.degrees(math.atan(pixel_offset_ratio * math.tan(half_fov_rad)))


def get_focal_px(img_w, fov_deg=HORIZONTAL_FOV_STEREO):
    return (img_w / 2) / math.tan(math.radians(fov_deg / 2))

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
        return math.sqrt((p["x"]-med["x"])**2 + (p["y"]-med["y"])**2 + (p["z"]-med["z"])**2)

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
    if not left_pos or not right_pos or not left_rot:
        return None

    baseline = math.sqrt(
        (left_pos["x"] - right_pos["x"]) ** 2 +
        (left_pos["y"] - right_pos["y"]) ** 2 +
        (left_pos["z"] - right_pos["z"]) ** 2
    )

    cxL, cyL = bbox_center(left_bbox)
    cxR, _ = bbox_center(right_bbox)
    disparity = abs(cxL - cxR)
    if disparity < 1e-3:
        return None

    focal_px = get_focal_px(img_w)
    depth = baseline * focal_px / disparity

    h_offset = pixel_offset_to_angle((cxL - img_w / 2) / (img_w / 2), HORIZONTAL_FOV_STEREO)
    v_offset = pixel_offset_to_angle((cyL - img_h / 2) / (img_h / 2), VERTICAL_FOV)

    bearing = (left_rot["y"] + h_offset) % 360
    vertical = left_rot["x"] - v_offset   # 지난번 검증한 부호

    rad_h, rad_v = math.radians(bearing), math.radians(vertical)
    dx = depth * math.cos(rad_v) * math.sin(rad_h)
    dz = depth * math.cos(rad_v) * math.cos(rad_h)
    dy = depth * math.sin(rad_v)

    world_pos = {"x": left_pos["x"] + dx, "y": left_pos["y"] + dy, "z": left_pos["z"] + dz}

    player_pos = LATEST_INFO.get("playerPos")
    distance_3d = None
    if player_pos:
        distance_3d = math.sqrt(
            (player_pos["x"] - world_pos["x"]) ** 2 +
            (player_pos["y"] - world_pos["y"]) ** 2 +
            (player_pos["z"] - world_pos["z"]) ** 2
        )

    return {"world_pos": world_pos, "distance": distance_3d, "bearing": bearing}

def scan_all_objects(target_classes, left_path="temp_left.jpg", right_path="temp_right.jpg"):
    # target_classes: {class_id: class_name, ...}
    all_objects = []
    
    left_detections, img_w, img_h = run_inference(left_path)    # [NEW] 딱 1번만 추론
    right_detections, _, _ = run_inference(right_path)          # [NEW] 딱 1번만 추론
    
    for class_id, class_name in target_classes.items():
        left_boxes = filter_boxes_by_class(left_detections, class_id)     # [NEW] 재추론 없이 필터링만
        right_boxes = filter_boxes_by_class(right_detections, class_id)   # [NEW]
        
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
                continue
            result["raw_world_pos"] = result["world_pos"]   # 원본값도 참고용으로 남겨둠
            result["world_pos"] = smoothed_pos

            player_pos = LATEST_INFO.get("playerPos")
            if player_pos:
                result["distance"] = math.sqrt(
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



# 여기부터 서버 통신 함수
# def detect():
#     image = request.files.get('image')
#     if not image:
#         return jsonify({"error": "No image received"}), 400

#     image_path = 'temp_image.jpg'
#     image.save(image_path)

#     results = model(image_path, verbose=False)
#     detections = results[0].boxes.data.cpu().numpy()
#     #print(results[0].boxes.data)
#     #target_classes = {0: "human1",1: "human2"}
#     target_classes = {
#         0: 'Car', 
#         1: 'House', 
#         2: 'Human1', 
#         3: 'Human2', 
#         4: 'Human3', 
#         5: 'Mine', 
#         6: 'Rock', 
#         7: 'Tank1', 
#         8: 'Tank2', 
#         9: 'Tent', 
#         10: 'Tree', 
#         11: 'Wall'
#     }
#     filtered_results = []
#     for box in detections:
#         class_id = int(box[5])
#         if class_id in target_classes:
#             filtered_results.append({
#                 'className': target_classes[class_id],
#                 'bbox': [float(coord) for coord in box[:4]],
#                 'confidence': float(box[4]),
#                 'color': '#00FF00',
#                 'filled': False,
#                 'updateBoxWhileMoving': False
#             })

#     return jsonify(filtered_results)
    
def stereo_image():                             # 오브젝트 좌표, 위협도, 거리 계산은 다 여기서 실시.    
    global THREAT_PER                           # (위협도 관련)
    global DETECTED_OBJECTS_INFO
    
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    req_id = uuid.uuid4().hex   # [NEW] 요청마다 고유 ID
    left_path = f"temp_left_{req_id}.jpg"     # [NEW]
    right_path = f"temp_right_{req_id}.jpg"   # [NEW]
    left_image.save(left_path)
    right_image.save(right_path)

    #target_classes = {0: "human1", 1: "human2"}   # 나중에 실제 클래스로 확장
    target_classes = {
        0: 'Car', 
        1: 'House', 
        2: 'Human1', 
        3: 'Human2', 
        4: 'Human3', 
        5: 'Mine', 
        6: 'Rock', 
        7: 'Tank1', 
        8: 'Tank2', 
        9: 'Tent', 
        10: 'Tree', 
        11: 'Wall'
    }
    
    objects = scan_all_objects(target_classes, left_path, right_path)
    ranked = rank_objects_by_threat(objects)
    DETECTED_OBJECTS_INFO = save_detected_object_info(objects)
    total = total_threat_score(ranked)          # 눈(카메라)에 보이는 위협도의 총합 (위협도 관련)
    THREAT_PER = total                          # 이 end point에서 나온 위협도를 전역변수에 저장 (위협도 관련)
    print(DETECTED_OBJECTS_INFO)
    # print(f"[위험도 순위] 총 {len(ranked)}개 객체, 전체 위험도 합계: {total:.3f}")
    # for i, obj in enumerate(ranked, 1):
    #     print(f"  {i}순위 - {obj['class_name']}: 거리={obj['distance']:.1f}m, "
    #           f"위험도={obj['threat_score']:.3f}, 위치={obj['world_pos']}")

    # print(f"[스캔 결과] 총 {len(objects)}개 객체 탐지")
    # for obj in objects:
    #     print(f"  - {obj['class_name']}: 위치={obj['world_pos']}, 거리={obj['distance']:.1f}m")
    os.remove(left_path)
    os.remove(right_path)
    return jsonify({"result": "success"})
    
def info():              # 내 위치값, 회전값등을 가져와야하기 때문에 여기서 LATEST_INFO에 로그데이터를 저장.
    # info는 Log Mode를 켜야만 작동이 되는 함수.
    global LATEST_INFO
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    LATEST_INFO = data   # <- 추가   lidarRotation

    return jsonify({"status": "success", "control": ""})


def get_action():
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    if combined_commands:
        command = combined_commands.pop(0)
    else:
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }

    print("🔁 Sent Combined Action:", command)
    return jsonify(command)

def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400


def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    
    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})


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

#Endpoint called when the episode starts
def init():
    return jsonify(config)

def start():
    return jsonify({"control": ""})