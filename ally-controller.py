from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.dstar_lite_planner_cost import ObstacleRect as DStarLiteObstacleRect
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
model = YOLO('yolov8n.pt')
print(model.names)

@app.route("/detect", methods=["POST"])
def detect():

    image_file = request.files.get("image")

    if image_file is None:
        return jsonify({
            "error": "No image received"
        }), 400

    try:
        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({
                "error": "Empty image received"
            }), 400

        encoded_image = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            encoded_image,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Invalid image data"
            }), 400


        # ============================
        # YOLO11s 객체 탐지
        # ============================

        result = model.predict(
            source=frame,
            imgsz=1280,
            conf=0.15,
            iou=0.45,
            device=DEVICE,
            verbose=False,
            max_det=100
        )[0]


        detections = []


        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                    .cpu()
                    .item()
                )

                class_name = model.names[
                    class_id
                ]

                confidence = float(
                    box.conf[0]
                    .cpu()
                    .item()
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )


                center_x = (
                    x1 + x2
                ) / 2

                center_y = (
                    y1 + y2
                ) / 2


                detection = {

                    "className": class_name,

                    "classId": class_id,

                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ],

                    "confidence": confidence,

                    "center": [
                        float(center_x),
                        float(center_y)
                    ],

                    "color": "#00FF00",

                    "filled": False,

                    "updateBoxWhileMoving": False
                }


                detections.append(
                    detection
                )


        print(
            f"Detection count: "
            f"{len(detections)}"
        )

        for d in detections:
            print(
                d["className"],
                round(
                    d["confidence"],
                    3
                )
            )


        return jsonify(
            detections
        )


    except Exception as e:

        print(
            "Detection Error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500

@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    left_path = "temp_left.jpg"
    right_path = "temp_right.jpg"

    try:
        left_image.save(left_path)
        right_image.save(right_path)
    except Exception as e:
        return jsonify({"result": "error", "message": str(e)}), 500

    return jsonify({"result": "success"})
    
@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    global all_info
    all_info = data

    return jsonify({"status": "success", "control": ""})

@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    # print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    # print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    if path_idx >= len(path):
        return jsonify({"moveAD": {"command": "", "weight": 0}, "moveWS": {"command": "", "weight": 0}})

    # print("🔁 Sent Combined Action:", command)
    return jsonify({"moveAD": {"command": "", "weight": 0}, "moveWS": {"command": "", "weight": 0}})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))

        global dest
        dest = (x, y, z)

        # print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    path_planner.update_dstar_obstacles_from_payload(data)

    if path_flag:
        global path, path_idx
        path = [] # path 초기화
        path_idx = 0 # path idx 초기화
        path = path_planner.find_path((all_info['playerPos']['x'], all_info['playerPos']['z']), (dest[0], dest[2]))
        path_planner.plot(path, save_path='terrain_map')

    # print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
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
@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": False,
        "detectMode": False,
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit" : True
    }
    global path, path_idx, path_flag

    path_planner.set_risk_layers()
    path = path_planner.find_path((60, 27.23), (dest[0], dest[2]))
    path_planner.plot(path, save_path='terrain_map')

    path_idx = 0
    path_flag = True
    
    # print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    # print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
