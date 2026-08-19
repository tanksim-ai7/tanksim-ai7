import TankSim as ts

planner = ts.AStarPlanner(
    grid_min_x=0.0,
    grid_max_x=300.0,
    grid_min_z=0.0,
    grid_max_z=300.0,
    cell_size=1.0,
    obstacle_margin=2.0,
    allow_diagonal=True,
)

    # 1) /update_obstacle 에서 호출
def update_obstacles_from_payload(payload: dict):
    obs_list = []
    for item in payload.get("obstacles", []):
        obs = ts.ObstacleRect.from_min_max(
            x_min=item["x_min"],
            x_max=item["x_max"],
            z_min=item["z_min"],
            z_max=item["z_max"],
        )
        obs_list.append(obs)
        planner.set_obstacles(obs_list)


if not ts.MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO 모델 파일을 찾을 수 없습니다: {ts.MODEL_PATH.resolve()}"
    )

# GPU가 있으면 GPU 사용
DEVICE = 0 if ts.torch.cuda.is_available() else "cpu"

print("=" * 60)
print("✅ YOLO 모델 로드 완료")
print(f"📦 Model : {ts.MODEL_PATH}")
print(f"🖥 Device : {DEVICE}")
print(f"🎯 Classes : {ts.model.names}")
print("=" * 60)

dest = None
current_pos = None

# Tank Challenge가 실제로 전송한 화면과 YOLO 결과를 저장합니다.
#DEBUG_IMAGE_DIR = BASE_DIR / "ClassIntegration_Dataset"
#DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_DEBUG_IMAGES = True
DEBUG_SAVE_LIMIT = 30

detect_request_count = 0
last_detections = []

dest = None
current_pos = None

#@app.route("/detect", methods=["POST"])
def detect():
    image_file = ts.request.files.get("image")
    if image_file is None:
        return ts.jsonify({
            "error": "No image received"
        }), 400
    try:
        image_bytes = image_file.read()

        if not image_bytes:
            return ts.jsonify({
                "error": "Empty image received"
            }), 400
        encoded_image = ts.np.frombuffer(
            image_bytes,
            dtype=ts.np.uint8
        )
        frame = ts.cv2.imdecode(
            encoded_image,
            ts.cv2.IMREAD_COLOR
        )
        if frame is None:
            return ts.jsonify({
                "error": "Invalid image data"
            }), 400


        # ============================
        # YOLO11s 객체 탐지
        # ============================
        result = ts.model.predict(
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

                class_name = ts.model.names[
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

        return ts.jsonify(
            detections
        )


    except Exception as e:

        print(
            "Detection Error:",
            e
        )

        return ts.jsonify({
            "error": str(e)
        }), 500

#@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    left_image = ts.request.files.get('left_image')
    right_image = ts.request.files.get('right_image')

    if not left_image or not right_image:
        return ts.jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    left_path = "temp_left.jpg"
    right_path = "temp_right.jpg"

    try:
        left_image.save(left_path)
        right_image.save(right_path)
    except Exception as e:
        return ts.jsonify({"result": "error", "message": str(e)}), 500

    return ts.jsonify({"result": "success"})
    
#@app.route('/info', methods=['POST'])
def info():
    data = ts.request.get_json(force=True)
    if not data:
        return ts.jsonify({"error": "No JSON received"}), 400

    return ts.jsonify({"status": "success", "control": ""})

#@app.route('/get_action', methods=['POST'])
def get_action():
    global current_pos
    data = ts.request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    current_pos = [pos_x, pos_z]
    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    # if combined_commands:
    #     command = combined_commands.pop(0)
    # else:
    #     command = {
    #         "moveWS": {"command": "STOP", "weight": 1.0},
    #         "moveAD": {"command": "", "weight": 0.0},
    #         "turretQE": {"command": "", "weight": 0.0},
    #         "turretRF": {"command": "", "weight": 0.0},
    #         "fire": False
    #     }
    if pos_z > 150:
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "E", "weight": 0.5},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": True
            }
    else:
        command = {
            "moveWS": {"command": "W", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.3},
            "turretQE": {"command": "", "weight": 0.5},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": True
            }
    
    
    print("🔁 Sent Combined Action:", command)
    return ts.jsonify(command)

#@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = ts.request.get_json()
    if not data:
        return ts.jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return ts.jsonify({"status": "OK", "message": "Bullet impact data received"})

#@app.route('/set_destination', methods=['POST'])
def set_destination():
    global dest, current_pos, planner
    
    data = ts.request.get_json()
    if not data or "destination" not in data:
        return ts.jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")

        dest = [x,z]
        print(current_pos, dest)
        path = planner.find_path(current_pos, dest)
        planner.plot(path = path, show_grid=True, title="A* Demo (300x300)")
        
        return ts.jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return ts.jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

#@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global planner
    data = ts.request.get_json()
    if not data:
        return ts.jsonify({'status': 'error', 'message': 'No data received'}), 400

    update_obstacles_from_payload(data)
    planner.plot()
    print("🪨 Obstacle Data:", data)
    return ts.jsonify({'status': 'success', 'message': 'Obstacle data received'})

#@app.route('/collision', methods=['POST']) 
def collision():
    data = ts.request.get_json()
    if not data:
        return ts.jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return ts.jsonify({'status': 'success', 'message': 'Collision data received'})

    
#Endpoint called when the episode starts
def init():
    return ts.jsonify(config)

def start():
    return ts.jsonify({"control": ""})

if __name__ == '__main__':
    ts.app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False
    )