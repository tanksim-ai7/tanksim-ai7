0. tank challenge 폴더의 lidar data들어있는 폴더를 정리(기존 데이터들 삭제)

1. 모든 파일을 한 폴더에 넣는다.

2. tank challenge 에서 Tracking Mode, Log Mode, Save LIDAR Data 활성화.

3. Properties에서 Lidar Setting (Max Distance -> 150 변경, Send Detected Lidar 체크)

4. 폴더에서 tank_server.py 실행.

5. localhost:5000/patrol 을 브라우저에 입력. 

6. 스캔현황을 누르고 촘촘히(12줄)로 변경. (기본도 괜찮긴한데 더 자세한 데이터를 위해)

7. 스캔이 끝나면 tank challenge 폴더의 lidar data 저장된 폴더(csv, json있는 폴더) 들어가서 lidar_terrain.py 를 이 폴더에 넣고 실행.

8. 실행 후 만들어진 .npy 파일을 numpy로 시각화.
