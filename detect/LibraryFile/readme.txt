4파일을 같은곳에 넣어주시고, 모델은 파일이 있는곳/models/best.yolov11s.pt 로 넣으시면 됩니다.

26-08-21
TankSim_Server, TankSim파일 삭제. 변수등은 각자 injee, kijun파일이 각자 가지고 있게 세팅하였음.
그에따라 ally-controller.py에서 import TankSim as ts 삭제. 기존 파일과의 충돌을 대비하기 위해 detect/Library 폴더에 ally-controller.py 넣어놨음.
기준 : stereo_image는 내가 작업 했으니 현재와 맞게 돌릴수 있는데 인지의 detect의 기능은 조금더 살펴봐야 bbox를 화면에 띄울수 있을거 같음. 아니면 인지에게 직접 물어봐야 함.