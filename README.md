# tanksim-ai7
아군/적군 탱크 유닛 시뮬레이션과, 서버 측 상태를 실시간으로 확인할 수 있는 모니터링 대시보드를 포함하는 프로젝트 저장소입니다.

### - ally-controller.py : 아군 전차 제어 코드
### - enemy-controller.py : 적군 전차 제어 코드



## Git 운영 규칙

- 아래 규칙을 꼭 확인하시고 지켜주시면 감사하겠습니다.
- 문제가 발생했을 때는 git 담당자에게 문의 주시면 감사하겠습니다.

### 1. Branch 생성 규칙

- 각 조별 main branch에서 개인별 branch를 추가하여 작업할 것
  - 예) 자율주행팀 : `auto/main` branch에서 `auto/a_star` : 팀명/기능
- `origin/main`에는 직접 새로운 branch를 만들지 말 것
- 조별 main branch에 병합 완료된 개인 branch는 삭제할 것

### 2. Commit 규칙

모든 코드는 commit 전 아래 내용을 확인할 것

1. 코드에 오류가 없는지 확인할 것 (AI를 활용한 코드 검증 권장)
2. Commit 시에는 항상 커밋 메시지를 작성할 것
   - 첫 줄: 추가/수정한 기능에 대한 핵심 요약
   - 다음 줄부터: 추가/수정한 내용에 대한 상세 사항 기입 (다른 사람이 커밋 메시지만 보고도 내용을 파악할 수 있도록 작성)
   - 예시:
```
feat: A* 경로탐색 알고리즘 추가

 - 격자맵 기반 A* pathfinding 구현
 - 장애물 회피 로직 포함
 - 기존 Dijkstra 대비 처리 속도 개선
```

- 소스트리 기준 첫 줄은 커밋 목록에 요약으로 표시되고, 빈 줄 이후 작성한 본문은 상세보기(설명란)에 전체 표시됨
3. 개인 branch에서 작업한 내용을 조별 main branch에 병합하기 전, 각 조 git 담당자의 허가를 받을 것
4. 하나의 커밋에는 하나의 작업 단위만 담을 것 (여러 기능을 한 커밋에 몰아넣지 말고 기능/수정 단위로 나눠서 커밋)

### 3. origin/main 병합 규칙

`origin/main`에 병합하기 전 아래 과정을 반드시 거칠 것

1. 코드 내용에 오류가 없는지 최종 확인할 것 (AI 사용 권장) — 자잘한 오류가 쌓이면 추후 문제가 커질 수 있음
2. 조별 main branch에 `origin/main`으로 병합할 코드 내용을 올려둘 것
3. 조별 main branch(작업 branch)로 이동한 상태에서 `origin/main`을 병합하여 충돌 여부를 먼저 확인할 것
   - 충돌이 있다면 해결 후 새로운 commit을 생성
   - 충돌 해결이 애매하거나 어떤 코드를 남길지 판단이 서지 않는 경우, 임의로 처리하지 말고 반드시 해당 조 git 담당자와 상의할 것
4. 조별 main branch → `origin/main`은 Pull Request를 통해 진행하며, git 담당자 확인 후 병합한다
5. PR 작성 시 제목에는 작업 내용을 간단히, 본문에는 변경 사항과 확인이 필요한 부분을 함께 적을 것

### 4. Branch 네이밍 규칙

각 조별 main branch:

- 자율주행팀: `auto/main`
- 오브젝트 디텍션팀: `detect/main`
- 터렛 제어팀: `turret/main`

개인 작업 branch는 조별 main branch 하위에 작업 내용을 알 수 있는 이름으로 생성:
```
main                (건들지 않음, 최종 병합 대상)
├─ auto/main        ← 자율주행팀 조별 main branch
│   └─ auto/a_star  ← 개인 작업 branch
├─ detect/main       ← 오브젝트 디텍션팀 조별 main branch
│   └─ detect/YOLO   ← 개인 작업 branch
└─ turret/main       ← 터렛 제어팀 조별 main branch
    └─ turret/Calc   ← 개인 작업 branch
```

### 5. 커밋 제외 대상 (.gitignore)

아래 항목들은 git으로 직접 관리하지 않으며, `.gitignore`에 등록하여 커밋되지 않도록 할 것

- YOLO 가중치 파일 등 학습된 모델 파일 (`*.pt`, `*.pth`, `*.h5` 등)
- 가상환경 및 캐시 폴더 (`__pycache__`, `.venv`, `venv`, `.ipynb_checkpoints` 등)
- Unity 빌드 결과물, 라이브러리/패키지 폴더 (`Library/`, `Temp/`, `Build/` 등)
- 개인 API 키, 설정 파일 등 민감 정보 (`.env` 등)
- 대용량 모델 파일이 필요한 경우 Git LFS 사용을 원칙으로 하며, 사용 전 git 담당자와 상의할 것

### 6. Force Push 관련 규칙

- 공용 branch(`main`, 각 조별 `main`)에는 `git push --force` (강제 push)를 임의로 진행하지 말 것
- Force push가 반드시 필요한 상황이라면 사전에 반드시 git 담당자에게 문의 후 진행할 것
