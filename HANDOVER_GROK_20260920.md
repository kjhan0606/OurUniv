# OurUniv CF4 — Grok 업무인수인계서

작성일: 2026-09-20 KST  
프로젝트 디렉터리: `/home/kjhan/BACKUP/CF4`  
호스트: `syntax` (Slurm 서버)  
Git 브랜치: `agent/freeze-zoom-pipeline`  
원격: `git@github.com:kjhan0606/OurUniv.git`

## 1. 최종 과학 목표

Cosmicflows-4 관측 제약으로 통계적으로 유효한 LCDM 초기조건을 만들고, 우주론적 규모에서 다음을 재현할 수 있는 zoom-in IC를 확보한다.

- Local Group의 MW, M31, M33 환경
- 주변의 Virgo/Coma 및 Local Void/Bootes Void 같은 대규모 구조
- Local Group 영역의 유효 분해능 `<= 0.3 cMpc/h`
- 그 밖의 영역은 약 `1–2 cMpc/h`까지 허용
- low-k는 CF4 posterior를 보존하고, high-k는 LCDM power spectrum 및 물리적 조건에 맞게 보완
- “그럴듯한 시각화”가 아니라 후속 RAMSES zoom 진화가 가능한 IC여야 함

## 2. 8단계 진행 현황

1. 관측 선택/완전성: 외부 ARES/CAMELS angular survival 검증 완료.
2. 접선속도 불확실성: 완료.
3. `z=0` 밀도 posterior: tracer calibration 및 holdout 검증 완료.
4. low-k + LCDM high-k + joint tracer/RSD: 외부 bias/FoG prior와 splice 수리 완료; holdout 통과.
5. IC 생성: GRAFIC parent IC 생성 및 preflight 통과.
6. PM/PMWD 전방 진화: uniform L8 parent의 preflight, nonlinear pilot, z=0 DMO run 완료.
7. 구조 재현: large-scale HOP은 통과했으나 uniform L8 입자질량이 약 `2.89e11 Msun/h`라 MW/M31 식별은 불가.
8. Zoom IC: trace-derived v3 생성 및 RAMSES 초기화/Poisson/AMR preflight 통과. 현재 이 단계의 후속 진화 검증은 선택적 진단으로 분리해야 함.

## 3. 확정된 주요 산출물

- Parent IC: `/gpfs/kjhan/CF4/kf_design/production_ic_grafic_v1`
  - `N=256`, box `384 cMpc/h`, cell `1.5 cMpc/h`, `a_start=0.02`
- Trace-derived zoom IC v3:
  - `/gpfs/kjhan/CF4/zoom/cf4_lg_zoom_pilot_v3`
  - IC levels L8–L12, runtime ceiling L19
  - L8 mask padding: base-L8 12 cells
- LG trace mask:
  - `/gpfs/kjhan/CF4/zoom/cf4_lg_trace_v1/lg_mask_l8.npz`
- Preflight decision:
  - `config/cf4_lg_zoom_pilot_preflight_decision_v2.json`
  - Slurm job `388636`
  - 경계 경고 0, fine-Poisson 경고 0, FFTW base solve 완료, 초기 refinement L14 진입
- Preflight binary:
  - `/home/kjhan/BACKUP/lagRamses-de-nonstd/build_lb_minimax_maxiter30/ramses_lb_minimax3d_maxiter30`
  - `maxiter_fine=30` 지원 빌드

## 4. 최근 판단과 주의점

기존 uniform L8 PM run은 zoom IC의 검증이 아니다. 반대로 zoom preflight는 IC 배선·초기 Poisson·AMR 안정성만 확인하며, `z=0`에서 MW/M31이 재현된다는 증거가 아니다.

따라서 다음을 명확히 구분한다.

- **IC 검증:** 현재 v3 preflight로 통과.
- **초기 비선형 진화 pilot:** 선택적 안정성 진단일 뿐, IC 확정의 필수 조건이 아님.
- **z=0 Local Group 구조 재현:** 아직 미완료이며, 충분한 질량분해능의 실제 zoom 진화와 halo/환경 분석이 필요.

`a=0.02 -> 0.05` bounded nonlinear pilot은 Slurm job `388638`로 제출된 상태였으나, 사용자가 중단/보류를 검토 중이다. 새 계산을 자동 제출하지 말고 먼저 상태와 필요성을 판단한다. 명시적 종료 지시 전에는 기존 작업을 임의로 죽이지 않는다.

## 5. Grok이 맡는 업무

Grok은 다음 묶음 단계의 **독립 기획 및 구현 감사**를 담당한다.

### Q-GOAL

각 제안 단계가 최종 목표에 직접 기여하는가?

- CF4 제약 보존
- low-k posterior와 LCDM high-k의 일관성
- Local Group `<=0.3 cMpc/h` zoom IC
- 후속 RAMSES 진화와 MW/M31/M33 및 환경 검증

최종 목표와 직접 연결되지 않는 분석·시각화·게이트·반복 계산은 제거 또는 연기한다.

### Q-LEAN

각 단계가 지금 꼭 필요한가?

- 이미 통과한 preflight를 반복하지 않는가?
- 단순 계측/로그/문서가 과도하게 커지지 않는가?
- parent uniform run을 zoom 검증처럼 중복 수행하지 않는가?
- z=0 구조 재현에 필요한 계산과 선택적 진단을 구분하는가?
- “검증을 위한 검증”이 본래 IC/구조 목표보다 커지지 않는가?

## 6. Grok 감사 출력 형식

각 묶음 계획은 다음 형식으로 반환한다.

1. 최대 3–5개 작업의 순서
2. 각 작업별 `Q-GOAL: GO/CONDITIONAL/NO-GO`
3. 각 작업별 `Q-LEAN: KEEP/DEFER/REMOVE`
4. 필수 증거와 중단 조건
5. 현재 실행 중인 선택적 pilot의 유지/중단/연기 판정
6. 최종 `GO / CONDITIONAL GO / NO-GO`

코드 감사는 파일/라인 근거를 제시하고 다음을 반드시 확인한다.

- v3 IC 경로와 level 슬롯 일치
- parent grid, mask padding, levelmin/levelmax와 물리적 분해능의 일관성
- binary가 `maxiter_fine`을 실제 지원하는지
- boundary/Poisson/NaN/출력 정책의 fail-closed 여부
- Slurm 메모리·시간·출력량 설정의 현실성

## 7. 안전 규칙

- 감사는 읽기 전용: 편집, 빌드, Slurm 제출, 시뮬레이션 실행 금지.
- 새 계산은 Grok 감사 결과와 운전자 판단 후에만 수행.
- 모든 수치 Python/NumPy/JAX 계산과 RAMSES 작업은 Slurm으로 제출.
- `pgrep`/무차별 프로세스 검색 금지.
- `/gpfs`는 정상 공유 파일시스템이며 별도 특수 기능을 구현하지 않는다.
- 기존 출력 삭제·덮어쓰기 금지.
- 본 handover와 관련된 판단은 Git에 기록하되, 사용자의 명시적 요구 없이 global shared context는 변경하지 않는다.

## 8. 현재 운전자에게 필요한 다음 판단

Grok은 먼저 “현재 v3 IC preflight 통과로 Stage 8의 IC 검증을 종료할 수 있는가”를 판단하고, 그 다음에만 선택적 nonlinear pilot 또는 z=0 zoom 진화의 필요성을 비교한다. 목적과 무관한 추가 계산은 NO-GO로 판정한다.
