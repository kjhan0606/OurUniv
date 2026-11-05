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

## 9. 프로젝트 히스토리 요약

### 초기 접근

- CF4 점 자료의 위치·특이속도를 이용해 `CF4 -> IC`를 직접 구성하려 했다.
- 초기 고분해능 보정은 high-k LCDM power를 왜곡할 위험이 있었고, 임의 amplitude/phase 보정은 통계적으로 방어하기 어려웠다.
- 이후 low-k는 관측 posterior로 유지하고 high-k는 LCDM power spectrum 및 물리적 조건으로 completion하는 방향으로 수정했다.

### 현재 밀도장·tracer 경로

- CF4 tracer bias, RSD/FoG, Carrick reference field를 분리해 calibration/holdout을 수행했다.
- `z=0` posterior, low/high-k splice, joint tracer calibration을 단계별 decision 파일로 기록했다.
- 이 결과는 대규모·저주파 제약을 제공하지만, CF4의 소수 LG 점 자료만으로 MW/M31/M33의 개별 halo를 직접 결정하지는 않는다.

### PM/HOP 검증

- uniform L8 parent GRAFIC IC를 RAMSES PM으로 진화하고 z=0 HOP을 수행했다.
- HOP clean rebuild 후 large-scale catalog는 정상화되었다.
- 그러나 parent 입자질량이 약 `2.89e11 Msun/h`라 MW/M31은 수 입자 수준이다. 따라서 “large-scale HOP pass”와 “Local Group halo reproduction”을 분리 판정했다.

### Zoom 전환

- z=0의 CF4 LG anchor 주변 입자를 tracing하여 Lagrangian mask를 만들었다.
- 첫 narrow mask(v2)는 경계 및 fine-Poisson 경고가 발생해 폐기/비확정 처리했다.
- mask padding을 L8 기준 12 cells로 넓힌 v3 IC를 생성했다.
- 새 lagRamses binary에 `maxiter_fine` 지원을 포함해 preflight를 재실행했고, job 388636에서 경계 0, Poisson 0으로 통과했다.

## 10. 현재 상황의 정확한 판정

- **확정:** low-k posterior/calibration, parent IC, parent PM/HOP 대규모 검증, trace-derived v3 zoom IC 생성, v3 초기 RAMSES preflight.
- **미확정:** v3 zoom을 실제로 z=0까지 진화했을 때 MW/M31/M33 및 주변 환경이 관측과 맞는지.
- **미확정:** CF4 점 자료만으로 LG high-k의 실제 위상/halo pair를 유일하게 결정할 수 있는지.
- **현재 핵심 병목:** 관측으로 직접 결정되지 않는 LG high-k를 LCDM prior와 zoom 실행 조건으로 어떻게 제한하고, 어떤 관측량으로 검증할지.
- **선택적 작업:** job 388638의 `a=0.02 -> 0.05` bounded nonlinear pilot. 이것은 안정성 진단이지 최종 구조 재현 증거가 아니다.

## 11. 앞으로 해야 할 일

### 반드시 해야 할 일

1. v3 IC의 provenance와 preflight 결과를 보존하고, IC level/parent/mask/해상도 계산을 최종 문서화한다.
2. Local Group 영역의 목표 질량 해상도와 contamination/zoom volume 기준을 명시한다.
3. 실제 z=0 zoom 진화가 필요하다고 판정될 경우에만, 최소 출력 정책으로 제한된 production 설계를 만든다.
4. z=0 진화 후 HOP/halo finder에서 MW/M31/M33 후보, pair separation/relative velocity, Local Void·Virgo 방향 환경을 평가한다.
5. 여러 high-k realization 또는 seed ensemble을 사용할 경우, 단일 성공 사례가 아니라 LCDM prior와 관측 holdout을 함께 보고한다.

### 조건부 또는 연기할 일

- job 388638 bounded nonlinear pilot: Grok의 Q-LEAN 판정 전에는 추가 제출·확장하지 않는다.
- 장시간 z=0 zoom production: contamination, 메모리, 출력량, 목표 관측량의 사전 검토 없이는 시작하지 않는다.
- hydrodynamics/stellar·AGN feedback: 현재 DMO/IC 검증의 병목을 해결하기 전에는 연기한다.
- 새 ML 학습, 새 external calibration, 대규모 시각화: 최종 목적에 직접 기여하는 경우에만 계획에 포함한다.

## 12. 중요한 메모

- “초기조건 preflight 통과”는 “현재 우주 구조 재현”이 아니다.
- uniform L8 HOP 결과로 MW/M31 재현을 주장하지 않는다.
- CF4 LG 점 자료는 anchor/velocity 제약이지, 개별 은하의 완전한 고-k density field가 아니다.
- high-k를 임의로 증폭하거나 random phase로 교체하면 LCDM 통계와 관측 posterior의 의미가 바뀔 수 있다.
- `0.3 cMpc/h`는 전체 상자 해상도가 아니라 Local Group zoom 영역의 유효 해상도 목표다.
- Slurm job은 메모리 최대 예상치에 약 20% 여유를 둔다.
- GPFS는 `/home`과 같은 공유 파일시스템이다. GPFS 특수 기능을 구현하거나 별도 검증 대상으로 만들지 않는다.
- 모든 감사는 읽기 전용이며, 감사자가 직접 코드·잡·시뮬레이션을 변경하지 않는다.
- 새 단계는 Q-GOAL과 Q-LEAN을 먼저 통과해야 하며, 불필요한 검증 단계가 본 계산보다 커지지 않게 한다.
