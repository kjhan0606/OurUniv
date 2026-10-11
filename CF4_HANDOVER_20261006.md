# OurUniv / CF4 업무인수인계서

작성 기준: 2026-10-06 (KST). 실행 상태는 `CF4_HANDOVER_20261010.md`가 이어 받는다. 이 파일은 그때까지의 역사다.

저장소: `/home/kjhan/BACKUP/CF4` · 브랜치 `agent/freeze-zoom-pipeline`
기준 계획: `CF4_MASTER_PLAN.md`, `CF4_END_TO_END_REPLAN_20260913.md`

## 1. 프로젝트가 달성하려는 것

실제 CF4 및 주변 은하 관측에 조건부인, 동역학적으로 가능한 국부우주의 현재 상태와 그 상태를 만드는 초기조건(IC)의 앙상블을 얻는다. 우선 전달할 과학 결과는 **z=0 현재 밀도·속도장과 그 불확실성**이다. 내부 추론은 잠재 IC를 LCDM 초기조건으로 두고 중력으로 진화시켜 관측과 현재장을 공동으로 연결한다. 따라서 “IC를 먼저 만들고 나중에 z=0을 기대한다”가 아니며, 독립 ML 지도를 시간 역적분하는 경로도 아니다.

상자 전체의 환경 지도 목표 격자는 384 cMpc/h 상자에서 N256, 즉 명목 1.5 cMpc/h다. 관측 정보가 실제로 보장하는 해상도와 격자 간격은 구분한다. 국부은하군(LG)은 후속 줌에서 0.3 cMpc/h 이하를 목표로 하고, 그 밖의 영역은 약 1–2 cMpc/h로 충분할 수 있다. 최종 목적은 MW, M31, M33 및 Virgo/Coma와 Local/Boötes Void 등 주변 환경을 일관된 우주론적 맥락에서 재현하는 것이다. 다만 MW/M31의 성분 역할은 아직 모호하고 M33는 식별·결합 상태가 미해결이다. 세 성분의 관측량은 모두 **같은 새로 생성되고 진화한 LG 장**을 제약해야 한다. 원자료의 정답 ID는 보정·평가에만 쓰며 후보장을 seed하거나 선택하는 데 쓰지 않는다.

고-k 위상은 관측되지 않은 자유도로서 LCDM 조건부 사전분포에 남는다. 임의 고-k 증폭/위상 난수, 임의 low/high-k 경계 이동, 가장 잘 맞는 seed 하나의 선택으로 물리적 제약을 대신하지 않는다. 지도 해상도, 입자 간격, 헤일로 질량·힘 분해능은 서로 다른 양이다.

## 2. 경로가 여기까지 바뀐 이유

초기에는 CF4에서 곧바로 IC를 만들거나, Hong 계열/TNG 학습으로 현재장을 만들거나, 지도와 IC를 별도로 역변환하는 여러 방향을 탐색했다. 그 과정에서 데이터의 제한된 관측 정보를 고분해능 구조의 복원으로 과대해석했고, 반복 학습·작은 표본 통과율을 과학적 재현 진척으로 간주한 문제가 드러났다. TNG/Hong 접근은 현재 프로젝트의 실제 CF4 조건부 posterior를 대신하지 못한다.

2026-09-13에 사용자가 승인한 상위 경로는 **잠재 IC와 중력 진화를 포함한 공동 추론**으로 정리됐다. z=0 장은 첫 과학 전달물이지만, 이를 뒷받침할 IC·전방모형은 추론 안에서 함께 다룬다. 2026-10-06 현재도 이 경로가 유효하며, 과거의 독립-z=0-first 또는 임의 고-k 생성 계획을 되살리지 않는다.

계획서의 R1–R5는 큰 결과물 묶음이다: R1 물리 연결·비용 확인, R2 실제 CF4 환경의 z=0 장, R3 MW/M31/M33 공동 LG 제약, R4 정밀 solver 전방검증, R5 위상 연결된 줌 IC 전달. 현재는 **R2**다. R1의 일부 역학/미분·운영 기반은 마련됐지만 전체 목표 완료로 해석하지 않는다.

## 3. 현재까지의 상태와 중요한 한계

- R2의 활성 관측대상은 N256/384 cMpc/h 상자(1.5 cMpc/h 격자)의 실제 CF4/2M++ 조건부 z=0 장이다. 활성 v6 코호트에는 47,121개의 2M++ training count와 1,414개의 연결된 CF4 raw-FP mark가 있다.
- 이 1,414개는 관측된 count-point와 mark가 주어졌다는 조건 아래의 제한된 conditional-mark 추정량을 만들 수 있을 뿐이다. **CF4 그룹 포함 확률, redshift 성공/생존 선택, tracer bias 및 다중 구성원 공분산을 완전히 보정한 전체 조사 likelihood가 아니다.** 따라서 완전 보정된 z=0 posterior/production map은 아직 NO-GO다.
- 2026-10-05의 conditional N256 MAP 진단(job 413612)은 6회 exact evaluation, 4회 optimizer iteration 후 max-iteration 한도에 도달했다. 목적함수는 약 0.362% 개선됐지만 종료 gradient 무한노름은 약 13,525로, 계획된 1e-4 수렴 조건과 큰 차이가 있다. **수렴 MAP, posterior 표본, production IC 또는 LG 결과가 아니다.** 동일한 낮은 반복 한도를 무작정 재실행하지 않는다. 상세: `CF4_R2_CONDITIONAL_MAP_20261005.md`.
- v6 angular 현장 연산자의 2026-10-06 ray-volume 비교(job 414452)는 7 geometry, 5 mask, 15 shell-CDF 테스트를 통과했다. 선택된 12개 기하 제어점에서 active GL2와 refined parent의 intensity 차이는 최대 0.21%였지만, native angular map 대비 parent 차이는 -0.77%~+11.42%였다. 전체 likelihood 변화나 모든 population/field에 대한 수렴 증명이 아니며, production 연산자를 바꾸거나 field fit을 정당화하지 않는다. 상세: `CF4_R2_ACTIVE_V6_RAY_VOLUME_REFERENCE_20261006.md`.
- SDSS-PV 2,048 mock ensemble은 선택된 host의 FP 잔차/풍부도 효과를 진단했다. 이는 선택 보정의 일부이지 CF4 그룹 포함법·공분산 또는 전체 관측법의 교정이 아니다. 상세 결정은 `CF4_MASTER_PLAN.md`의 2026-10-05 항목과 연결 문서를 참조한다.
- CF4TF/MDPL2/Hollinger-Hudson mock 자료는 현재 필요한 전체 CF4 selection denominator나 N256 matter truth를 제공한다고 확인되지 않았다. COSMOSIM MDPL2 원시 입자 snapshot은 약 1.716 TB이고 제한된 cutout 방식도 입증되지 않았다. 대용량 다운로드, 계정 등록, API token 취득은 진행하지 않는다. 공개 SMDPL z=0 밀도표도 단독으로 tracer-law 교정에 충분하지 않다고 판정됐다.
- 집계 가능한 현재 결과는 제한된 operator mechanics와 conditional diagnostics다. held-out 결과는 아직 과학 판정에 쓰지 않았고, 실제 z=0 posterior/map은 완성되지 않았다. 이 제한을 “데이터가 충분하다” 또는 “진화하면 맞을 것”으로 바꿔 말하지 않는다.

관련 최근 기록:

- `CF4_R2_CF4TF_MOCK_APPLICABILITY_20261006.md`
- `CF4_R2_ACTIVE_V6_RAY_VOLUME_REFERENCE_20261006.md`
- `CF4_R2_CONDITIONAL_MAP_20261005.md`
- `CF4_R2_SELECTED_MARK_DENOMINATOR_AUDIT_20261005.md`
- `CF4_R2_COSMOSIM_MDPL2_ACCESS_20261005.md`

## 4. 다음 우선순위

1. **R2의 동일 conditional target을 먼저 안정화한다.** 다음 N256 fit 전에 현 optimizer의 max-iteration 종료가 아니라 효율적인 수렴 경로, gradient/evaluation 예산, 중단·체크포인트 규칙을 코드와 실제 비용에 맞춰 설계한다. 같은 objective의 항별 재현과 수치 안정성은 확인하되 새 prior나 임의 likelihood 항을 넣지 않는다.
2. 관측법의 실제 공백을 해결할 수 있는지 평가한다. 필요한 것은 CF4 그룹 포함·성공 선택과 bias/RSD/FoG, 그룹/구성원 연결 및 공분산에 대한 출처가 확인된 보정 자료 또는 물리적으로 검증된 mock이다. 자료가 없으면 추정해 채우지 말고, conditional field 산출물과 완전 posterior의 차이를 명시한다.
3. conditional map/optimum을 개선하더라도 미수렴·비정상 상태의 결과를 posterior로 부르지 않는다. posterior 단계로 넘어갈 때는 독립성/혼합(ESS), 수렴, prior predictive 및 사전 고정한 untouched held-out 평가를 갖춘다. 필요하면 held-out 평가 계획을 먼저 고정하고 결과를 열람한다.
4. R2의 현 장과 불확실성이 유효하게 확보된 뒤 R3에서 LG 관측을 같은 latent/dynamical field에 연결한다. MW/M31 역할 모호성, M33 bound/substructure 판정 및 미해결 구성원을 명시한다. 이후 R4 정밀 전방검증, R5 라그랑지안 mask 기반 줌 IC와 오염·수렴 검증 순으로 간다.

지금은 RAMSES/별도 비선형 중력진화를 새로 돌려 지도를 “만드는” 단계가 아니다. PMWD 전방진화는 IC-현재장 공동 target의 일부로 이미 계산되며, 대형 별도 시뮬레이션은 검증 질문과 자원 근거가 있을 때만 사용한다. 광범위한 데이터 재탐색, 전체 해상도 ladder, 반복 quadrature ladder, heldout 열람, 기존 target에 대한 sampler 재시작을 자동으로 추가하지 않는다.

## 5. 운영·업무분장·안전 메모

- **운전자(Codex 세션):** 과학 설계의 일상 판단, 코드 구현, 검사, Slurm 제출, 결과 해석, 결정 기록, 커밋/푸시를 맡는다. 실제 코드/결과를 직접 확인하며 routine bundle마다 외부 감사를 반복하지 않는다.
- **중요 발견·목표 변경·대형 계산의 외부 자문:** Fable5가 우선이며 사용 불가 시 Astra가 백업이다. Q-GOAL(최종 CF4/LG 목적 기여), Q-LEAN(필요성·과도한 계측/게이트 여부)을 명시한다. 감사가 자문이면 결과를 근거와 대조해 운전자가 채택/수정/기각하고 이유를 남긴다. 불리한 과학 결론을 도구 실패로 치환하지 않는다. Grok/AGY/Opus를 상시 감사자로 두지 않는다.
- 큰 문제가 없으면 승인 대기 없이 묶음 단위로 연속 진행한다. 목표 변경, 새 외부 권한, 중대한 계산 또는 범위 밖 조치만 사용자 판단이 필요하다.
- **Syntax는 Slurm 로그인 노드**다. 수동 GPU 노드 실행은 금지. GPU 계산은 H200을 먼저 확인하고 비어 있으면 `--partition=h200 --gres=gpu:H200:1`을 우선한다. 점유 중이면 허용된 H100/A100 모드를 확인해 하나를 선택하고 partition/typed GRES를 기록한다. 예상 최대 메모리보다 최소 20% 여유를 요청한다. 현재 이 문서 작성 때 계정 queue에는 여러 다른 프로젝트의 실행/대기 작업이 있었고, CF4로 명확히 식별되는 job은 보이지 않았다. 다른 작업은 건드리지 않는다.
- `pgrep`/프로세스 스캔 반복문으로 모니터링하지 않는다. Slurm 상태는 필요한 때 단발성으로 확인한다. /gpfs는 정상 공유 저장소이며 inode/파일시스템 진단 대상으로 삼지 않는다. 재귀적 전체 파일시스템 검색을 하지 않는다.
- COSMOSIM 자격증명, API key, 비밀번호를 이 문서나 로그에 복사하지 않는다. 이메일을 보내지 않는다. 사용자 계정으로의 제한된 README 조회는 승인된 metadata 확인에 한정되며, 대용량 데이터 다운로드/등록 권한을 뜻하지 않는다.
- 실패 산출물과 관련 없는 작업물은 보존한다. 특히 현재 Git 상태에서 확인된 다음 untracked 사용자 파일은 이번 변경에 포함하거나 수정/삭제하지 않는다: `config/cf4_q0_gh_boundary_diagnostic_bundle_v1.json`, `config/grok_bundle_plan_audit_prompt_20260920.md`, `config/grok_code_audit_prompt_20260920.md`, `scripts/tripwire/`.

## 6. 다음 세션 시작 체크리스트

1. `/home/kjhan/.codex/SHARED_CONTEXT.md`를 전부 읽고, 이 전달문서와 `CF4_MASTER_PLAN.md`, `CF4_END_TO_END_REPLAN_20260913.md`를 읽는다.
2. 현재 디렉터리와 저장소/브랜치를 확인한다. 작업 전 `git status --short --branch`로 위 untracked 자료를 포함한 기존 변경을 보존한다.
3. Slurm queue를 한 번 확인해 새 실행 여부를 구분한다. 오래된 로그나 이름만 보고 현재 실행 중이라고 추정하지 않는다.
4. R2 목표를 유지하며 다음 conditional target optimizer 개선을 코드 근거와 함께 설계한다. 먼저 R2의 관측 likelihood 한계를 해결 또는 명시하지 않은 채 full posterior, LG 재현, 줌 IC를 완료했다고 선언하지 않는다.

저장소 기준 HEAD는 `05659b6` (`Record CF4TF mock availability and MAP stop audit`)이며, 해당 시점까지 origin에 push되어 있다. 이 전달문서는 이후 별도 변경으로 기록한다.
