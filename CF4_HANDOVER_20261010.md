# OurUniv / CF4 업무인수인계서

작성 기준: 2026-10-10 23:41 KST

저장소: `/home/kjhan/BACKUP/CF4` · 브랜치 `agent/freeze-zoom-pipeline`
과학 기준 커밋: `a7b81a7fb7578640a74729730cfdc42ab01abf8e`
`Record the reduced population-5 secant and queue one full gradient.`
Job 418601은 이 커밋을 요구한다. 이 인수인계서는 그 다음 문서 커밋이다. 다섯 과학 경로는 `a7b81a7`과 같다. 그 커밋을 amend하지 않는다.

이 문서는 `CF4_HANDOVER_20261006.md`의 실행 상태를 이어 받는다. 2026-10-06 문서의 과학 목표와 데이터 한계는 그대로다. 활성 계획의 긴 좌표 기록은 `CF4_MASTER_PLAN.md`의 2026-10-06 continuation과 `CF4_R2_CONDITIONAL_OPTIMIZER_DESIGN_20261006.md`에 있다. 종료 절차는 `CF4_R2_EXIT_AUDIT_GATE_20261006.md`다.

## 1. 지금 위치

현재는 **R2**다. 활성 대상은 N256, 384 cMpc/h 상자의 conditional v6 목적함수다. IC 좌표는 고정이다. Optimizer는 호출하지 않는다. Held-out은 열지 않는다. `longer_warm_start_authorized`는 false다. 18회 IC warm start는 승인되지 않았고, 하려면 종료 감사와 별개의 Astra 감사가 필요하다.

받아들여진 최신 상태는 job 418557의 population-5 secant다. 상태 `CONDITIONAL_POP5_REVISIT5_REDUCED`. 목적함수는 8208466.393512041에서 8208466.323127546로 내려갔다. Population 5 white는 0.0784613121345734에서 0.07937656860297954로 움직였고, 물리값은 0.03968828430148977이다. 원점은 0, 스케일은 0.5이다. 도함수는 -155.83164153568046에서 2.0260452150729957로 바뀌어 검증 열 배 15.583164153568045 안에 들어왔고, 옛 own gate 0.2070679793428482 밖에 있다. 그 열 배는 15.583으로 보이지만 15.583과 같지 않고, line 열 배 15.583164153568024와도, gradient 77 파일 열 배 15.583164153567989와도 같지 않다. Population 5는 여기서 멈춘다.

저장된 secant 행에서 확인한 도함수는 모두 절댓값 1 이상이다. 가장 작은 값은 population 13의 1.5304409677873398이고, 가장 큰 값은 population 2의 -354.618602279332이다. Population 0의 저장값은 -344.48453713017767이다. 검증 행의 population-0 도함수 0.0018867937843086435만 1보다 작다. 이 값은 job 418545 검증값 0.0018867937847633909, gradient 77 파일값 0.0018867937824896541, population-0 secant 저장값 0.0018867937852181382와 서로 다른 float다. 합치지 않는다. 이 검증값은 직전 상태이므로 job 418601을 H200에 고정하지 않는다.

검증 identity는 0이다. Secant identity는 -9.313225746154785e-10이다. 변화 잔차는 -8.737366385958012e-11이고, 그 값을 secant identity로 나눈 몫은 0.09381675720214844이다. 검증 identity로 나누지 않는다. Prior 잔차는 -4.848552115355176e-16이다. 측정 경과 1062.4386371369474초의 정수는 1062다. Slurm 18:02는 1082초다. Host peak 13.783817291259766 GiB는 13.78로 보이지만 13.78과 같지 않다.

이 상태는 수렴 MAP도, posterior 표본도, production IC도, LG 결과도 아니다. R2는 NO-GO다. Astra의 중간 판정 B는 종료 승인이 아니다.

## 2. 진행 중인 작업

Job **418601** (`cf4_R2_fg78`)은 이 문서를 쓸 때 RUNNING이었다. 확인 시각 2026-10-10 23:41 KST, RunTime 00:07:21, partition `h200`, node syn104, `TresPerNode=gres/gpu:H200:1`, WorkDir `/home/kjhan/BACKUP/CF4`. 제한 시간은 01:20:00이다. 제출 커밋은 `a7b81a7fb7578640a74729730cfdc42ab01abf8e`다. 제출 직전 자유 GPU는 H200 1, H100 2, A100 5였고 H200이 비어 있어 그 파티션을 골랐다. `CF4_GPU_QUEUE_OCCUPIED`는 넣지 않았다. 고정 작업이 아니다.

출력 디렉터리 `/gpfs/kjhan/CF4/z0_density/r2_conditional_full_gradient78_20261010`는 작업이 만들며, 작성 시각의 `result.json` 상태는 `STARTED`다. Gradient는 아직 없다. 이 디렉터리를 지우거나 다시 만들지 않는다. 채점은 작업이 COMPLETED된 뒤 그 작업의 stdout, stderr, sacct, 끝난 `result.json`만 본다. 무한노름의 위치, 다음 좌표, 개선 여부를 미리 정하지 않는다.

이 세션은 job 418601에 persistent watcher를 하나 켜 두었다. Watcher가 살아 있으면 두 번째를 켜지 않는다. Watcher가 없고 sacct가 아직 종료가 아니면 watcher를 하나만 다시 켠다. 이미 끝난 작업에 watcher를 다시 달지 않는다. 418601을 다시 제출하지 않는다. `scancel -u kjhan`과 `pkill -f`는 쓰지 않는다. 취소는 이 저장소 WorkDir가 확인된, 이 세션이 제출한 job id만 대상으로 한다.

Gradient 78은 gradient 77의 `q`에서 nuisance index 14만 0.07937656860297954로 바꾼다. 변위는 0.0009152564684061365이다. Index 6, 9, 11, 12, 17, 21, 22는 움직이지 않는다. `pop5_line_step`, `pop0_revisit15_step`, `pop3_revisit5_step`은 취하지 않는다.

## 3. 사용자 판단 2026-10-10

좌표 순환의 남은 횟수와 종료 시각은 정해져 있지 않다. 받아들여진 목적함수는 오르지 않는다. 최근에는 한 번에 약 0.01–0.07이고, job 418285의 8208466.743509629에서 job 418557의 8208466.323127546까지 여러 순환을 거쳐 약 0.42가 줄었다. 전체 규모 8.2×10⁶에 비하면 실질적인 개선이 아니다.

결합 무한노름은 한 좌표를 줄이면 다른 좌표로 옮겨 간다. 마지막으로 기록된 값은 population 5의 155.83이고, secant는 그 좌표만 2.03으로 줄였다. 같은 행의 다른 도함수는 300 근처다. 1e-4 정지 조건에 접근하는 궤적은 보이지 않는다.

R2 종료점은 둘 중 하나다. 현재 목적함수의 conditional z=0 산출물이 기록되거나, 현재 목적함수·자료·자원 안에서는 남은 차이를 더 줄일 수 없다고 명시하는 경우다. 그룹 포함·선택·bias를 출처 없이 채워 완전 posterior로 만들지 않는다. 418601을 채점하기 전에 그 종료점을 선언하지 않는다.

## 4. 종료까지 하지 않는 일

- R3를 시작하지 않는다. 종료 메일을 보낸 턴과 사용자가 답하기 전에도 시작하지 않는다.
- Astra 종료 감사의 명시적 승인과 저장소 대조가 끝나기 전에 `kjhan0606@gmail.com`으로 종료 메일을 보내지 않는다. 승인이 없으면 메일을 보내지 않고, 근거가 있는 조언만 받아 R2를 계속한다.
- 중간 감사와 종료 감사는 Astra만 쓴다. 응답을 못 쓰면 다른 모델로 바꾸지 않고 Astra를 다시 호출한다. Q-GOAL과 Q-LEAN을 넣고, MW/M31은 ambiguous, M33는 unresolved이며 그 관측량이 같은 새 LG 장을 `<=0.3` cMpc/h에서 제약한다고 적는다. 원자료 정답 ID는 평가에만 쓴다.
- Held-out을 열지 않는다. Optimizer를 호출하지 않는다. IC를 움직이지 않는다. Likelihood나 prior 항을 만들지 않는다.
- 논문은 사용자가 요청할 때만 쓴다. 그러면 `~/polish-astronomy-english/SKILL.md`와 `references/preferences.md`를 따른다. 숫자, 부호, 정밀도, 단위, 인과 방향, 단서를 바꾸거나 결과를 만들지 않는다.

## 5. 운영

- Syntax는 Slurm 로그인 노드다. GPU는 H200이 비어 있으면 `--partition=h200 --gres=gpu:H200:1`을 우선한다. 점유 중이면 H100 또는 A100 중 맞는 하나만을 제출한다. 저장된 측정에서 확인한 도함수의 절댓값이 1보다 작으면 그 다음 작업은 그 측정의 GPU 종류에 고정한다. 고정은 방금 끝난 측정의 받아들여진 행을 따른다. 검증 행의 옛 상태는 고정 근거가 아니다. 거절된 line step이면 그 측정의 검증 행이나 거절 행 중 1보다 작은 도함수가 다음 secant를 고정한다. 노드에서 수동 실행하지 않는다.
- 자유 GPU 수는 제출 때마다 다시 센다. 이전 작업이 기록한 H200/H100/A100 개수를 재사용하지 않는다. `scontrol show job`이 `CF4_*`를 표시하지 않는 것은 export 실패가 아니다.
- 작업 이름에는 `cf4_` 접두를 붙인다. 저장소 루트의 `OWNER.txt`는 untracked이며 커밋하지 않는다.
- 다음 untracked 파일은 수정하거나 커밋하지 않는다. `config/cf4_q0_gh_boundary_diagnostic_bundle_v1.json`, `config/grok_bundle_plan_audit_prompt_20260920.md`, `config/grok_code_audit_prompt_20260920.md`, `scripts/tripwire/`.
- 단위 테스트는 pytest가 아니다. Circle 해석기 `/home/kjhan/miniconda3/envs/circle/bin/python3.11`로 `tests/test_cf4_r2_conditional_optimizer_diagnostic.py -q`를 실행한다. `PYTHONPATH=scripts:src`, `JAX_ENABLE_X64=1`. HEAD `a7b81a7`의 결과는 222개 OK다.
- 실행 중인 작업이 보는 다섯 경로는 그 작업이 끝나기 전에 다시 고치지 않는다. `src/cf4_r2_resolution_target.py`, `scripts/cf4_r2_conditional_map.py`, `scripts/cf4_r2_conditional_optimizer_diagnostic.py`, `scripts/run_cf4_r2_conditional_full_gradient78.sbatch`, `tests/test_cf4_r2_conditional_optimizer_diagnostic.py`.

## 6. 다음 세션이 바로 할 일

1. `/home/kjhan/.codex/SHARED_CONTEXT.md`, 이 문서, `CF4_MASTER_PLAN.md`를 읽는다.
2. `git status --short --branch`와 `sacct -j 418601`으로 상태를 확인한다. 23:41의 RunTime 00:07:21을 종료 시간으로 쓰지 않는다.
3. 418601이 끝나기 전에는 채점하지 않는다. 끝나면 stdout, stderr, sacct, `result.json`만으로 채점하고, 그 다음에만 다음 측정을 정한다.
4. 미수렴 상태를 posterior, production map, LG 결과로 부르지 않는다.
