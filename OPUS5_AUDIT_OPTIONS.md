# Opus5 감사 실행 규약

이 프로젝트의 코드 구현·묶음 종료 감사는 Claude Code의 안정적인 `opus`
별칭을 통해 수행한다. 현재 CLI에서 `opus-5`는 모델 카탈로그에 등록되지
않으며, `claude-opus-4-8`은 별도 시험에서 응답했지만 Opus5 감사 규약에는
`opus` 별칭을 사용한다.

```bash
cd /home/kjhan/BACKUP/CF4
timeout 600 claude -p "$AUDIT_PROMPT" \
  --model opus \
  --permission-mode plan \
  --allowed-tools Read,Glob,Grep \
  --disallowed-tools Edit,Write,Bash \
  --effort medium \
  --output-format text
```

감사 프롬프트에는 반드시 다음을 포함한다.

1. `Read-only audit. Do not edit files, build, submit jobs, or run simulations.`
2. 대상 파일과 감사 범위
3. CF4 프로젝트의 최종 목적
4. 변경된 알고리듬·배선·보존식·단위
5. 판정 형식: `PASS`, `CONDITIONAL PASS`, `REJECT`, `NO VERDICT`

금지 옵션: `--bare`, `--dangerously-skip-permissions`, `--yolo`.
감사 결과는 운전자가 코드·실행 결과와 대조하여 채택 여부를 판단한다.
Opus CLI가 응답하지 않거나 모델이 미등록이면 감사 통과로 간주하지 않고,
실패 원인과 자체 검토 범위를 기록한다.
