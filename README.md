# azure-korean-stt-evaluation

Azure의 음성 인식(STT) 모델을 한국어 대화 음성으로 평가하는 도구다. KsponSpeech 평가 세트를 실제 Azure 엔드포인트로 보내고 정확도, 지연 시간, 실패율, 예상 비용을 같은 기준으로 집계한다.

이 도구는 평가만 한다. 모델 학습이나 평가 정답을 이용한 튜닝은 하지 않는다.

## 평가 경로

`configs/models.yaml`에 29개 경로가 있다. 경로는 모델과 호출 방식의 조합이다.

| Track | 호출 방식 | 경로 |
|---|---|---|
| A (파일 전사) | Azure AI Speech Fast Transcription | `fast-azure`, `fast-llm`, `fast-mai2`, `fast-mai15` |
| A (파일 전사) | Azure OpenAI 파일 전사 | `aoai-whisper`, `aoai-gpt-transcribe`, `aoai-gpt-4o-transcribe`, `aoai-gpt-4o-mini-transcribe`, `aoai-gpt-4o-transcribe-diarize` |
| B (스트리밍) | Azure AI Speech SDK 연속 인식 | `sdk-azure` |
| B (스트리밍) | MAI-Transcribe-2-Streaming Realtime | `mai-rt` |
| B (스트리밍) | Azure OpenAI Realtime 전사 | `aoai-rt-live`, `aoai-rt-whisper` 및 각 `-vad` |
| B (스트리밍) | Voice Live 입력 전사 7개 모델 | `vl-*` 및 각 `-vad` |

Track B는 오디오를 1배속으로 보낸다.

- `sdk-azure`는 연속 인식을 사용하고 오디오 스트림을 닫으면 최종 결과를 받는다.
- Realtime·Voice Live 경로 중 이름에 `-vad`가 없는 경로는 오디오를 모두 보낸 뒤 commit한다.
- `-vad` 경로는 서버 VAD가 발화 구간을 나눈다. 클립 뒤에 무음 1초를 덧붙여 보낸다.

`kstt models`로 경로별 모델, 리전, 설정 상태를 확인한다.

## 데이터

KsponSpeech 평가 세트의 Hugging Face 사본 두 개를 사용한다.

| split | 저장소 | 발화 수 |
|---|---|---|
| `eval_clean` | `Kyudan/KsponSpeech-eval-clean` | 3,000 |
| `eval_other` | `Kyudan/KsponSpeech_eval_other` | 3,000 |

- 저장소 revision은 코드에 고정되어 있다. revision이 바뀌면 `prepare-data`가 중단된다.
- 이 저장소는 오디오와 정답 문장을 포함하지 않는다. `data/`, `runs/`, `reports/`는 git에서 제외된다.
- 데이터를 사용하기 전에 원 데이터의 이용 조건을 확인한다.

## 준비

### 필요 조건

- Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)
- Azure CLI 로그인(`az login`). 평가 대상 리소스는 키 인증 대신 Microsoft Entra ID 토큰을 사용한다.
- 로그인한 계정의 각 리소스 데이터 호출 권한. `kstt doctor`로 확인한다.

### 설치

```powershell
uv sync --extra speech-sdk
```

`--extra speech-sdk`는 `sdk-azure` 경로에 필요한 Speech SDK를 설치한다. 이 경로를 쓰지 않으면 생략한다.

pypi.org에 접속할 수 없는 네트워크에서는 `UV_DEFAULT_INDEX` 환경 변수로 조직의 패키지 미러를 지정한다. `uv.lock`에는 미러 주소가 기록되므로 git에서 제외한다.

### 엔드포인트 설정

1. `configs/endpoints.env.example`을 `configs/endpoints.env`로 복사한다.
2. 복사한 파일에 리소스 엔드포인트와 배포 이름을 입력한다.

`configs/endpoints.env`는 git에서 제외된다. 같은 이름의 환경 변수가 이미 있으면 환경 변수가 우선한다.

값이 없는 경로는 `kstt models`에서 `ready`가 `no`로 표시되고 빠진 변수 이름이 함께 표시된다. 이런 경로를 `--routes`로 선택하면 `kstt run`은 Azure를 호출하기 전에 중단된다.

## 실행 순서

아래 명령은 Windows PowerShell 기준이다. 실제 Azure 호출은 4단계부터 과금된다.

1. 인증, 엔드포인트, 배포를 확인한다. 오디오는 보내지 않는다.

   ```powershell
   uv run kstt doctor --ws
   ```

2. 스모크 데이터(clean 3개, other 2개)를 준비한다.

   ```powershell
   uv run kstt prepare-data --subset smoke
   ```

3. 전송할 오디오 양을 확인한다. Azure를 호출하지 않는다.

   ```powershell
   uv run kstt run --routes all --subset smoke --run-id smoke-1 --dry-run
   ```

4. 스모크 실행을 한다. 끝나면 `reports/smoke-1/report.md`가 생성된다.

   ```powershell
   uv run kstt run --routes all --subset smoke --run-id smoke-1 --max-audio-hours 0.5
   ```

5. 전체 평가를 한다. 전체 평가에는 `--max-audio-hours`가 필수다. 이 값은 모든 경로와 재시도를 합친 오디오 시간 상한이다.

   ```powershell
   uv run kstt prepare-data --subset full
   uv run kstt run --routes all --subset full --run-id full-1 --dry-run
   uv run kstt run --routes all --subset full --run-id full-1 --max-audio-hours <승인한 시간>
   ```

`--routes`에는 경로 키, glob(`fast-*`), `all`, `track:A`, `track:B`를 쉼표로 이어서 쓴다.

### 재개와 캐시

- 같은 `--run-id`로 다시 실행하면 마지막 상태가 `ok` 또는 `empty`인 발화는 건너뛰고 나머지 발화만 다시 보낸다.
- 새 `--run-id`라도 같은 오디오와 같은 경로 설정의 응답이 로컬 캐시에 있으면 캐시에서 가져온다. 캐시 적중은 Azure를 호출하지 않으므로 `--max-audio-hours` 예산을 쓰지 않는다.
- 캐시 적중 결과는 처음 호출에서 측정한 지연 시간을 그대로 사용한다. 보고서의 `cached` 열에서 캐시 적중 수를 확인한다.
- `--no-cache`를 주면 캐시를 쓰지 않고 다시 호출한다.
- 채점 규칙만 바뀌었으면 `uv run kstt report --run <run-id>`로 보고서만 다시 만든다.

## 보고서

`reports/<run-id>/`에 다음 파일이 생성된다.

| 파일 | 내용 |
|---|---|
| `report.md` | 경로별 결과, split별 결과, 기준 경로 대비 차이, 확인 필요 항목 |
| `summary.csv`, `summary.json` | 경로 × split 집계 |
| `comparisons.csv` | 기준 경로(기본 `fast-azure`) 대비 paired bootstrap 결과 |
| `per_utterance.csv` | 발화별 정답, 인식 결과, 오류 수, 지연 시간 |

`per_utterance.csv`에는 정답 문장이 들어 있다. 이 파일을 외부에 공유하지 않는다.

### 지표

- **sCER(주 지표):** 공백을 제거한 문자 오류율이다. 한국어 띄어쓰기 차이를 오류로 세지 않는다.
- **CER, WER:** 공백을 포함한 문자 오류율과 어절 오류율이다.
- **95% CI:** 발화 단위 bootstrap 신뢰구간이다. split별 층화 표본을 사용한다.
- **지연:** Track A는 요청 시간이다. Track B는 원본 클립 끝부터 마지막 최종 결과까지의 시간이다. 클립 끝의 무음 구간에서 최종 결과가 나오면 음수일 수 있다.
- **한글 비율, script mismatch:** 출력이 한국어가 아닌 경우를 찾는다. 출력의 한글 비율이 정답보다 0.2 이상 낮은 발화를 `script_mismatch`로 센다.

정답과 인식 결과에 같은 정규화(`ko-norm-v1`)를 적용한다.

1. KsponSpeech 이중 전사 `(철자)/(발음)`에서 철자 전사를 고른다.
2. 잡음 표식(`b/`, `l/`, `o/`, `n/`, `u/`)을 제거한다. 간투사 표식은 제거하고 간투사 단어는 유지한다.
3. 영문을 소문자로 바꾼다.
4. 문장부호와 기호를 제거하고 공백을 하나로 줄인다.

숫자 변환, 동의어 사전, 정답 문장에서 만든 규칙은 사용하지 않는다. 따라서 간투사를 생략하는 모델은 그만큼 오류율이 높게 나온다.

### 해석할 때 주의할 점

- 스모크 결과는 발화 5개다. 연결과 파이프라인 동작 확인용이며 모델 성능 비교에 쓰지 않는다.
- 경로마다 리소스 리전이 다르다. 지연 시간 차이에는 모델 차이와 네트워크 거리 차이가 함께 들어 있다.
- 가격을 설정하지 않은 경로의 비용은 `미설정`으로 표시한다. 0으로 표시하지 않는다.

## 테스트

```powershell
uv run pytest
```

기본 테스트는 네트워크를 쓰지 않는다. Azure 호출 테스트는 `--live`를 줄 때만 실행한다.

```powershell
uv run pytest --live
```

## GitHub workflow

- Keep the Copilot app and default GitHub CLI account unchanged.
- Use the existing URL-scoped Git credentials for normal fetch and push operations.
- For repository write operations with GitHub CLI, scope the `EON-LEE` token to the current process only.
- Do not re-include credential configuration, print tokens, weaken branch protection, or force-push.