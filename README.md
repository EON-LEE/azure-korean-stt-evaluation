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

5. 전체 데이터를 준비하고 전송할 오디오 양을 확인한다. Azure를 호출하지 않는다.

   ```powershell
   uv run kstt prepare-data --subset full
   uv run kstt run --routes all --subset full --run-id full-1 --dry-run
   ```

6. 전체 평가를 한다. 전체 평가에는 `--max-audio-hours`가 필수다. 이 값은 모든 경로와 재시도를 합친 오디오 시간 상한이다.

   ```powershell
   uv run kstt supervise --routes all --subset full --run-id full-1 --max-audio-hours <승인한 시간>
   ```

   `kstt run`은 발화 목록을 한 번 처리(pass)하고 끝난다. 실패한 발화를 다시 보내려면 같은 명령을 다시 실행해야 한다. 수 시간 이상 걸리는 실행에는 `kstt supervise` 또는 `scripts\run-full.ps1`을 사용한다. 아래 "긴 실행" 절을 참고한다.

`--subset full`로 실제 실행을 하면 Azure를 호출하기 전에 다음을 검사한다. 검사에 실패하면 실행을 시작하지 않는다.

- git 작업 트리에 commit하지 않은 변경이 없어야 한다. 결과를 하나의 commit에 대응시키기 위해서다.
- 선택한 모든 발화의 오디오 sha256이 manifest와 일치해야 한다. 이 검사는 모든 subset에 적용된다.

### 실행 옵션

`kstt run`과 `kstt supervise`는 같은 옵션을 사용한다.

| 옵션 | 설명 |
|---|---|
| `--routes` | 경로 키, glob(`fast-*`), `all`, `track:A`, `track:B`를 쉼표로 이어서 쓴다. |
| `--run-id` | 실행 이름이다. 결과는 `runs/<run-id>/`에 저장된다. `supervise`에는 필수다. |
| `--sample-per-split N` | 각 split에서 seed 순서의 처음 N개 발화를 고른다. 예: `300`이면 두 split에서 모두 600개다. |
| `--limit N` | 전체 발화에서 seed 순서의 처음 N개를 고른다. `--sample-per-split`과 함께 쓸 수 없다. |
| `--seed` | 발화 순서와 표본 선택의 seed다. 기본값은 `kstt-1`이다. 같은 seed는 같은 표본을 고른다. |
| `--max-audio-hours` | 모든 경로와 모든 시도를 합친 오디오 시간 상한이다. 같은 `--run-id`의 이전 실행에서 쓴 시간도 포함한다. |
| `--max-attempts` | 발화 하나를 모델까지 보낼 수 있는 최대 횟수다. 기본값은 4다. |
| `--resource-limit NAME=N` | Azure 리소스 하나에 동시에 보내는 요청 수 상한이다. 여러 번 줄 수 있다. `NAME`은 경로 설정의 `resource` 값이고, 이 값이 없으면 리전 이름이다. |
| `--parallel-routes` | 동시에 실행하는 경로 수다. 기본값 `0`은 모든 경로를 동시에 실행한다. |
| `--allow-dirty` | commit하지 않은 변경이 있어도 `--subset full`을 실행한다. 보고서에 "커밋되지 않은 변경 포함"이 표시된다. |
| `--no-verify-audio` | 오디오 sha256 검사를 건너뛴다. |
| `--no-cache` | 캐시를 쓰지 않고 다시 호출한다. |
| `--no-report` | 실행이 끝나도 보고서를 만들지 않는다. |

### 긴 실행

`kstt supervise`는 모든 발화가 최종 상태가 될 때까지 pass를 반복한다. 최종 상태는 `ok`, `empty`, `failed_final`이다.

- pass 사이에 `--cooldowns`(기본값 `300,900,1800`초)만큼 기다린다. 진척이 없는 pass가 이어지면 대기 시간이 다음 값으로 늘어난다. 진척이 있으면 첫 값으로 돌아간다.
- 진척이 없는 pass가 `--max-idle-passes`(기본값 2)번 이어지면 종료 코드 3으로 끝난다.
- Windows에서는 실행 중 절전 모드로 들어가지 않도록 요청한다. `--no-keep-awake`를 주면 요청하지 않는다.
- 모든 발화가 최종 상태가 되면 보고서를 만들고 종료 코드 0으로 끝난다.
- 오디오 시간 상한에 도달하거나 차단된 경로만 남으면 보고서를 만들고 종료 코드 3으로 끝난다.

실행 중 다음 조건이 생기면 실행이 일시 중지되고, 조건이 해소되면 자동으로 재개된다.

- Azure 로그인 토큰을 받을 수 없다. `az login`을 다시 실행한다.
- `runs/` 디스크의 여유 공간이 2 GiB보다 작다.

한 실행 디렉터리는 한 프로세스만 사용한다. 다른 프로세스가 같은 `--run-id`를 사용 중이면 종료 코드 5로 끝난다. 같은 컴퓨터에서 이전 프로세스가 비정상 종료했으면 `runs/<run-id>/lock`을 바로 회수한다. 다른 컴퓨터의 lock은 120초 동안 갱신되지 않을 때만 회수한다.

#### `scripts\run-full.ps1`

> [!WARNING]
> 이 스크립트는 실제 Azure를 호출하고 과금된다. 먼저 `--dry-run`으로 전송량을 확인한다.

이 스크립트는 `kstt supervise`를 실행한다. 프로세스가 비정상 종료하면 같은 명령으로 다시 시작한다. `-BackupDir`를 주면 실행 결과를 정기적으로 복사한다.

PowerShell 7:

```powershell
pwsh -File scripts\run-full.ps1 -RunId full-1 -MaxAudioHours <승인한 시간> -BackupDir <백업 폴더>
```

Windows PowerShell 5.1(스크립트 실행 정책이 기본값인 경우):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-full.ps1 -RunId full-1 -MaxAudioHours <승인한 시간> -BackupDir <백업 폴더>
```

`-ExecutionPolicy Bypass`는 이 프로세스에만 적용된다. 시스템 설정을 바꾸지 않는다.

| 매개 변수 | 기본값 | 설명 |
|---|---|---|
| `-RunId`, `-MaxAudioHours` | 필수 | `kstt supervise`의 `--run-id`, `--max-audio-hours`로 전달한다. |
| `-Routes`, `-Subset` | `all`, `full` | `--routes`, `--subset`으로 전달한다. |
| `-ExtraArgs` | 없음 | 그 밖의 `supervise` 옵션을 문자열 하나로 쓴다. 예: `"--resource-limit koreacentral=8"` |
| `-MaxCrashes` | 5 | 완료 발화 수가 늘지 않은 채 비정상 종료가 이 횟수만큼 이어지면 종료 코드 1로 끝난다. |
| `-RestartDelaySeconds` | 60 | 비정상 종료 뒤 다시 시작하기 전에 기다리는 시간이다. |
| `-BackupDir` | 없음 | 백업 폴더다. 없으면 백업하지 않는다. |
| `-BackupEveryMinutes` | 60 | `runs/<run-id>/`를 복사하는 간격이다. |
| `-CacheBackupEveryHours` | 24 | 공유 캐시 `runs/cache/`를 복사하는 간격이다. |

- `kstt`가 종료 코드 0, 2, 3, 4, 5, 130으로 끝나면 스크립트도 같은 코드로 끝난다. 이때 실행 결과와 캐시를 한 번 더 백업한다.
- 그 밖의 종료 코드는 비정상 종료로 처리한다.
- 백업은 `robocopy`로 하며 `lock`과 `STOP` 파일은 복사하지 않는다.

### 상태 확인과 중지

실행 상태는 다른 터미널에서 확인한다. 이 명령은 파일을 읽기만 한다.

```powershell
uv run kstt status --run full-1
uv run kstt status --run full-1 --json
```

경로별 전체·최종·남은 발화 수, 실행 중 여부, 일시 중지 사유, STOP 파일 여부를 표시한다.

실행을 중지하려면 `STOP` 파일을 만든다.

```powershell
New-Item -ItemType File runs\full-1\STOP
```

1. 진행 중인 요청이 끝날 때까지 기다린 뒤 종료 코드 4로 끝난다.
2. `STOP` 파일이 있는 동안에는 같은 `--run-id`의 실행이 시작되지 않는다.
3. 계속하려면 `STOP` 파일을 삭제하고 같은 명령을 다시 실행한다.

`supervise`는 `STOP`으로 끝날 때 보고서를 만들지 않는다. 중간 결과가 필요하면 `uv run kstt report --run full-1`을 실행한다.

### 종료 코드

| 코드 | 의미 |
|---|---|
| 0 | 완료했다. |
| 1 | `run-full.ps1`이 비정상 종료 반복으로 재시작을 포기했다. |
| 2 | 설정 또는 데이터 오류다. 예: 옵션 오류, 빠진 환경 변수, 오디오 불일치, 손상된 로그 줄. |
| 3 | 완료하지 못했다. 오디오 시간 상한에 도달했거나, 차단된 경로만 남았거나, 진척 없는 pass가 반복됐다. `kstt judge`는 토큰 상한에 도달했다. |
| 4 | `STOP` 파일로 중지했다. |
| 5 | 다른 프로세스가 같은 실행을 사용 중이다. |
| 130 | Ctrl+C로 중단했다. 같은 명령을 다시 실행하면 재개한다. |

### 재개와 캐시

- 같은 `--run-id`로 다시 실행하면 최종 상태(`ok`, `empty`, `failed_final`)인 발화는 건너뛰고 나머지 발화만 다시 보낸다.
- 발화가 `failed_final`이 되는 조건은 다음 중 하나다.
  - 모델까지 보낸 횟수가 `--max-attempts`에 도달했다.
  - 같은 발화에서 재시도해도 결과가 바뀌지 않는 오류(예: HTTP 400 요청 오류)가 2번 발생했다.
- 모델에 도달하지 않은 오류(429 제한, 네트워크 오류, 로그인 오류, 배포 없음)는 시도 횟수와 오디오 시간 상한에 넣지 않는다.
- 시도 횟수와 사용한 오디오 시간은 `runs/<run-id>/ledger.jsonl`에 기록된다. 프로세스가 다시 시작돼도 누적된다.
- 새 `--run-id`라도 같은 오디오와 같은 경로 설정의 응답이 로컬 캐시에 있으면 캐시에서 가져온다. 캐시 적중은 Azure를 호출하지 않으므로 `--max-audio-hours` 예산을 쓰지 않는다.
- 캐시 적중 결과는 지연 시간 통계에서 제외한다. 보고서의 `캐시` 열에서 캐시 적중 수를 확인한다.
- 채점 규칙만 바뀌었으면 `uv run kstt report --run <run-id>`로 보고서만 다시 만든다.

## 보고서

`reports/<run-id>/`에 다음 파일이 생성된다.

| 파일 | 내용 |
|---|---|
| `report.md` | 실행 정보, 경로별 결과, split별 결과, 실행 상태, 기준 경로 대비 차이, 확인 필요 항목 |
| `summary.csv`, `summary.json` | 경로 × split 집계 |
| `comparisons.csv` | 기준 경로(기본 `fast-azure`) 대비 paired bootstrap 결과 |
| `per_utterance.csv` | 발화별 정답, 인식 결과, 오류 수, 지연 시간, 지연 제외 여부(`latency_invalid`), 지연 경고(`latency_flags`) |

`per_utterance.csv`에는 정답 문장이 들어 있다. 이 파일을 외부에 공유하지 않는다.

### 지표

- **sCER(주 지표):** 공백을 제거한 문자 오류율이다. 한국어 띄어쓰기 차이를 오류로 세지 않는다.
- **CER, WER:** 공백을 포함한 문자 오류율과 어절 오류율이다.
- **95% CI:** 발화 단위 bootstrap 신뢰구간이다. split별 층화 표본을 사용한다.
- **지연:** Track A는 요청 시간이다. Track B는 원본 클립 끝부터 마지막 최종 결과까지의 시간이다. 클립 끝의 무음 구간에서 최종 결과가 나오면 음수일 수 있다.
  - 지연 통계에는 이번 실행에서 실제로 요청한 결과만 사용한다.
  - 캐시 적중 결과는 제외한다.
  - 요청 중에 컴퓨터가 절전 또는 일시 중지 상태였던 결과(`latency_invalid`)도 제외한다.
  - 프로그램 지연이나 오디오 전송 지연이 있었던 결과는 집계에 포함하고 `latency_flags`에 표시한다. 보고서의 "실행 상태" 절에서 경로별 개수를 확인한다.
- **한글 비율, script mismatch:** 출력이 한국어가 아닌 경우를 찾는다. 출력의 한글 비율이 정답보다 0.2 이상 낮은 발화를 `script_mismatch`로 센다.

정답과 인식 결과에 같은 정규화(`ko-norm-v1`)를 적용한다.

1. KsponSpeech 이중 전사 `(철자)/(발음)`에서 철자 전사를 고른다.
2. 잡음 표식(`b/`, `l/`, `o/`, `n/`, `u/`)을 제거한다. 간투사 표식은 제거하고 간투사 단어는 유지한다.
3. 영문을 소문자로 바꾼다.
4. 문장부호와 기호를 제거하고 공백을 하나로 줄인다.

숫자 변환, 동의어 사전, 정답 문장에서 만든 규칙은 사용하지 않는다. 따라서 간투사를 생략하는 모델은 그만큼 오류율이 높게 나온다.

### 실행 상태

`report.md`의 "실행 상태" 절은 경로별 실행 품질을 보여 준다. 지연 시간 비교 전에 이 절을 확인한다.

| 열 | 내용 |
|---|---|
| 시도 | 모델까지 보낸 요청 수의 합계다. 프로세스가 다시 시작돼도 `ledger.jsonl`에서 누적한다. |
| 429 | 발화별 마지막 처리에서 받은 429 제한 횟수의 합계다. |
| 최종 실패 | `failed_final` 발화 수다. |
| 지연 표본 | 지연 통계에 사용한 결과 수다. |
| 지연 제외(시계 이상) | 절전 또는 일시 중지 때문에 지연 통계에서 제외한 결과 수다. |
| 지연 경고 | `loop_lag`(프로그램 지연) 또는 `send_delay`(오디오 전송 지연)가 표시된 결과 수다. |
| 전송 지연 p95 | 실시간 경로에서 오디오 전송 시간 − 클립 길이의 p95다. |

지연 경고나 전송 지연이 큰 경로는 지연 시간 순위를 신뢰하지 않는다. 같은 경로를 다른 시간에 다시 측정한다.

### 해석할 때 주의할 점

- 스모크 결과는 발화 5개다. 연결과 파이프라인 동작 확인용이며 모델 성능 비교에 쓰지 않는다.
- 경로마다 리소스 리전이 다르다. 지연 시간 차이에는 모델 차이와 네트워크 거리 차이가 함께 들어 있다.
- 가격을 설정하지 않은 경로의 비용은 `미설정`으로 표시한다. 0으로 표시하지 않는다.

## LLM 의미 판정 (`kstt judge`, 검증 전)

`kstt judge`는 STT 실행이 끝난 뒤 따로 실행한다. 판정 모델이 인식 결과가 정답 문장의 의미를 보존하는지 판정한다. sCER을 대체하지 않는 보조 지표다.

> [!WARNING]
> 사람 라벨로 판정 모델을 검증하기 전까지 결과는 `검증 전`이다. 판정 결과만으로 모델 순위를 정하지 않는다.

판정 방식은 다음과 같다.

1. 정규화 후 공백을 제거한 문장이 정답과 같으면 판정 모델을 호출하지 않고 `preserved`로 판정한다.
2. 인식 결과가 비어 있으면 호출하지 않고 `lost`로 판정한다.
3. STT 요청이 실패한 발화는 판정하지 않는다.
4. 나머지 쌍은 같은 정답과 같은 인식 결과를 한 번만 판정한다. 요청에는 고정 지시문, 정답 문장, 인식 결과만 넣는다. 경로 이름과 모델 이름은 보내지 않는다.

### 판정 모델 설정

판정 모델은 `configs\judge.yaml`에 정의한다. 기본 판정 모델은 `gpt-5-mini`이고 `gpt-5.5`도 선택할 수 있다. `configs\endpoints.env`에 다음 값을 설정한다.

```dotenv
KSTT_JUDGE_AOAI_ENDPOINT=https://<judge-ai-services-resource>.openai.azure.com
# Optional deployment name overrides
# KSTT_JUDGE_MINI_DEPLOYMENT=gpt-5-mini
# KSTT_JUDGE_LARGE_DEPLOYMENT=gpt-5.5
```

### 판정 실행 순서

> [!WARNING]
> `--dry-run` 이외의 명령은 Azure OpenAI를 호출하고 비용이 발생한다. 실행 전에 승인을 받는다.

1. 호출 없이 판정 대상 수, 규칙 판정 수, 중복 제거 후 호출 수, 캐시 적중 수를 확인한다.

   ```powershell
   uv run kstt judge --run <run-id> --dry-run
   ```

2. 판정 모델을 점검 세트로 확인한다. `configs\judge_checkset.yaml`에는 기대 판정이 정해진 20쌍이 있다. 20쌍이 모두 기대 판정과 같으면 로그에 `PASS`가 출력된다. `FAIL`이면 3단계로 가지 않는다.

   ```powershell
   uv run kstt judge --checkset configs\judge_checkset.yaml --max-judge-tokens 100000
   ```

   결과는 `reports\judge-checkset\`에 저장된다.

3. 실행 결과를 판정한다. `--baseline`을 주면 기준 경로 대비 MPR과 손실률 차이를 계산한다.

   ```powershell
   uv run kstt judge --run <run-id> --judge gpt-5-mini --max-judge-tokens 2000000 --baseline <route>
   ```

### 토큰 상한과 재개

- `--max-judge-tokens`는 `--dry-run`이 아니면 필수다. 이 판정 모델이 사용한 입력과 출력 토큰의 누적 상한이다. 누적 값은 캐시에서 계산한다.
- 상한에 도달하면 새 호출을 멈추고 종료 코드 3으로 끝난다. 이미 보낸 요청은 끝까지 기다린다.
- 상한을 높여 같은 명령을 다시 실행하면 이어서 판정한다. 판정이 끝난 쌍은 다시 호출하지 않는다. 판정 실패(`judge_error`) 쌍만 다시 호출한다.
- 캐시는 `runs\<run-id>\judge\<judge-id>\cache.jsonl`이다.

### 판정 결과

| 파일 | 내용 |
|---|---|
| `runs\<run-id>\judge\<judge-id>\judgments.jsonl` | 발화별 판정 |
| `reports\<run-id>\judge_<judge-id>.csv` | 경로 × split 집계 |
| `reports\<run-id>\judge_<judge-id>.md` | 판정 정보, 경로별 결과, split별 결과, 오류 유형 분포, 기준 경로 대비 차이, 판정 실패 목록 |

- **MPR(의미 보존율):** `preserved` 수 ÷ 판정 수다. `partial`(부분 보존)은 보존으로 세지 않는다.
- **손실률(`lost_rate`):** `lost` 수 ÷ 판정 수다.
- 95% CI는 split 비율을 유지한 bootstrap 구간이다.

판정 결과 파일에는 정답 문장이 들어 있다. 이 파일을 외부에 공유하지 않는다.

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