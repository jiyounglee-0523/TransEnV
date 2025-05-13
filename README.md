# Trans-EnV: A Framework for Transforming into English Varieties to Evaluate the Robustness of LLMs
Trans-EnV repository

&nbsp;

&nbsp;



## Quick Starts 🚀
### Environment Setup
```bash
# PyTorch install (CUDA 12.1)
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Requirements
pip3 install -r requirements.txt
```

&nbsp;

### Benchmark Test
#### 1. .env 설정
먼저 아래같이 `.env` 파일에 `${GOOGLE_API_KEY}`, `${DATA_DIR}`을 설정해야합니다.
```bash
GOOGLE_API_KEY=${GCP_API_KEY}
DATA_DIR=${HF_BENCHMARK_PATH}
```

#### 2. Shell script 실행
그리고 아래의 shell script를 실행 시킬 수 있습니다.
* `run_dialect_benchmark.sh`
* `run_l1_benchmark.sh`
```bash
# Dialect benchmark test 실행
bash run_dialect_benchmark.sh

# ESL benchmark test 실행
bash run_l1_benchmark.sh
```

shell script에서 돌리고싶은 benchmark를 수동으로 설정할 수 있습니다.
<details>
<summary><code>run_dialect_benchmark.sh</code> 예시</summary>

```bash
#!/bin/bash

models=("winogrande") # 예시 ("winogrande" "gsm8k" "arc")
dialects=(
    "aave_rerun" "irish_rerun" "australian_rerun" "bhamanian_rerun" "east_anglian_rerun" "appalachian_rerun" "southeast_england_rerun" "australian_vernacular_rerun" "north_england_rerun" "southwest_england_rerun" "falkland_islands_rerun" "manx_rerun" "new_zealand_rerun" "ozark_rerun" "scottish_rerun" "southeast_american_rerun" "cunha_rerun" "welsh_rerun"
)


log_dir="./logs_dialect"
mkdir -p "$log_dir"

MAX_JOBS=40   # 동시에 실행할 최대 task 개수
PIDS=()

run_limited() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done

    model=$1
    dialect=$2
    log_file="${log_dir}/${model}_${dialect}.out"
    err_file="${log_dir}/${model}_${dialect}.err"

    echo "Launching: $model - $dialect"
    nohup bash benchmark_test.sh models/gemini-2.5-pro-exp-03-25 "$model" dialect "$dialect" >"$log_file" &
}

for model in "${models[@]}"; do
    for dialect in "${dialects[@]}"; do
        run_limited "$model" "$dialect"
        sleep 0.2  # 과도한 부하 방지
    done
done

# 남은 모든 백그라운드 작업 기다리기
wait

echo "✅ All nohup-limited jobs finished. Logs in $log_dir"

```
</details>


&nbsp;
