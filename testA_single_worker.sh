#!/bin/bash

# ===== CONFIG =====
MODEL="/opt/dllama/src/models/deepseek_r1_distill_llama_8b_q40/dllama_model_deepseek_r1_distill_llama_8b_q40.m"
TOKENIZER="/opt/dllama/src/models/deepseek_r1_distill_llama_8b_q40/dllama_tokenizer_deepseek_r1_distill_llama_8b_q40.t"
PROMPT="Hello world"
STEPS=64
NTHREADS=4

WORKERS=(
  "192.168.111.54:9999"
  "192.168.111.52:9999"
  "192.168.111.53:9999"
)

# ===== SETUP =====
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="testA_logs_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Logs will be saved in: $LOG_DIR"
echo "===================================="

# ===== TEST LOOP =====
for WORKER in "${WORKERS[@]}"
do
    SAFE_NAME=$(echo $WORKER | tr ':' '_')
    LOG_FILE="$LOG_DIR/worker_${SAFE_NAME}.log"

    echo "🚀 Running test for worker: $WORKER"
    echo "Log: $LOG_FILE"

    ./dllama inference \
        --prompt "$PROMPT" \
        --steps $STEPS \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --buffer-float-type q80 \
        --nthreads $NTHREADS \
        --max-seq-len 4096 \
        --workers $WORKER \
        > "$LOG_FILE" 2>&1

    echo "✅ Done: $WORKER"
    echo "------------------------------------"
done

# ===== SUMMARY EXTRACTION =====
echo ""
echo "📊 Summary:"
echo "===================================="

for LOG in $LOG_DIR/*.log
do
    echo "File: $LOG"

    grep "tokens/s" "$LOG"
    echo ""

done
