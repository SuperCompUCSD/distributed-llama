#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/sweep.sh --model PATH --tokenizer PATH --workers host:port[,host:port...] [options]

Required:
  --model PATH           Model file to benchmark.
  --tokenizer PATH       Tokenizer file to benchmark.
  --workers LIST         Comma-separated root worker list, e.g. node1:9991,node2:9992.

Options:
  --repo-dir PATH        Repo root to build/run from. Default: script parent.
  --out-dir PATH         Sweep output directory. Default: logs/sweeps/<timestamp>.
  --deploy-script PATH   Optional deploy_repo.sh path. If set with DEPLOY_AFTER_BUILD=1, runs after each build.
  --deploy-nodes-file F  Optional nodes file for deploy_repo.sh.
  --build-jobs N         Parallel build jobs. Default: nproc.
  --prompt-word WORD     Word repeated to generate prompts. Default: hello.
  --buffer-float-type T   Buffer float type. Default: q80.
  --nthreads LIST        Comma-separated thread counts. Default: 1,2,4.
  --max-seq-lens LIST    Comma-separated max seq lens. Default: 1024,2048,4096,8192.
  --batches LIST         Comma-separated batch sizes. Default: 1,2,4,8,16,32.
  --chunk-sizes LIST     Comma-separated MAX_CHUNK_SIZE values. Default: 65536,131072,262144.
  --prompt-lengths LIST  Comma-separated prompt lengths in words. Default: 16,32,64,128,256.
  --temperature VALUE    Sampling temperature. Default: 0.8.
  --topp VALUE           Top-p sampling. Default: 0.9.
  --steps-factor N       Steps = max-seq-len * factor. Default: 1.
  --deploy-after-build   Run deploy_repo.sh --build after each local build.
  --help                 Show this message.

Environment variables can also be used for the sweep lists:
  NTHREADS_LIST, MAX_SEQ_LENS, BATCHES, CHUNK_SIZES, PROMPT_LENGTHS.
EOF
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir=""
deploy_script=""
deploy_nodes_file=""
build_jobs="$(nproc)"
prompt_word="hello"
buffer_float_type="q80"
temperature="0.8"
topp="0.9"
steps_factor="1"
deploy_after_build="${DEPLOY_AFTER_BUILD:-0}"
model_path=""
tokenizer_path=""
workers_csv=""

nthreads_list="${NTHREADS_LIST:-4}"
max_seq_lens="${MAX_SEQ_LENS:-1024,2048,4096}"
batches_list="${BATCHES:-8,16,32,64}"
chunk_sizes="${CHUNK_SIZES:-65536,131072,262144}"
prompt_lengths="${PROMPT_LENGTHS:-64,128,256}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_path="${2:-}"; shift 2 ;;
    --tokenizer) tokenizer_path="${2:-}"; shift 2 ;;
    --workers) workers_csv="${2:-}"; shift 2 ;;
    --repo-dir) repo_dir="${2:-}"; shift 2 ;;
    --out-dir) out_dir="${2:-}"; shift 2 ;;
    --deploy-script) deploy_script="${2:-}"; shift 2 ;;
    --deploy-nodes-file) deploy_nodes_file="${2:-}"; shift 2 ;;
    --build-jobs) build_jobs="${2:-}"; shift 2 ;;
    --prompt-word) prompt_word="${2:-}"; shift 2 ;;
    --buffer-float-type) buffer_float_type="${2:-}"; shift 2 ;;
    --temperature) temperature="${2:-}"; shift 2 ;;
    --topp) topp="${2:-}"; shift 2 ;;
    --steps-factor) steps_factor="${2:-}"; shift 2 ;;
    --nthreads) nthreads_list="${2:-}"; shift 2 ;;
    --max-seq-lens) max_seq_lens="${2:-}"; shift 2 ;;
    --batches) batches_list="${2:-}"; shift 2 ;;
    --chunk-sizes) chunk_sizes="${2:-}"; shift 2 ;;
    --prompt-lengths) prompt_lengths="${2:-}"; shift 2 ;;
    --deploy-after-build) deploy_after_build=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$model_path" || -z "$tokenizer_path" || -z "$workers_csv" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -x "$repo_dir/dllama" ]]; then
  echo "Building will create $repo_dir/dllama" >&2
fi

if [[ -z "$out_dir" ]]; then
  out_dir="$repo_dir/logs/sweeps/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$out_dir"

results_csv="$out_dir/results.csv"
run_manifest="$out_dir/manifest.txt"

cat > "$run_manifest" <<EOF
repo_dir=$repo_dir
model_path=$model_path
tokenizer_path=$tokenizer_path
workers_csv=$workers_csv
buffer_float_type=$buffer_float_type
temperature=$temperature
topp=$topp
steps_factor=$steps_factor
nthreads_list=$nthreads_list
max_seq_lens=$max_seq_lens
batches_list=$batches_list
chunk_sizes=$chunk_sizes
prompt_lengths=$prompt_lengths
EOF

printf 'chunk_size,max_seq_len,batches,nthreads,prompt_words,steps,status,summary_path,run_log,prompt_tokens,eval_tokens,pred_tokens,eval_ms_per_tok,pred_ms_per_tok,eval_tokens_per_s,pred_tokens_per_s,run_total_ms\n' > "$results_csv"

split_csv() {
  local value="$1"
  local -n out_ref="$2"
  local IFS=','
  read -ra out_ref <<< "$value"
}

trim_value() {
  local value="$1"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"
  printf '%s' "$value"
}

make_prompt() {
  local words="$1"
  local token_word="$2"
  yes "$token_word" | head -n "$words" | tr '\n' ' ' | sed 's/[[:space:]]*$//'
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

get_summary_value() {
  local key="$1"
  local file="$2"
  local line
  line="$(grep -m1 "^${key}=" "$file" 2>/dev/null || true)"
  if [[ -n "$line" ]]; then
    printf '%s' "${line#*=}"
  fi
}

append_result() {
  local chunk_size="$1"
  local max_seq_len="$2"
  local batches="$3"
  local nthreads="$4"
  local prompt_words="$5"
  local steps="$6"
  local status="$7"
  local summary_path="$8"
  local run_log="$9"

  local prompt_tokens eval_tokens pred_tokens eval_ms_per_tok pred_ms_per_tok eval_tokens_per_s pred_tokens_per_s run_total_ms
  prompt_tokens="$(get_summary_value prompt_tokens "$summary_path")"
  eval_tokens="$(get_summary_value eval_tokens "$summary_path")"
  pred_tokens="$(get_summary_value pred_tokens "$summary_path")"
  eval_ms_per_tok="$(get_summary_value eval_ms_per_tok "$summary_path")"
  pred_ms_per_tok="$(get_summary_value pred_ms_per_tok "$summary_path")"
  eval_tokens_per_s="$(get_summary_value eval_tokens_per_s "$summary_path")"
  pred_tokens_per_s="$(get_summary_value pred_tokens_per_s "$summary_path")"
  run_total_ms="$(get_summary_value run_total_ms "$summary_path")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$chunk_size" "$max_seq_len" "$batches" "$nthreads" "$prompt_words" "$steps" "$status" \
    "$(csv_escape "$summary_path")" "$(csv_escape "$run_log")" \
    "${prompt_tokens:-}" "${eval_tokens:-}" "${pred_tokens:-}" \
    "${eval_ms_per_tok:-}" "${pred_ms_per_tok:-}" "${eval_tokens_per_s:-}" "${pred_tokens_per_s:-}" "${run_total_ms:-}" \
    >> "$results_csv"
}

split_csv "$nthreads_list" nthreads_arr
split_csv "$max_seq_lens" max_seq_arr
split_csv "$batches_list" batches_arr
split_csv "$chunk_sizes" chunk_arr
split_csv "$prompt_lengths" prompt_arr
split_csv "$workers_csv" workers_arr

workers_args=()
for worker in "${workers_arr[@]}"; do
  worker="$(trim_value "$worker")"
  [[ -n "$worker" ]] && workers_args+=("$worker")
done

if [[ ${#workers_args[@]} -eq 0 ]]; then
  echo "No workers provided after parsing workers_csv=$workers_csv" >&2
  exit 1
fi

for chunk_size in "${chunk_arr[@]}"; do
  chunk_size="$(trim_value "$chunk_size")"
  [[ -z "$chunk_size" ]] && continue

  echo "==> Building chunk size $chunk_size"
  (cd "$repo_dir" && make clean >/dev/null && make -j"$build_jobs" dllama MAX_CHUNK_SIZE="$chunk_size")

  if [[ "$deploy_after_build" == "1" ]]; then
    if [[ -z "$deploy_script" ]]; then
      echo "DEPLOY_AFTER_BUILD=1 requested, but --deploy-script was not set" >&2
      exit 1
    fi
    echo "==> Deploying via $deploy_script"
    if [[ -n "$deploy_nodes_file" ]]; then
      "$deploy_script" --build "$deploy_nodes_file"
    else
      "$deploy_script" --build
    fi
  fi

  for max_seq_len in "${max_seq_arr[@]}"; do
    max_seq_len="$(trim_value "$max_seq_len")"
    [[ -z "$max_seq_len" ]] && continue

    for batches in "${batches_arr[@]}"; do
      batches="$(trim_value "$batches")"
      [[ -z "$batches" ]] && continue

      for nthreads in "${nthreads_arr[@]}"; do
        nthreads="$(trim_value "$nthreads")"
        [[ -z "$nthreads" ]] && continue

        for prompt_words in "${prompt_arr[@]}"; do
          prompt_words="$(trim_value "$prompt_words")"
          [[ -z "$prompt_words" ]] && continue

          if (( prompt_words >= max_seq_len )); then
            echo "Skipping prompt_words=$prompt_words for max_seq_len=$max_seq_len"
            continue
          fi

          prompt_text="$(make_prompt "$prompt_words" "$prompt_word")"
          steps=$(( max_seq_len * steps_factor ))
          run_tag="chunk${chunk_size}_seq${max_seq_len}_b${batches}_t${nthreads}_p${prompt_words}_$(date +%Y%m%d-%H%M%S-%3N)"
          run_log="$out_dir/${run_tag}.log"
          summary_path=""

          echo "==> Running $run_tag"
          if "$repo_dir/dllama" inference \
            --model "$model_path" \
            --tokenizer "$tokenizer_path" \
            --prompt "$prompt_text" \
            --steps "$steps" \
            --max-seq-len "$max_seq_len" \
            --buffer-float-type "$buffer_float_type" \
            --nthreads "$nthreads" \
            --workers "${workers_args[@]}" \
            --temperature "$temperature" \
            --topp "$topp" \
            >"$run_log" 2>&1; then
            status=0
          else
            status=$?
          fi

          summary_path="$(grep -o '📝 Run summary saved: .*' "$run_log" | tail -n1 | sed 's/^📝 Run summary saved: //')"
          if [[ -z "$summary_path" || ! -f "$summary_path" ]]; then
            summary_path=""
          fi

          append_result "$chunk_size" "$max_seq_len" "$batches" "$nthreads" "$prompt_words" "$steps" "$status" "$summary_path" "$run_log"

          if [[ "$status" -ne 0 ]]; then
            echo "  [WARN] Run failed with status $status: $run_log" >&2
          fi
        done
      done
    done
  done
done

echo "==> Sweep complete"
echo "    manifest: $run_manifest"
echo "    results:  $results_csv"
