#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-cpu-ops.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"
#include <stdexcept>
#include <cmath>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

static bool ensureDir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0)
        return S_ISDIR(st.st_mode);
    return mkdir(path, 0755) == 0;
}

static std::string basenameFromPath(const char *path) {
    if (path == nullptr || path[0] == '\0')
        return "unknown";

    const char *lastSlash = std::strrchr(path, '/');
    const char *name = lastSlash == nullptr ? path : (lastSlash + 1);
    return std::string(name);
}

static std::string sanitizeLabel(const std::string &value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); i++) {
        char c = value[i];
        bool isDigit = c >= '0' && c <= '9';
        bool isLower = c >= 'a' && c <= 'z';
        bool isUpper = c >= 'A' && c <= 'Z';
        if (isDigit || isLower || isUpper || c == '-' || c == '_')
            out.push_back(c);
        else
            out.push_back('_');
    }
    if (out.empty())
        return "unknown";
    return out;
}

static const char *floatTypeToLabel(NnFloatType syncType) {
    switch (syncType) {
        case F_32: return "f32";
        case F_16: return "f16";
        case F_Q40: return "q40";
        case F_Q80: return "q80";
        default: return "unknown";
    }
}

static void inference(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0)
        throw std::runtime_error("Number of steps is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    if (nInputTokens > context->header->seqLen)
        throw std::runtime_error("The number of prompt tokens is greater than the sequence length");
    if (nInputTokens > context->args->steps)
        throw std::runtime_error("The number of prompt tokens is greater than the number of steps");

    NnSize sentBytes = 0;
    NnSize recvBytes = 0;
    NnUint evalTotalTime = 0;
    NnUint predTotalTime = 0;
    std::chrono::steady_clock::time_point runStart = std::chrono::steady_clock::now();

    int token = inputTokens[pos];
    printf("%s\n", context->args->prompt);
    for (;;) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0)
            break;
        NnUint batchSize = remainingTokens < context->args->nBatches
            ? remainingTokens
            : context->args->nBatches;

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; i++)
            context->inference->setToken(i, inputTokens[pos + i]);

        context->inference->forward();

        pos += batchSize;
        token = inputTokens[pos + 1];

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint evalTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔷️ Eval%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | (%d tokens)\n",
            evalTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            batchSize);
        context->executor->printLastForwardHotspots(3);
        evalTotalTime += evalTime + syncTime;
    }

    fflush(stdout);

    context->inference->setBatchSize(1);
    context->tokenizer->resetDecoder();

    const NnUint maxPos = std::min(context->header->seqLen, context->args->steps);
    for (; pos < maxPos; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        context->inference->forward();

        token = context->sampler->sample(context->inference->logitsPipe);

        char *piece = context->tokenizer->decode(token);

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔶 Pred%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | %s\n",
            predTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            piece == nullptr ? "~" : piece);
        context->executor->printLastForwardHotspots(3);
        fflush(stdout);
        predTotalTime += predTime + syncTime;
    }

    context->executor->printTimingDistributions();

    NnUint nEvalTokens = nInputTokens - 1;
    NnUint nPredTokens = pos - nEvalTokens;
    float evalTotalTimeMs = evalTotalTime / 1000.0;
    float predTotalTimeMs = predTotalTime / 1000.0;
    printf("\n");
    printf("Evaluation\n");
    printf("   nBatches: %d\n", context->args->nBatches);
    printf("    nTokens: %d\n", nEvalTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nEvalTokens * 1000) / evalTotalTimeMs,
        evalTotalTimeMs / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("    nTokens: %d\n", nPredTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nPredTokens * 1000) / predTotalTimeMs,
        predTotalTimeMs / ((float) nPredTokens));

    // Persist only end-of-run summaries to a dedicated run folder.
    auto now = std::chrono::system_clock::now();
    std::time_t nowTimeT = std::chrono::system_clock::to_time_t(now);
    std::tm localTime = *std::localtime(&nowTimeT);
    auto msPart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    char timestamp[40];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d-%H%M%S", &localTime);
    std::snprintf(timestamp + std::strlen(timestamp), sizeof(timestamp) - std::strlen(timestamp), "-%03lld", (long long)msPart.count());

    std::string modelName = sanitizeLabel(basenameFromPath(context->args->modelPath));
    std::string runLabel = std::string("run-") + timestamp +
        "_m-" + modelName +
        "_seq-" + std::to_string(context->header->seqLen) +
        "_t-" + std::to_string(context->args->nThreads) +
        "_b-" + std::to_string(context->args->nBatches) +
        "_w-" + std::to_string(context->args->nWorkers) +
        "_s-" + floatTypeToLabel(context->args->syncType) +
        "_chunk-" + std::to_string(MAX_CHUNK_SIZE);

    std::string logsRoot = "logs";
    std::string runDir = logsRoot + "/" + runLabel;

    if (ensureDir(logsRoot.c_str()) && ensureDir(runDir.c_str())) {
        std::string summaryName = std::string("summary_") +
            "seq" + std::to_string(context->header->seqLen) +
            "t" + std::to_string(context->args->nThreads) +
            "_b" + std::to_string(context->args->nBatches) +
            "_w" + std::to_string(context->args->nWorkers) +
            "_sync-" + floatTypeToLabel(context->args->syncType) +
            "_chunk-" + std::to_string(MAX_CHUNK_SIZE) +
            "_" + timestamp + ".log";
        std::string summaryPath = runDir + "/" + summaryName;

        FILE *fp = std::fopen(summaryPath.c_str(), "w");
        if (fp != nullptr) {
            auto runDurationUs = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - runStart
            ).count();
            float runDurationMs = runDurationUs / 1000.0f;

            std::fprintf(fp, "run_timestamp=%s\n", timestamp);
            std::fprintf(fp, "mode=inference\n");
            std::fprintf(fp, "model=%s\n", context->args->modelPath == nullptr ? "" : context->args->modelPath);
            std::fprintf(fp, "tokenizer=%s\n", context->args->tokenizerPath == nullptr ? "" : context->args->tokenizerPath);
            std::fprintf(fp, "threads=%u\n", context->args->nThreads);
            std::fprintf(fp, "batches=%u\n", context->args->nBatches);
            std::fprintf(fp, "workers=%u\n", context->args->nWorkers);
            std::fprintf(fp, "steps=%u\n", context->args->steps);
            std::fprintf(fp, "seq_len=%u\n", context->header->seqLen);
            std::fprintf(fp, "chunk_size=%u\n", (unsigned int)MAX_CHUNK_SIZE);
            std::fprintf(fp, "sync_type=%s\n", floatTypeToLabel(context->args->syncType));
            std::fprintf(fp, "prompt_tokens=%u\n", nEvalTokens);
            std::fprintf(fp, "eval_tokens=%u\n", nEvalTokens);
            std::fprintf(fp, "pred_tokens=%u\n", nPredTokens);
            std::fprintf(fp, "eval_total_ms=%.2f\n", evalTotalTimeMs);
            std::fprintf(fp, "pred_total_ms=%.2f\n", predTotalTimeMs);
            std::fprintf(fp, "eval_tokens_per_s=%.2f\n", (nEvalTokens * 1000) / evalTotalTimeMs);
            std::fprintf(fp, "pred_tokens_per_s=%.2f\n", (nPredTokens * 1000) / predTotalTimeMs);
            std::fprintf(fp, "eval_ms_per_tok=%.2f\n", evalTotalTimeMs / ((float)nEvalTokens));
            std::fprintf(fp, "pred_ms_per_tok=%.2f\n", predTotalTimeMs / ((float)nPredTokens));
            std::fprintf(fp, "sent_kb=%zu\n", sentBytes / 1024);
            std::fprintf(fp, "recv_kb=%zu\n", recvBytes / 1024);
            std::fprintf(fp, "run_total_ms=%.2f\n", runDurationMs);

            std::fclose(fp);
            std::printf("📝 Run summary saved: %s\n", summaryPath.c_str());
        } else {
            std::printf("⚠️ Could not write run summary file: %s\n", summaryPath.c_str());
        }
    } else {
        std::printf("⚠️ Could not create logs directory for run summaries\n");
    }
}

static NnUint readStdin(const char *guide, char *buffer, NnUint size) {
    std::fflush(stdin);
    std::printf("%s", guide);
    if (std::fgets(buffer, size, stdin) != NULL) {
        NnUint length = std::strlen(buffer);
        if (length > 0 && buffer[length - 1] == '\n') {
            buffer[length - 1] = '\0';
            length--;
        }
        return length;
    }
    return 0;
}

static void perplexity(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    printf("Evaluating %d tokens...\n", nInputTokens);

    float totalLogProb = 0.0f;
    NnUint pos = 0;

    context->inference->setBatchSize(1);

    for (pos = 0; pos < nInputTokens - 1; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, inputTokens[pos]);
        context->inference->forward();

        float *logits = context->inference->logitsPipe;
        softmax_F32(logits, context->header->vocabSize);

        int targetToken = inputTokens[pos + 1];
        float prob = logits[targetToken];

        totalLogProb += std::log(std::max(prob, 1e-30f));
        printf("%5d / %d, prob=%f\n", pos + 1, nInputTokens - 1, prob);
    }

    float avgLogProb = totalLogProb / (float)(nInputTokens - 1);
    float perplexity = expf(-avgLogProb);

    printf("\n");
    printf("Results\n");
    printf("   perplexity: %f (lower = better)\n", perplexity);
    printf("   avgLogProb: %f\n", avgLogProb);
    printf("   bitPerToken: %f\n", -avgLogProb / std::log(2.0));
}

static void chat(AppInferenceContext *context) {
    const NnUint seqLen = context->header->seqLen;
    char prompt[2048];

    TokenizerChatStops stops(context->tokenizer);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);

    const NnUint sysPromptLength = readStdin("💻 System prompt (optional): ", prompt, sizeof(prompt));
    std::vector<ChatItem> deltaItems;
    if (sysPromptLength > 0)
        deltaItems.push_back(ChatItem{"system", prompt});

    NnUint pos = 0;
    NnUint userPromptLength;
    int token;
    int nInputTokens;
    do {
        do {
            userPromptLength = readStdin("\n👱 User\n> ", prompt, sizeof(prompt));
        } while (userPromptLength == 0);

        deltaItems.push_back(ChatItem{"user", prompt});

        GeneratedChat inputPrompt = templateGenerator.generate(deltaItems.size(), deltaItems.data(), true);
        std::unique_ptr<int[]> inputTokensPtr(new int[inputPrompt.length + 2]);
        int *inputTokens = inputTokensPtr.get();

        bool isStart = pos == 0;
        context->tokenizer->encode((char*)inputPrompt.content, inputTokens, &nInputTokens, isStart, true);

        NnUint userPromptEndPos = (NnUint)std::min<unsigned int>(seqLen, pos + nInputTokens - 1);
        for (NnUint i = 0; ;) {
            int remainingTokens = userPromptEndPos - pos;
            if (remainingTokens <= 0)
                break;
            NnUint batchSize = remainingTokens < context->args->nBatches
                ? remainingTokens
                : context->args->nBatches;

            context->inference->setBatchSize(batchSize);
            context->inference->setPosition(pos);
            for (NnUint j = 0; j < batchSize; j++)
                context->inference->setToken(j, inputTokens[i + j]);

            context->inference->forward();

            i += batchSize;
            pos += batchSize;
            token = inputTokens[i + 1];
        }

        context->inference->setBatchSize(1);
        context->tokenizer->resetDecoder();

        printf("\n🤖 Assistant\n");
        if (inputPrompt.publicPrompt != nullptr)
            printf("%s", inputPrompt.publicPrompt);

        while (pos < seqLen) {
            context->inference->setPosition(pos);
            context->inference->setToken(0, token);
            context->inference->forward();

            token = context->sampler->sample(context->inference->logitsPipe);

            char *piece = context->tokenizer->decode(token);
            EosDetectorType eosType = eosDetector.append(token, piece);
            if (eosType == NOT_EOS || eosType == EOS) {
                char *delta = eosDetector.getDelta();
                if (delta != nullptr) {
                    printf("%s", delta);
                    fflush(stdout);
                }
                eosDetector.reset();
            }
            pos++;
            if (eosType == EOS) break;
        }

        deltaItems.clear();
    } while (pos < seqLen);

    printf("(end of context)\n");
}

int main(int argc, char **argv) {
    initQuants();
    initSockets();

    int returnCode = EXIT_SUCCESS;
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        if (std::strcmp(args.mode, "inference") == 0) {
            args.benchmark = true;
            runInferenceApp(&args, &inference);
        } else if (std::strcmp(args.mode, "perplexity") == 0)
            runInferenceApp(&args, &perplexity);
        else if (std::strcmp(args.mode, "chat") == 0)
            runInferenceApp(&args, &chat);
        else if (std::strcmp(args.mode, "worker") == 0)
            runWorkerApp(&args);
        else
            throw std::runtime_error("Unsupported mode");
    } catch (const std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        returnCode = EXIT_FAILURE;
    }

    cleanupSockets();
    return returnCode;
}
