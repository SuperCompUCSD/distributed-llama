#include <cassert>
#include <cstring>
#include <algorithm>
#include <limits>
#include "nn-executor.hpp"

void NnFakeNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    // Nothing
}

NnNetExecution::NnNetExecution(NnUint nThreads, NnNetConfig *netConfig) {
    this->nThreads = nThreads;
    this->nBatches = netConfig->nBatches;
    this->nPipes = netConfig->nPipes;
    this->batchSize = 0; // This value must be overwritten before calling forward

    pipes = new NnByte *[netConfig->nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &netConfig->pipes[pipeIndex];
        NnByte *pipe = new NnByte[pipeConfig->size.nBytes];
        std::memset(pipe, 0, pipeConfig->size.nBytes);
        pipes[pipeIndex] = pipe;
    }
}

NnNetExecution::~NnNetExecution() {
    for (NnUint pipeIndex = 0; pipeIndex < nPipes; pipeIndex++)
        delete[] pipes[pipeIndex];
    delete[] pipes;
}

void NnNetExecution::setBatchSize(NnUint batchSize) {
    assert(batchSize <= nBatches);
    this->batchSize = batchSize;
}

NnExecutorDevice::NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo) {
    this->device = std::unique_ptr<NnDevice>(device);
    this->segmentFrom = segmentFrom;
    this->segmentTo = segmentTo;
}

NnExecutorException::NnExecutorException(const std::string message)
    : std::runtime_error(message) 
{}

NnExecutor::NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *devices, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark)
    : segments(nodeConfig->nSegments), steps(), stepLabels(), lastStepTime(), stepCount(), stepTotalTime(), stepMinTime(), stepMaxTime(), stepSamples(), nForwards(0), profilingEnabled(benchmark)
{
    NnUint maxNThreads = 0;
    for (NnExecutorDevice &d : *devices) {
        if (d.device->maxNThreads() > maxNThreads)
            maxNThreads = d.device->maxNThreads();
    }
    if (netExecution->nThreads > maxNThreads)
        throw std::invalid_argument("This configuration supports max " + std::to_string(maxNThreads) + " threads");

    this->netExecution = netExecution;
    this->nodeConfig = nodeConfig;

    bool useSynchronizer = netConfig->nNodes > 1;
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDevice *device = nullptr;
        for (NnExecutorDevice &d : *devices) {
            if (
                (d.segmentFrom == -1 && d.segmentTo == -1) ||
                (segmentIndex >= d.segmentFrom && segmentIndex <= d.segmentTo)
            ) {
                device = d.device.get();
                break;
            }
        }
        if (device == nullptr)
            throw std::invalid_argument("Cannot locate device for segment " + std::to_string(segmentIndex));

        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        if (segmentConfig->nOps > 0) {
            NnDeviceSegment *segment = device->createSegment(segmentIndex);
            segments[segmentIndex] = std::unique_ptr<NnDeviceSegment>(segment);

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++)
                steps.push_back(NnExecutorStep{ STEP_EXECUTE_OP, segment, opIndex, &segmentConfig->ops[opIndex] });
        }
        if (useSynchronizer && segmentConfig->nSyncs > 0)
            steps.push_back(NnExecutorStep{ STEP_SYNC_NODES, nullptr, segmentIndex, nullptr });
    }

    steps.shrink_to_fit();

    stepLabels.resize(steps.size());
    lastStepTime.resize(steps.size(), 0);
    stepCount.resize(steps.size(), 0);
    stepTotalTime.resize(steps.size(), 0);
    stepMinTime.resize(steps.size(), std::numeric_limits<NnUint>::max());
    stepMaxTime.resize(steps.size(), 0);
    stepSamples.resize(steps.size());
    for (NnUint i = 0; i < (NnUint)steps.size(); i++)
        stepLabels[i] = formatStepLabel(steps[i]);

    context.nThreads = netExecution->nThreads;
    context.synchronizer = synchronizer;
    context.nSteps = (NnUint)steps.size();
    context.steps = steps.data();
    context.lastStepTime = lastStepTime.data();
    context.stepCount = stepCount.data();
    context.stepTotalTime = stepTotalTime.data();
    context.stepMinTime = stepMinTime.data();
    context.stepMaxTime = stepMaxTime.data();
    context.stepSamples = stepSamples.data();
    if (profilingEnabled)
        context.timer = new Timer();
    else
        context.timer = nullptr;

    threads = new NnExecutorThread[netExecution->nThreads];
    for (NnUint threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        NnExecutorThread *thread = &threads[threadIndex];
        thread->threadIndex = threadIndex;
        thread->context = &context;
    }
}

NnExecutor::~NnExecutor() {
    if (context.timer != nullptr)
        delete context.timer;
    delete[] threads;
}

void NnExecutor::loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        for (NnUint i = 0; i < segmentConfig->nOps; i++) {
            NnOpConfig *opConfig = &segmentConfig->ops[i];
            if (opConfig->index == opIndex && std::strcmp(opConfig->name, name) == 0) {
                NnDeviceSegment *segment = segments[segmentIndex].get();
                assert(segment != nullptr);
                segment->loadWeight(i, offset, nBytes, weight);
                return;
            }
        }
    }
    throw std::invalid_argument("Cannot locate op by name: " + std::string(name));
}

inline void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context) {
    if (step->type == STEP_EXECUTE_OP) {
        step->segment->forward(step->arg0, nThreads, thread->threadIndex, context->batchSize);
    } else if (step->type == STEP_SYNC_NODES) {
        context->synchronizer->sync(step->arg0, nThreads, thread->threadIndex);
    } else {
        throw std::invalid_argument("Unsupported step type");
    }
}

static inline void *executorThreadHandler(void *arg) {
    NnExecutorThread *thread = (NnExecutorThread *)arg;
    NnExecutorContext *context = thread->context;
    NnUint nThreads = context->nThreads;
    NnUint doneCount = nThreads - 1;

    while (context->isAlive.load()) {
        const unsigned int currentStepIndex = context->currentStepIndex.load();
        if (currentStepIndex == context->nSteps)
            break;

        NnExecutorStep *step = &context->steps[currentStepIndex];
        try {
            executeStep(step, nThreads, thread, context);
        } catch (const std::runtime_error &e) {
            context->isAlive.store(false);
            printf("🚨 Execution error: %s\n", e.what());
            break;
        }

        NnUint currentCount = context->doneThreadCount.fetch_add(1);
        if (currentCount == doneCount) {
            if (context->timer != nullptr) {
                NnUint time = context->timer->elapsedMicroseconds();
                context->totalTime[step->type] += time;
                context->lastStepTime[currentStepIndex] = time;
                context->stepCount[currentStepIndex] += 1;
                context->stepTotalTime[currentStepIndex] += time;
                if (time < context->stepMinTime[currentStepIndex])
                    context->stepMinTime[currentStepIndex] = time;
                if (time > context->stepMaxTime[currentStepIndex])
                    context->stepMaxTime[currentStepIndex] = time;
                context->stepSamples[currentStepIndex].push_back(time);
                context->timer->reset();
            }

            context->doneThreadCount.store(0);
            context->currentStepIndex.fetch_add(1);
        } else {
            while (
                context->currentStepIndex.load() == currentStepIndex &&
                context->isAlive.load()
            );
        }
    }
    return nullptr;
}

void NnExecutor::forward() {
    assert(netExecution->batchSize > 0);

    NnUint nThreads = netExecution->nThreads;
    context.isAlive.exchange(true);
    context.currentStepIndex.exchange(0);
    context.doneThreadCount.exchange(0);
    context.batchSize = netExecution->batchSize;

    if (context.timer != nullptr) {
        nForwards += 1;
        std::memset(context.totalTime, 0, sizeof(context.totalTime));
        std::memset(context.lastStepTime, 0, sizeof(NnUint) * context.nSteps);
        context.timer->reset();
    }

    NnUint threadIndex;
    for (threadIndex = 1; threadIndex < nThreads; threadIndex++) {
        int result = pthread_create(&threads[threadIndex].handler, NULL, (PthreadFunc)executorThreadHandler, (void *)&threads[threadIndex]);
        assert(result == 0 && "Failed to create thread");
    }
    executorThreadHandler((void *)&threads[0]);
    for (threadIndex = 1; threadIndex < nThreads; threadIndex++)
        pthread_join(threads[threadIndex].handler, NULL);

    if (!context.isAlive.load())
        throw NnExecutorException("Execution failed in one of the threads");
}

NnUint NnExecutor::getTotalTime(NnExecutorStepType type) {
    assert((NnUint)type < N_STEP_TYPES);
    return context.totalTime[type];
}

std::string NnExecutor::formatStepLabel(const NnExecutorStep &step) const {
    if (step.type == STEP_EXECUTE_OP) {
        assert(step.opConfig != nullptr);
        if (std::strncmp(step.opConfig->name, "block_", 6) == 0)
            return "L" + std::to_string(step.opConfig->index) + ":" + std::string(step.opConfig->name);
        return std::string(step.opConfig->name);
    }

    NnSegmentConfig *segmentConfig = &nodeConfig->segments[step.arg0];
    if (segmentConfig->nOps > 0) {
        NnOpConfig *opConfig = &segmentConfig->ops[0];
        if (std::strncmp(opConfig->name, "block_", 6) == 0)
            return "SYNC L" + std::to_string(opConfig->index) + ":" + std::string(opConfig->name);
        return "SYNC " + std::string(opConfig->name);
    }
    return "SYNC segment_" + std::to_string(step.arg0);
}

void NnExecutor::printLastForwardHotspots(NnUint topK) const {
    if (context.timer == nullptr || context.nSteps == 0 || topK == 0)
        return;

    std::vector<std::pair<NnUint, NnUint>> compute;
    std::vector<std::pair<NnUint, NnUint>> sync;
    compute.reserve(context.nSteps);
    sync.reserve(context.nSteps);

    for (NnUint i = 0; i < context.nSteps; i++) {
        NnUint us = lastStepTime[i];
        if (us == 0)
            continue;
        if (steps[i].type == STEP_EXECUTE_OP)
            compute.push_back({us, i});
        else
            sync.push_back({us, i});
    }

    auto byTimeDesc = [](const std::pair<NnUint, NnUint> &a, const std::pair<NnUint, NnUint> &b) {
        return a.first > b.first;
    };
    std::sort(compute.begin(), compute.end(), byTimeDesc);
    std::sort(sync.begin(), sync.end(), byTimeDesc);

    NnUint computeCount = std::min(topK, (NnUint)compute.size());
    NnUint syncCount = std::min(topK, (NnUint)sync.size());

    if (computeCount > 0) {
        printf("🧩 SlowCompute");
        for (NnUint i = 0; i < computeCount; i++) {
            NnUint us = compute[i].first;
            NnUint stepIndex = compute[i].second;
            printf(" | %s=%4.2fms", stepLabels[stepIndex].c_str(), us / 1000.0f);
        }
        printf("\n");
    }

    if (syncCount > 0) {
        printf("🔗 SlowSync   ");
        for (NnUint i = 0; i < syncCount; i++) {
            NnUint us = sync[i].first;
            NnUint stepIndex = sync[i].second;
            printf(" | %s=%4.2fms", stepLabels[stepIndex].c_str(), us / 1000.0f);
        }
        printf("\n");
    }
}

static NnUint pctl(const std::vector<NnUint> &values, float ratio) {
    if (values.empty())
        return 0;
    std::vector<NnUint> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    size_t index = (size_t)((sorted.size() - 1) * ratio);
    return sorted[index];
}

void NnExecutor::printStepDistribution(const char *title, bool includeSync) const {
    printf("%s\n", title);
    printf("%-45s %8s %8s %8s %8s %8s\n", "Stage", "avg(ms)", "p50", "p95", "max", "count");
    for (NnUint i = 0; i < context.nSteps; i++) {
        bool isSync = steps[i].type == STEP_SYNC_NODES;
        if (isSync != includeSync)
            continue;

        NnUint count = stepCount[i];
        if (count == 0)
            continue;

        float avg = (float)stepTotalTime[i] / (float)count / 1000.0f;
        float p50 = pctl(stepSamples[i], 0.50f) / 1000.0f;
        float p95 = pctl(stepSamples[i], 0.95f) / 1000.0f;
        float max = stepMaxTime[i] / 1000.0f;
        printf("%-45s %8.3f %8.3f %8.3f %8.3f %8u\n",
            stepLabels[i].c_str(),
            avg,
            p50,
            p95,
            max,
            count);
    }
}

void NnExecutor::printTimingDistributions() const {
    if (context.timer == nullptr)
        return;
    printf("\n📊 Timing distribution across %u forward passes\n", nForwards);
    printStepDistribution("Compute stages", false);
    printStepDistribution("Sync stages", true);
}
