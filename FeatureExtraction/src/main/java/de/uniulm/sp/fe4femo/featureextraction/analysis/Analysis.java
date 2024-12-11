package de.uniulm.sp.fe4femo.featureextraction.analysis;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;

public abstract class Analysis {

    private static final Logger LOGGER = LogManager.getLogger();

    protected final ExecutorService executor;
    protected final String name;
    protected final List<AnalysisStep> analysisSteps;

    protected Analysis(String name, ExecutorService executor, List<AnalysisStep> analysisSteps) {
        this.executor = executor;
        this.name = name;
        this.analysisSteps = analysisSteps;
    }


    public List<Result> analyseFM(FMInstance instance, int perStepTimeout) throws InterruptedException {
        LOGGER.info("Started analysis {} on FM {}", this, instance);

        List<Result> results = new ArrayList<>(analysisSteps.size());
        List<Future<Result>> futures = analysisSteps.stream().map(e -> executor.submit(() -> limitRuntime(instance, perStepTimeout, e))).toList();
        for (Future<Result> f : futures) {
            try {
                results.add(f.get());
            } catch (InterruptedException e) {
                LOGGER.info("Interrupted on Wrapper Future {} of FM instance {}", f, instance, e);
                throw new InterruptedException();
            } catch (ExecutionException e) {
                LOGGER.error("Error in Wrapper Future of FM {} on future {}", instance, f, e);
            }
        }
        return results;
    }

    protected Result limitRuntime(FMInstance instance, int perStepTimeout, AnalysisStep analysisStep) {
        ExecutorService commonExecutor = ForkJoinPool.commonPool();
        Instant startTime = Instant.now();
        Future<IntraStepResult> future = null;
        try {
            future = commonExecutor.submit(() -> analysisStep.analyze(instance, perStepTimeout, this));
            IntraStepResult intraStepResult = future.get(perStepTimeout, TimeUnit.SECONDS);
            Instant endTime = Instant.now();
            return new Result(name, intraStepResult, Duration.between(startTime, endTime));
        } catch (ExecutionException e) {
            LOGGER.warn("Error on step {} of FM instance {}", analysisStep, instance, e);
        } catch (InterruptedException e) {
            LOGGER.info("Interrupted on step {} of FM instance {}", analysisStep, instance, e);
            future.cancel(true);
            Thread.currentThread().interrupt();
        } catch (TimeoutException e) {
            LOGGER.debug("Timeout on step {} of FM instance {}", analysisStep, instance, e);
            future.cancel(true);
            return new Result(name, Map.of(), StatusEnum.TIMEOUT, Duration.between(startTime, Instant.now()));
        }
        return new Result(name, Map.of(), StatusEnum.ERROR, Duration.between(startTime, Instant.now()));
    }

}
