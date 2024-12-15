package de.uniulm.sp.fe4femo.featureextraction.analysis;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

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
        return analyseFM(instance, perStepTimeout, perStepTimeout + 1);
    }

    public List<Result> analyseFM(FMInstance instance, int perStepTimeout, int outerTimeout) throws InterruptedException {
        LOGGER.info("Started analysis {} on FM {}", this, instance);

        List<Result> results = new ArrayList<>(analysisSteps.size());
        List<Identifiable> futures = analysisSteps.stream().map(e -> new Identifiable(e, executor.submit(() -> limitRuntime(instance, perStepTimeout, outerTimeout, e)))).toList();
        for (Identifiable f : futures) {
            try {
                Result result = f.futureResult().get();

                //add mark for missing values
                Map<String, String> featureValues = HashMap.newHashMap(result.featureValues().size());
                result.featureValues().forEach((k,v) -> featureValues.put(result.analysisName()+"/"+k, v));
                Arrays.stream(f.analysisStep.getAnalysesNames()).forEach(e -> featureValues.putIfAbsent(result.analysisName()+"/"+e, "<NA>"));

                results.add(new Result(result.analysisName(), featureValues, result.statusEnum(), result.duration()));
            } catch (InterruptedException e) {
                LOGGER.info("Interrupted on Wrapper Future {} of FM instance {}", f, instance, e);
                throw new InterruptedException();
            } catch (ExecutionException e) {
                LOGGER.error("Error in Wrapper Future of FM {} on future {}", instance, f, e);
            }
        }
        return results;
    }

    protected Result limitRuntime(FMInstance instance, int perStepTimeout, int outerTimeout, AnalysisStep analysisStep) {
        ExecutorService commonExecutor = ForkJoinPool.commonPool();
        Instant startTime = Instant.now();
        Future<IntraStepResult> future = null;
        try {
            future = commonExecutor.submit(() -> analysisStep.analyze(instance, perStepTimeout));
            IntraStepResult intraStepResult = future.get(outerTimeout, TimeUnit.SECONDS);
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

    private record Identifiable (AnalysisStep analysisStep, Future<Result> futureResult) {

    }

}
