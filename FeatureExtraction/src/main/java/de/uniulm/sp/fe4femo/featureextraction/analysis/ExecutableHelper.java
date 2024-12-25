package de.uniulm.sp.fe4femo.featureextraction.analysis;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class ExecutableHelper {

    private static final Logger LOGGER = LogManager.getLogger();

    private ExecutableHelper() {
        // hide
    }

    //adapted from https://github.com/SundermannC/feature-model-batch-analysis/blob/main/src/main/java/org/collection/fm/util/BinaryRunner.java
    public static ExternalResult executeExternal(String[] commands, int timeout, Path workingDir) throws InterruptedException {
        Process ps = null;
        try {
            ps = new ProcessBuilder(commands).redirectErrorStream(true).directory(workingDir.toFile()).start();

            Process finalPs = ps;
            Future<String> output = ForkJoinPool.commonPool().submit(() -> {
                StringBuilder val = new StringBuilder();
                String line;
                try (BufferedReader in = new BufferedReader(new InputStreamReader(finalPs.getInputStream()))) {
                    while ((line = in.readLine()) != null) {
                        val.append(line).append("\n");
                    }
                }
                return val.toString();
            });

            if (!ps.waitFor(timeout, TimeUnit.SECONDS)) {
                killProcesses(ps.toHandle());
                return new ExternalResult("", StatusEnum.TIMEOUT);
            }

            if (ps != null)	killProcesses(ps.toHandle());
            return new ExternalResult(output.get(), StatusEnum.SUCCESS);
        } catch (IOException e) {
            if (ps != null)	killProcesses(ps.toHandle());
            return new ExternalResult("", StatusEnum.ERROR);
        } catch (InterruptedException e) {
            if (ps != null)	killProcesses(ps.toHandle());
            throw new InterruptedException();
        } catch (ExecutionException e) {
            if (ps != null)	killProcesses(ps.toHandle());
            LOGGER.error("Error reading from external command {}", commands, e);
            return new ExternalResult("", StatusEnum.ERROR);
        }

    }

    private static void killProcesses(ProcessHandle ps)  {
        ps.descendants().forEach(ExecutableHelper::killProcesses);
        ps.destroy();
    }

    public record ExternalResult(String output, StatusEnum status) {

    }



}
