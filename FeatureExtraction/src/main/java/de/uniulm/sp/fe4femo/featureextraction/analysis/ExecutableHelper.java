package de.uniulm.sp.fe4femo.featureextraction.analysis;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
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
            if (!ps.waitFor(timeout, TimeUnit.SECONDS)) {
                killProcesses(ps.toHandle());
                return new ExternalResult("", StatusEnum.TIMEOUT);
            }

            StringBuilder val = new StringBuilder();
            String line;
            try (BufferedReader in = new BufferedReader(new InputStreamReader(ps.getInputStream()))) {
                while ((line = in.readLine()) != null) {
                    val.append(line).append("\n");
                }
            }
            if (ps != null)	killProcesses(ps.toHandle());
            return new ExternalResult(val.toString(), StatusEnum.SUCCESS);
        } catch (IOException e) {
            if (ps != null)	killProcesses(ps.toHandle());
            return new ExternalResult("", StatusEnum.ERROR);
        } catch (InterruptedException e) {
            if (ps != null)	killProcesses(ps.toHandle());
            throw new InterruptedException();
        }

    }

    private static void killProcesses(ProcessHandle ps)  {
        ps.descendants().forEach(ExecutableHelper::killProcesses);
        ps.destroy();
    }

    public record ExternalResult(String output, StatusEnum status) {

    }



}
