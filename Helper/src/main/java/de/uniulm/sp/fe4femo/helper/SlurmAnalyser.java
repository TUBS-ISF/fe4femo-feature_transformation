package de.uniulm.sp.fe4femo.helper;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;


public abstract class SlurmAnalyser implements LineAnalyser{

    private static final Logger LOGGER = LogManager.getLogger();

    protected final Path path;

    protected int modelNumber = -1;
    protected final Map<Integer, String> modelPath = new HashMap<>();
    protected String toolName = "";
    protected final Map<Integer, BigDecimal> wallClockInner = new HashMap<>();
    protected final Map<Integer, BigDecimal> userClockInner = new HashMap<>();
    protected final Map<Integer, BigDecimal> sysClockInner = new HashMap<>();
    protected final Map<Integer, BigDecimal> wallClockOuter = new HashMap<>();
    protected final Map<Integer, BigDecimal> jobTime = new HashMap<>();
    protected final Map<Integer, BigDecimal> cpuUtilizationTime = new HashMap<>();
    protected final Map<Integer, BigDecimal> jobMemMb = new HashMap<>();

    protected SlurmAnalyser(Path path) {
        this.path = path;
        //TODO maybe parse modelNo here?
    }


    @Override
    public void handleLine(String line) {
        try {
            String[] equalSplit = line.trim().split("=");
            switch (equalSplit[0]) {
                case "MODEL_NUMBER" -> modelNumber = Integer.parseInt(equalSplit[1]);
                case "MODEL_PATH" -> modelPath.put(modelNumber, equalSplit[1]);
                case "TOOL_NAME" -> {
                    String[] pathSplit = equalSplit[1].split("/");
                    toolName = pathSplit[pathSplit.length -1].replace("_i.sqsh", "");
                }
                case "REALTIME" -> wallClockInner.put(modelNumber, new BigDecimal(equalSplit[1]));
                case "USERTIME" -> userClockInner.put(modelNumber, new BigDecimal(equalSplit[1]));
                case "SYSTIME" -> sysClockInner.put(modelNumber, new BigDecimal(equalSplit[1]));
                case "TS_RUNTIME" -> wallClockOuter.put(modelNumber, new BigDecimal(equalSplit[1]));
                default -> {} //ignore
            }
            Helper.returnSuffix(line, "Job Wall-clock time: ").ifPresent(e -> {
                LocalTime time = LocalTime.parse(e.trim(), DateTimeFormatter.ofPattern("HH:mm:ss"));
                jobTime.put(modelNumber, new BigDecimal(time.toSecondOfDay()));
            });
            Helper.returnSuffix(line, "CPU Utilized: ").ifPresent(e -> {
                LocalTime time = LocalTime.parse(e.trim(), DateTimeFormatter.ofPattern("HH:mm:ss"));
                cpuUtilizationTime.put(modelNumber, new BigDecimal(time.toSecondOfDay()));
            });
            Helper.returnSuffix(line, "Memory Utilized: ").ifPresent(e -> {
                String[] split = e.split(" ");
                BigDecimal amount = new BigDecimal(split[0]);
                if (Objects.equals(split[1], "GB")) amount = amount.multiply(BigDecimal.valueOf(1024));
                jobMemMb.put(modelNumber, amount);
            });
        } catch (NumberFormatException e) {
            LOGGER.warn("Parsing Error in line {}", line, e);
        }
    }

    @Override
    public LineAnalyser accumulate(LineAnalyser lineAnalyser) {
        if (!this.getClass().isAssignableFrom(lineAnalyser.getClass())) {
            LOGGER.error("Non-fitting line analyser in accumulation! Current {} is not compatible with given {}", this.getClass().getSimpleName(), lineAnalyser.getClass().getSimpleName());
            throw new RuntimeException("Non-fitting line analyser in accumulation!");
        }
        SlurmAnalyser other = (SlurmAnalyser) lineAnalyser;
        other.modelPath.putAll(modelPath);
        if (!toolName.isEmpty()) other.toolName = toolName;
        other.wallClockInner.putAll(wallClockInner);
        other.userClockInner.putAll(userClockInner);
        other.sysClockInner.putAll(sysClockInner);
        other.wallClockOuter.putAll(wallClockOuter);
        other.jobTime.putAll(jobTime);
        other.cpuUtilizationTime.putAll(cpuUtilizationTime);
        other.jobMemMb.putAll(jobMemMb);
        return other;
    }

}
