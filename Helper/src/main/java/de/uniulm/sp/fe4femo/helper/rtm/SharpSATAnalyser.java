package de.uniulm.sp.fe4femo.helper.rtm;

import de.uniulm.sp.fe4femo.helper.Helper;
import de.uniulm.sp.fe4femo.helper.LineAnalyser;
import de.uniulm.sp.fe4femo.helper.SlurmAnalyser;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SharpSATAnalyser extends SlurmAnalyser {

    private static final Logger LOGGER = LogManager.getLogger();

    public final Map<Integer, BigInteger> modelCount = new HashMap<>();

    public SharpSATAnalyser(Path path) {
        super(path);
    }

    @Override
    public void export(Path outputPath) throws IOException {
        List<Helper.NamedMap> outputMaps = List.of(
                new Helper.NamedMap("modelPath", modelPath),
                new Helper.NamedMap("modelCount", modelCount),
                new Helper.NamedMap("wallclockTimeS", wallClockInner),
                new Helper.NamedMap("weirdWallTimeS", wallClockOuter),
                new Helper.NamedMap("memUseMB", jobMemMb)
        );
        Helper.exportCSV(outputPath.resolve(toolName+".csv"), modelPath.keySet(), outputMaps);
    }

    @Override
    public LineAnalyser accumulate(LineAnalyser lineAnalyser) {
        if (!this.getClass().isAssignableFrom(lineAnalyser.getClass())) {
            LOGGER.error("Non-fitting line analyser in accumulation! Current {} is not compatible with given {}", this.getClass().getSimpleName(), lineAnalyser.getClass().getSimpleName());
            throw new RuntimeException("Non-fitting line analyser in accumulation!");
        }
        SharpSATAnalyser analyser = (SharpSATAnalyser) super.accumulate(lineAnalyser);
        analyser.modelCount.putAll(modelCount);
        return analyser;
    }

    @Override
    public void handleLine(String line) {
        super.handleLine(line);

        if (modelNumber != -1 && !toolName.isEmpty()) {
            String prefix = switch (toolName) {
                case "approxmc" -> "s mc ";
                case "countantom" -> "c model count............: ";
                case "d4v2_23", "d4v2_24" -> "s ";
                case "exactmc_arjun", "ganak", "sharpsattd" -> "c s exact arb int ";
                default ->{
                    LOGGER.error("Unknown tool name '{}' when handling file '{}'", toolName, path);
                    yield "°°°";
                }
            };
            try {
                Helper.returnSuffix(line, prefix).ifPresent(suffix -> modelCount.put(modelNumber, new BigInteger(suffix)));
            } catch (NumberFormatException e) {
                LOGGER.warn("Parsing Error in line {}", line, e);
            }
        }
    }
}
