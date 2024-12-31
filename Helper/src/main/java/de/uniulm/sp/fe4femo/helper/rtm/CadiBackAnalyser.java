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

public class CadiBackAnalyser extends SlurmAnalyser {

    private static final Logger LOGGER = LogManager.getLogger();

    protected Map<Integer, BigInteger> backBoneSize = new HashMap<>();

    public CadiBackAnalyser(Path path) {
        super(path);
    }

    @Override
    public void export(Path outputPath) throws IOException {
        List<Helper.NamedMap> outputMaps = List.of(
                new Helper.NamedMap("modelPath", modelPath),
                new Helper.NamedMap("backboneSize", backBoneSize),
                new Helper.NamedMap("wallclockTimeS", wallClockInner),
                new Helper.NamedMap("weirdWallTimeS", wallClockOuter),
                new Helper.NamedMap("memUseMB", jobMemMb)
        );
        Helper.exportCSV(outputPath.resolve("backbone.csv"), modelPath.keySet(), outputMaps);
    }

    @Override
    public LineAnalyser accumulate(LineAnalyser lineAnalyser) {
        if (!this.getClass().isAssignableFrom(lineAnalyser.getClass())) {
            LOGGER.error("Non-fitting line analyser in accumulation! Current {} is not compatible with given {}", this.getClass().getSimpleName(), lineAnalyser.getClass().getSimpleName());
            throw new RuntimeException("Non-fitting line analyser in accumulation!");
        }
        CadiBackAnalyser analyser = (CadiBackAnalyser) super.accumulate(lineAnalyser);
        analyser.backBoneSize.putAll(backBoneSize);
        return analyser;
    }

    @Override
    public void handleLine(String line) {
        super.handleLine(line);
        if (modelNumber != -1 && line.startsWith("b ") && !line.equals("b 0")) {
            BigInteger newValue = backBoneSize.getOrDefault(modelNumber, BigInteger.ZERO).add(BigInteger.ONE);
            backBoneSize.put(modelNumber, newValue);
        }
    }
}
