package de.uniulm.sp.fe4femo.helper.rtm;

import de.uniulm.sp.fe4femo.helper.Helper;
import de.uniulm.sp.fe4femo.helper.LineAnalyser;
import de.uniulm.sp.fe4femo.helper.SlurmAnalyser;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KissatAnalyser extends SlurmAnalyser {

    private static final Logger LOGGER = LogManager.getLogger();

    protected Map<Integer, Boolean> satisfiability = new HashMap<>();

    public KissatAnalyser(Path path) {
        super(path);
    }


    @Override
    public void export(Path outputPath) throws IOException {
        List<Helper.NamedMap> outputMaps = List.of(
                new Helper.NamedMap("modelPath", modelPath),
                new Helper.NamedMap("isSAT", satisfiability),
                new Helper.NamedMap("wallclockTimeS", wallClockInner),
                new Helper.NamedMap("memUseMB", jobMemMb)
        );
        Helper.exportCSV(outputPath.resolve("sat.csv"), satisfiability.keySet(), outputMaps);
    }

    @Override
    public void handleLine(String line) {
        super.handleLine(line);
        if (line.equalsIgnoreCase("s SATISFIABLE")) satisfiability.put(modelNumber, true);
    }

    @Override
    public LineAnalyser accumulate(LineAnalyser lineAnalyser) {
        if (!this.getClass().isAssignableFrom(lineAnalyser.getClass())) {
            LOGGER.error("Non-fitting line analyser in accumulation! Current {} is not compatible with given {}", this.getClass().getSimpleName(), lineAnalyser.getClass().getSimpleName());
            throw new RuntimeException("Non-fitting line analyser in accumulation!");
        }
        KissatAnalyser analyser = (KissatAnalyser) super.accumulate(lineAnalyser);
        analyser.satisfiability.putAll(satisfiability);
        return analyser;
    }
}
