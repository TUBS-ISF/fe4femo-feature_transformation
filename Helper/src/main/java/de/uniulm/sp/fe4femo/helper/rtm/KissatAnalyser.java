package de.uniulm.sp.fe4femo.helper.rtm;

import de.uniulm.sp.fe4femo.helper.LineAnalyser;
import de.uniulm.sp.fe4femo.helper.SlurmAnalyser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public class KissatAnalyser extends SlurmAnalyser {

    private static final Logger LOGGER = LogManager.getLogger();

    protected Map<Integer, Boolean> satisfiability;


    @Override
    public void export(Path outputPath) throws IOException { //TODO reuse common parts
        try (CSVPrinter printer = new CSVPrinter(Files.newBufferedWriter(outputPath), CSVFormat.DEFAULT)){
            printer.printRecord("ModelNo", "ModelPath", "isSAT", "wallclockTimeS", "memUseMB");
            int[] keys = satisfiability.keySet().stream().mapToInt(Integer::intValue).sorted().toArray();
            for (int i : keys) {
                printer.printRecord(i, modelPath.get(i), satisfiability.get(i), wallClockInner.get(i), jobMemMb.get(i));
            }
        }
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
