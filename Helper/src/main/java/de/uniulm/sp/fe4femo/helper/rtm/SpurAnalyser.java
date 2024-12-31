package de.uniulm.sp.fe4femo.helper.rtm;

import de.uniulm.sp.fe4femo.helper.Helper;
import de.uniulm.sp.fe4femo.helper.LineAnalyser;
import de.uniulm.sp.fe4femo.helper.SlurmAnalyser;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class SpurAnalyser extends SlurmAnalyser {

    public SpurAnalyser(Path path) {
        super(path);
    }

    @Override
    public void handleLine(String line) {
        super.handleLine(line);
    }

    @Override
    public LineAnalyser accumulate(LineAnalyser lineAnalyser) {
        return super.accumulate(lineAnalyser);
    }

    @Override
    public void export(Path outputPath) throws IOException {
        List<Helper.NamedMap> outputMaps = List.of(
                new Helper.NamedMap("modelPath", modelPath),
                new Helper.NamedMap("wallclockTimeS", wallClockInner),
                new Helper.NamedMap("weirdWallTimeS", wallClockOuter),
                new Helper.NamedMap("memUseMB", jobMemMb)
        );
        Helper.exportCSV(outputPath, modelPath.keySet(), outputMaps);
    }
}
