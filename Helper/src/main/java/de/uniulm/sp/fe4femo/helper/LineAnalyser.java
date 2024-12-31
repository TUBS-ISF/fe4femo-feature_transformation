package de.uniulm.sp.fe4femo.helper;

import java.io.IOException;
import java.nio.file.Path;

public interface LineAnalyser {

    void handleLine(String line);

    LineAnalyser accumulate(LineAnalyser lineAnalyser);

    void export(Path outputPath) throws IOException;

}
