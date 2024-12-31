package de.uniulm.sp.fe4femo.helper;

import de.uniulm.sp.fe4femo.helper.rtm.KissatAnalyser;
import de.uniulm.sp.fe4femo.helper.rtm.SharpSATAnalyser;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class Main {

    private static final Logger LOGGER = LogManager.getLogger();

    public static void main(String[] args) {
        if (args.length < 1) {
            LOGGER.error("Too few CLI parameters!");
            System.exit(1);
        }
        switch (args[0]){
            case "sharpsat" -> handleSharpSAT(args);
            case "kissat" -> handleNormal(args, KissatAnalyser::new);
            default -> LOGGER.error("Unrecognized command {}", args[0]);
        }
    }

    private static void handleSharpSAT(String[] args) {
        if (args.length < 3) {
            LOGGER.error("Too few CLI parameters! Supply <type> <outputFolder> <inputFolder> ... <inputFolder>");
            System.exit(1);
        }
        Path outputFolder = Paths.get(args[1]);
        List<SharpSATAnalyser> analyserList = Arrays.stream(args).skip(2).map(Path::of).map(path -> (SharpSATAnalyser) Helper.analyseAll(path, SharpSATAnalyser::new)).toList();
        try {
            List<Helper.NamedMap> exportList = new ArrayList<>();
            exportList.add(new Helper.NamedMap("modelPath", analyserList.getFirst().modelPath));
            for (SharpSATAnalyser analyser : analyserList) {
                analyser.export(outputFolder);
                String prefix = analyser.toolName + "_";
                exportList.add(new Helper.NamedMap(prefix + "modelCount", analyser.modelCount));
                exportList.add(new Helper.NamedMap(prefix + "wallclockTimeS", analyser.wallClockInner));
                exportList.add(new Helper.NamedMap(prefix + "weirdWallTimeS", analyser.wallClockOuter));
                exportList.add(new Helper.NamedMap(prefix + "memUseMB", analyser.jobMemMb));
            }
            Helper.exportCSV(outputFolder.resolve("sharpsat.csv"), analyserList.getFirst().modelPath.keySet(), exportList);
        } catch (IOException e) {
            LOGGER.error("Could not export analysis results", e);
            System.exit(1);
        }

    }

    private static void handleNormal(String[] args, Function<Path, LineAnalyser> analyserGenerator) {
        if (args.length != 3) {
            LOGGER.error("Too few CLI parameters! Supply <type> <inputFolder> <outputFolder>");
            System.exit(1);
        }
        Path inputPath = Paths.get(args[1]);
        Path outputPath = Paths.get(args[2]);

        LineAnalyser analyser = Helper.analyseAll(inputPath, analyserGenerator);
        try {
            Files.createDirectories(outputPath);
            analyser.export(outputPath);
        } catch (IOException e) {
            LOGGER.error("Could not export analysis results", e);
            System.exit(1);
        }
    }


}
