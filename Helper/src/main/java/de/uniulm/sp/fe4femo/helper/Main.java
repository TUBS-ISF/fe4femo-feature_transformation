package de.uniulm.sp.fe4femo.helper;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.function.Function;

public class Main {

    private static final Logger LOGGER = LogManager.getLogger();


    private static Map<String, Function<Path, LineAnalyser>> typeMap = Map.of(

    );

    public static void main(String[] args) {
        if (args.length != 3) {
            LOGGER.error("Too few CLI parameters! Supply <type> <inputFolder> <outputFolder>");
            System.exit(1);
        }

        String type = args[0];
        Path inputPath = Paths.get(args[1]);
        Path outputPath = Paths.get(args[2]);

        Function<Path, LineAnalyser> analyserGenerator = typeMap.get(type);
        if (analyserGenerator == null) {
            LOGGER.error("Unknown analysis type {}", type);
            LOGGER.info("Use one of the following: {}", typeMap.keySet());
            System.exit(1);
        }

        LineAnalyser analyser = Helper.analyseAll(inputPath, analyserGenerator);
        try {
            analyser.export(outputPath);
        } catch (IOException e) {
            LOGGER.error("Could not export analysis results", e);
            System.exit(1);
        }
    }


}
