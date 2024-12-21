package de.uniulm.sp.fe4femo.featureextraction;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import de.uniulm.sp.fe4femo.featureextraction.analyses.*;
import de.uniulm.sp.fe4femo.featureextraction.analyses.dymmer.AnalysisDyMMer;
import de.uniulm.sp.fe4femo.featureextraction.analysis.Analysis;
import de.uniulm.sp.fe4femo.featureextraction.analysis.Result;
import de.uniulm.sp.fe4femo.featureextraction.analysis.StatusEnum;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.collection.fm.util.FMUtils;

import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ExtractionHandler {

    private static final Logger LOGGER = LogManager.getLogger();
    private static final ObjectMapper objectMapper = JsonMapper.builder().findAndAddModules().build();

    private static List<Analysis> createAnalyses () {
        return List.of(
                new AnalysisFMBA(),
                new AnalysisSATfeatPy(),
                new AnalysisSatzilla(),
                new AnalysisFMChara(),
                new AnalysisDyMMer(),
                new AnalysisESYES()
        );
    }

    public static void main(String[] args) throws InterruptedException {
        FMUtils.installLibraries();
        Locale.setDefault(Locale.US);

        try {
            Path featureModelPath = Path.of(args[0]);
            FMInstance featureModel = FMInstanceFactory.createFMInstance(featureModelPath).orElseThrow();
            List<Result> results = analyseFM(createAnalyses(), featureModel, Integer.parseInt(args[1]));
            printResultsToConsole(results);
            System.exit(0);
        } catch (InvalidPathException e) {
            LOGGER.error("Invalid input path {}", args[0], e);
        } catch (NoSuchElementException e) {
            LOGGER.error("Error in creation of FM {}", args[0], e);
        } catch (NumberFormatException e) {
            LOGGER.error("Invalid per-step timeout \"{}\"", args[1], e);
        } catch (ArrayIndexOutOfBoundsException e) {
            LOGGER.error("Not enough CLI arguments given");
        }
        System.exit(1);
    }

    private static List<Result> analyseFM(List<Analysis> analyses, FMInstance fmInstance, int perStepTimeout) throws InterruptedException {
        List<Result> list = new ArrayList<>();
        for (Analysis analysis : analyses) {
            try {
                list.addAll(analysis.analyseFM(fmInstance, perStepTimeout));
            } catch (InterruptedException e) {
                LOGGER.error("Top level interrupt", e);
                throw new InterruptedException();
            }
        }
        return list;
    }

    private static void printResultsToConsole(List<Result> results) {
        try {
            String json = objectMapper.writeValueAsString(results);
            System.out.print("OUTPUT_JSON=");
            System.out.println(json);
        } catch (JsonProcessingException e) {
            LOGGER.error("Error while writing JSON to console", e);
        }

        List<Map.Entry<String, String>> features = results.stream().flatMap(e -> e.featureValues().entrySet().stream()).toList();

        System.out.print("OUTPUT_FEATURE_HEADER=");
        System.out.println(features.stream().map(Map.Entry::getKey).collect(Collectors.joining(",")));
        System.out.print("OUTPUT_FEATURE_LINE=");
        System.out.println(features.stream().map(Map.Entry::getValue).collect(Collectors.joining(",")));

    }

}
