package de.uniulm.sp.fe4femo.helper.fe;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import de.uniulm.sp.fe4femo.helper.Helper;
import de.uniulm.sp.fe4femo.helper.LineAnalyser;
import de.uniulm.sp.fe4femo.helper.SlurmAnalyser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;

public class FeatureExtractionAnalyser extends SlurmAnalyser {

    private static final Logger LOGGER = LogManager.getLogger();
    private final ObjectMapper objectMapper = JsonMapper.builder().findAndAddModules().build();

    private final Map<Integer, Map<String, String>> featureValues = new HashMap<>();
    private final Map<Integer, Map<String, Duration>> groupTime = new HashMap<>();
    private final Map<String, List<String>> groupAssociation  = new HashMap<>();
    private final Set<String> groupNames = new HashSet<>();

    public FeatureExtractionAnalyser(Path path) {
        super(path);
    }



    @Override
    public void export(Path outputPath) throws IOException {
        exportJobOverview(outputPath);
        exportValueCSV(outputPath);
        exportMetricGroups(outputPath);
    }

    private void exportMetricGroups(Path outputPath) throws IOException {
        try (CSVPrinter printer = new CSVPrinter(Files.newBufferedWriter(outputPath.resolve("groupMapping.csv")), CSVFormat.DEFAULT)) {
            List<String> orderedGroupNames = groupNames.stream().sorted().toList();
            printer.print("featureName");
            printer.printRecord(orderedGroupNames);
            List<String> toSort = new ArrayList<>(groupAssociation.keySet());
            toSort.sort(null);
            for (String featureName : toSort) {
                printer.print(featureName);
                for (String groupName : orderedGroupNames) {
                    printer.print(groupAssociation.get(featureName).contains(groupName));
                }
                printer.println();
            }
        }

        try (CSVPrinter printer = new CSVPrinter(Files.newBufferedWriter(outputPath.resolve("groupTimes.csv")), CSVFormat.DEFAULT)) {
            printer.printRecord("modelNo", "groupName", "groupTimeS");
            for (Map.Entry<Integer, Map<String, Duration>> entry : groupTime.entrySet()) {
                Integer modelNo = entry.getKey();
                for (Map.Entry<String, Duration> subEntry : entry.getValue().entrySet()) {
                    String groupName = subEntry.getKey();
                    Duration duration = subEntry.getValue();
                    printer.printRecord(modelNo, groupName, BigDecimal.valueOf(duration.toNanos(), 9).stripTrailingZeros());
                }
            }
        }
    }

    private void exportValueCSV(Path outputPath) throws IOException {
        Map<String, Map<Integer, String>> newMap = HashMap.newHashMap(featureValues.size());
        featureValues.forEach((modelNo, map) -> map.forEach((featureName, featureValue) -> {
            newMap.putIfAbsent(featureName, new HashMap<>());
            newMap.get(featureName).put(modelNo, featureValue);
        }));
        List<Helper.NamedMap> outputMaps = newMap.entrySet().stream().map(e -> new Helper.NamedMap(e.getKey(), e.getValue())).sorted(Comparator.comparing(Helper.NamedMap::headerName)).toList();
        Helper.exportCSV(outputPath.resolve("values.csv"), modelPath.keySet(), outputMaps);
    }

    private void exportJobOverview(Path outputPath) throws IOException {
        List<Helper.NamedMap> outputMaps = List.of(
                new Helper.NamedMap("modelPath", modelPath),
                new Helper.NamedMap("jobTimeS", jobTime),
                new Helper.NamedMap("cpuUtilTimeS", cpuUtilizationTime),
                new Helper.NamedMap("memUseMB", jobMemMb)
        );
        Helper.exportCSV(outputPath.resolve("job_overview.csv"), modelPath.keySet(), outputMaps);
    }

    @Override
    public LineAnalyser accumulate(LineAnalyser lineAnalyser) {
        if (!this.getClass().isAssignableFrom(lineAnalyser.getClass())) {
            LOGGER.error("Non-fitting line analyser in accumulation! Current {} is not compatible with given {}", this.getClass().getSimpleName(), lineAnalyser.getClass().getSimpleName());
            throw new RuntimeException("Non-fitting line analyser in accumulation!");
        }
        FeatureExtractionAnalyser analyser = (FeatureExtractionAnalyser) super.accumulate(lineAnalyser);
        analyser.featureValues.putAll(this.featureValues);
        analyser.groupAssociation.putAll(this.groupAssociation);
        analyser.groupTime.putAll(this.groupTime);
        return analyser;
    }

    @Override
    public void handleLine(String line) {
        super.handleLine(line);
        Helper.returnSuffix(line, "OUTPUT_JSON=").ifPresent(this::handleJson);

    }

    private void handleJson(String json) {
        try {
            groupTime.putIfAbsent(modelNumber, new HashMap<>());
            Map<String, Duration> groupTimeInstance = groupTime.get(modelNumber);

            List<Result> results = objectMapper.readValue(json, objectMapper.getTypeFactory().constructCollectionType(List.class, Result.class));
            featureValues.put(modelNumber, results.stream().flatMap(e -> e.featureValues().entrySet().stream()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));

            Map<String, Integer> groupCounts = new HashMap<>();
            for (Result result : results) {
                groupCounts.putIfAbsent(result.analysisName(), 0);
                String groupName = result.analysisName() + "_" + groupCounts.get(result.analysisName());
                groupNames.add(groupName);
                result.featureValues().keySet().forEach(featureName -> {
                    groupAssociation.putIfAbsent(featureName, new ArrayList<>());
                    groupAssociation.get(featureName).add(groupName);
                });

                groupTimeInstance.put(groupName, result.duration());

                groupCounts.put(result.analysisName(), groupCounts.get(result.analysisName()) + 1);
            }
        } catch (JsonProcessingException e) {
            LOGGER.error("Parser error in '{}' with json string '{}'", path, json, e);
        }
    }
}
