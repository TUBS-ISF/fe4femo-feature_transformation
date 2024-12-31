package de.uniulm.sp.fe4femo.helper;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

public class Helper {

    private static final Logger LOGGER = LogManager.getLogger();

    private Helper() {}

    public static LineAnalyser analyseAll(Path startPath, Function<Path, LineAnalyser> analyserGenerator)  {
        return getFilesInFolder(startPath).stream().parallel().flatMap(e -> analyseFiles(e, analyserGenerator))
                .reduce(analyserGenerator.apply(Path.of("")), LineAnalyser::accumulate);
    }

    private static List<Path> getFilesInFolder(Path folder) {
        try (Stream<Path> files = Files.walk(folder)) {
            return files.filter(Files::isRegularFile).toList();
        } catch (IOException e) {
            LOGGER.error("Error in folder analysis of folder {}", folder, e);
            throw new RuntimeException("Error in folder analysis of folder " + folder, e);
        }
    }

    private static Stream<LineAnalyser> analyseFiles(Path path, Function<Path, LineAnalyser> lineAnalyser) {
        LineAnalyser analyser = lineAnalyser.apply(path);
        try (Stream<String> lines = Files.lines(path)){
            lines.forEachOrdered(analyser::handleLine);
            return Stream.of(analyser);
        } catch (IOException e) {
            LOGGER.error("Error in file analysis of file {}", path, e);
            return Stream.empty();
        }
    }

    public static Optional<String> returnSuffix(String line, String prefix) {
        if (!line.startsWith(prefix)) return Optional.empty();
        String suffix = line.substring(prefix.length());
        if (suffix.isEmpty()) return Optional.empty();
        return Optional.of(suffix);
    }

    public static void exportCSV(Path outputPath, Collection<Integer> keys, List<NamedMap> inputMaps) throws IOException {
        try (CSVPrinter printer = new CSVPrinter(Files.newBufferedWriter(outputPath), CSVFormat.DEFAULT)) {
            List<String> header = inputMaps.stream().map(NamedMap::headerName).toList();
            printer.print("modelNo");
            printer.printRecord(header);
            int[] keyList = keys.stream().mapToInt(Integer::intValue).sorted().toArray();
            for (int modelNo : keyList) {
                printer.print(modelNo);
                for (NamedMap namedMap : inputMaps) {
                    printer.print(namedMap.map.get(modelNo));
                }
                printer.println();
            }
        }
    }

    public record NamedMap(String headerName, Map<Integer, ?> map){}


}
