package de.uniulm.sp.fe4femo.helper;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

public class Helper {

    private static final Logger LOGGER = LogManager.getLogger();

    private Helper() {}

    public static LineAnalyser analyseAll(Path startPath, Function<Path, LineAnalyser> analyserGenerator)  {
        return iterateFolder(startPath, path -> analyseFiles(path, analyserGenerator))
                .reduce(analyserGenerator.apply(Path.of("")), LineAnalyser::accumulate);
    }

    private static Optional<LineAnalyser> analyseFiles(Path path, Function<Path, LineAnalyser> lineAnalyser) {
        LineAnalyser analyser = lineAnalyser.apply(path);
        try (Stream<String> lines = Files.lines(path)){
            lines.forEachOrdered(analyser::handleLine);
            return Optional.of(analyser);
        } catch (IOException e) {
            LOGGER.error("Error in file analysis of file {}", path, e);
            return Optional.empty();
        }
    }

    private static <R> Stream<R> iterateFolder(Path path, Function<Path, Optional<R>> function) {
        try (Stream<Path> files =Files.walk(path)){
            return files.filter(Files::isRegularFile).flatMap(e -> function.apply(e).stream());
        } catch (IOException e) {
            LOGGER.error("Error in folder analysis of folder {}", path, e);
            throw new RuntimeException("Error in folder analysis of folder " + path, e);
        }
    }

    public static Optional<String> returnSuffix(String line, String prefix) {
        if (!line.startsWith(prefix)) return Optional.empty();
        String suffix = line.substring(prefix.length());
        if (suffix.isEmpty()) return Optional.empty();
        return Optional.of(suffix);
    }


}
