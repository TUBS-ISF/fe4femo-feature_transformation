package de.uniulm.sp.fe4femo.helper;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

public class Helper {

    private Helper() {}

    private static final Logger LOGGER = LogManager.getLogger();

    public static LineAnalyser analyseAll(Path startPath, Class<LineAnalyser> analyserClass)  {
        return iterateFolder(startPath, path -> analyseFiles(path, createLineAnalyserInstance(path, analyserClass)))
                .reduce(createLineAnalyserInstance(null, analyserClass), LineAnalyser::accumulate);
    }

    private static LineAnalyser createLineAnalyserInstance(Path filePath, Class<LineAnalyser> analyserClass){
        try {
            return analyserClass.getDeclaredConstructor(new Class[]{Path.class}).newInstance(filePath);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            LOGGER.error("Could not instantiate analyser class {}", analyserClass.getName(), e);
            throw new RuntimeException("Could not instantiate analyser class", e);
        }
    }

    private static Optional<LineAnalyser> analyseFiles(Path path, LineAnalyser lineAnalyser) {
        try (Stream<String> lines = Files.lines(path)){
            lines.forEachOrdered(lineAnalyser::handleLine);
            return Optional.of(lineAnalyser);
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


}
