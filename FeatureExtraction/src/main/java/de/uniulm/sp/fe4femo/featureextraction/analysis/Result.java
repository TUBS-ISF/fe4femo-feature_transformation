package de.uniulm.sp.fe4femo.featureextraction.analysis;

import java.time.Duration;
import java.util.Map;

public record Result(String analysisName, Map<String, String> featureValues, StatusEnum statusEnum, Duration duration) {

    public Result (String analysisName, IntraStepResult result, Duration runtime) {
        this(analysisName, result.featureValues(), result.statusEnum(), runtime);
    }

}