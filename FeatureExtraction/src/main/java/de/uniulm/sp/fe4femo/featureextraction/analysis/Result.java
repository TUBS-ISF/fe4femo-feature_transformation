package de.uniulm.sp.fe4femo.featureextraction.analysis;

import java.math.BigDecimal;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;

public record Result(String analysisName, Map<String, BigDecimal> featureValues, StatusEnum statusEnum, Duration duration) {

    public Result (String analysisName, IntraStepResult result, Duration runtime) {
        this(analysisName, result.featureValues(), result.statusEnum(), runtime);
    }

}