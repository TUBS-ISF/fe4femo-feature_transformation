package de.uniulm.sp.fe4femo.featureextraction.analysis;

import java.math.BigDecimal;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;

public record IntraStepResult (Map<String, BigDecimal> featureValues, StatusEnum statusEnum) {
}
