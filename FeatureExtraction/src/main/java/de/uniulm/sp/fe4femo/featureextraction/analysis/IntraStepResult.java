package de.uniulm.sp.fe4femo.featureextraction.analysis;

import java.util.Map;

public record IntraStepResult (Map<String, String> featureValues, StatusEnum statusEnum) {
}
