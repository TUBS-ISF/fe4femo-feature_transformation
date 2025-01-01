package de.uniulm.sp.fe4femo.helper.fe;


import java.time.Duration;
import java.util.Map;

public record Result(String analysisName, Map<String, String> featureValues, String statusEnum, Duration duration) {

}
