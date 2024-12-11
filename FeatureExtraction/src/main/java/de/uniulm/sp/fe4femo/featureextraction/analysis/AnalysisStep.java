package de.uniulm.sp.fe4femo.featureextraction.analysis;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;


public interface AnalysisStep {

    IntraStepResult analyze(FMInstance fmInstance, int timeout);

    List<String> getAnalysesNames();


}
