package de.uniulm.sp.fe4femo.featureextraction.analysis;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;

import java.util.List;


public interface AnalysisStep {

    IntraStepResult analyze(FMInstance fmInstance, int timeout, Analysis parentAnalysis) throws InterruptedException;

    List<String> getAnalysesNames();


}
