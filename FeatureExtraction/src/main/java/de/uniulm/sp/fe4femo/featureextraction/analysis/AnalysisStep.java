package de.uniulm.sp.fe4femo.featureextraction.analysis;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;

public interface AnalysisStep {

    IntraStepResult analyze(FMInstance fmInstance, int timeout, Analysis parentAnalysis) throws InterruptedException;

    String[] getAnalysesNames();


}
