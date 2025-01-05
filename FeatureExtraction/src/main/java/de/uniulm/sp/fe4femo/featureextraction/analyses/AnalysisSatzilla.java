package de.uniulm.sp.fe4femo.featureextraction.analyses;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.Analysis;
import de.uniulm.sp.fe4femo.featureextraction.analysis.AnalysisStep;
import de.uniulm.sp.fe4femo.featureextraction.analysis.ExecutableHelper;
import de.uniulm.sp.fe4femo.featureextraction.analysis.IntraStepResult;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * @see <a href="https://github.com/RSD6170/revisiting_satzilla">https://github.com/RSD6170/revisiting_satzilla</a>
 */
public class AnalysisSatzilla extends Analysis {

    public AnalysisSatzilla() {
        super("SATZilla2024",
                Executors.newSingleThreadExecutor(),
                getAnalysisSteps()
        );
    }

    private static List<AnalysisStep> getAnalysisSteps(){
        List<AnalysisStep> analysisSteps = new ArrayList<>();
        
        String[] parts = new String[]{"-base", "-sp", "-dia", "-cl", "-unit", "-ls", "-lobjois"};
        List<String[]> names = List.of(
                new String[]{"nvarsOrig", "nclausesOrig", "nvars", "nclauses", "reducedVars", "reducedClauses", "Pre-featuretime", "vars-clauses-ratio", "POSNEG-RATIO-CLAUSE-mean", "POSNEG-RATIO-CLAUSE-coeff-variation", "POSNEG-RATIO-CLAUSE-min", "POSNEG-RATIO-CLAUSE-max", "POSNEG-RATIO-CLAUSE-entropy", "VCG-CLAUSE-mean", "VCG-CLAUSE-coeff-variation", "VCG-CLAUSE-min", "VCG-CLAUSE-max", "VCG-CLAUSE-entropy", "UNARY", "BINARY+", "TRINARY+", "Basic-featuretime", "VCG-VAR-mean", "VCG-VAR-coeff-variation", "VCG-VAR-min", "VCG-VAR-max", "VCG-VAR-entropy", "POSNEG-RATIO-VAR-mean", "POSNEG-RATIO-VAR-stdev", "POSNEG-RATIO-VAR-min", "POSNEG-RATIO-VAR-max", "POSNEG-RATIO-VAR-entropy", "HORNY-VAR-mean", "HORNY-VAR-coeff-variation", "HORNY-VAR-min", "HORNY-VAR-max", "HORNY-VAR-entropy", "horn-clauses-fraction", "VG-mean", "VG-coeff-variation", "VG-min", "VG-max", "KLB-featuretime", "CG-mean", "CG-coeff-variation", "CG-min", "CG-max", "CG-entropy", "cluster-coeff-mean", "cluster-coeff-coeff-variation", "cluster-coeff-min", "cluster-coeff-max", "cluster-coeff-entropy", "CG-featuretime", "solved"},
                new String[]{"SP-bias-mean", "SP-bias-coeff-variation", "SP-bias-min", "SP-bias-max", "SP-bias-q90", "SP-bias-q10", "SP-bias-q75", "SP-bias-q25", "SP-bias-q50", "SP-unconstraint-mean", "SP-unconstraint-coeff-variation", "SP-unconstraint-min", "SP-unconstraint-max", "SP-unconstraint-q90", "SP-unconstraint-q10", "SP-unconstraint-q75", "SP-unconstraint-q25", "SP-unconstraint-q50", "sp-featuretime", "solved"},
                new String[]{"DIAMETER-mean", "DIAMETER-coeff-variation", "DIAMETER-min", "DIAMETER-max", "DIAMETER-entropy", "DIAMETER-featuretime", "solved"},
                new String[]{"cl-num-mean", "cl-num-coeff-variation", "cl-num-min", "cl-num-max", "cl-num-q90", "cl-num-q10", "cl-num-q75", "cl-num-q25", "cl-num-q50", "cl-size-mean", "cl-size-coeff-variation", "cl-size-min", "cl-size-max", "cl-size-q90", "cl-size-q10", "cl-size-q75", "cl-size-q25", "cl-size-q50", "cl-featuretime", "solved"},
                new String[]{"vars-reduced-depth-1", "vars-reduced-depth-4", "vars-reduced-depth-16", "vars-reduced-depth-64", "vars-reduced-depth-256", "unit-featuretime", "solved"},
                new String[]{"saps_BestSolution_Mean", "saps_BestSolution_CoeffVariance", "saps_FirstLocalMinStep_Mean", "saps_FirstLocalMinStep_CoeffVariance", "saps_FirstLocalMinStep_Median", "saps_FirstLocalMinStep_Q.10", "saps_FirstLocalMinStep_Q.90", "saps_BestAvgImprovement_Mean", "saps_BestAvgImprovement_CoeffVariance", "saps_FirstLocalMinRatio_Mean", "saps_FirstLocalMinRatio_CoeffVariance", "ls-saps-featuretime", "gsat_BestSolution_Mean", "gsat_BestSolution_CoeffVariance", "gsat_FirstLocalMinStep_Mean", "gsat_FirstLocalMinStep_CoeffVariance", "gsat_FirstLocalMinStep_Median", "gsat_FirstLocalMinStep_Q.10", "gsat_FirstLocalMinStep_Q.90", "gsat_BestAvgImprovement_Mean", "gsat_BestAvgImprovement_CoeffVariance", "gsat_FirstLocalMinRatio_Mean", "gsat_FirstLocalMinRatio_CoeffVariance", "ls-gsat-featuretime", "solved"},
                new String[]{"lobjois-mean-depth-over-vars", "lobjois-log-num-nodes-over-vars", "lobjois-featuretime", "solved"}
        );
        for (int i = 0; i < parts.length; i++) {
            analysisSteps.add(new SATzillaStep(parts[i], names.get(i)));
        }
        return analysisSteps;
    }

    public static class SATzillaStep implements AnalysisStep{

        private static final Logger LOGGER = LogManager.getLogger();


        private final String part;
        private final String[] names;

        protected SATzillaStep(String part, String[] names){
            this.part = part;
            this.names = names;
        }

        @Override
        public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
            ExecutableHelper.ExternalResult result = ExecutableHelper.executeExternal(getCommand(part, fmInstance.dimacsPath()), timeout, Path.of("external/revisiting_satzilla/SAT-features-competition2024"));
            return switch (result.status()){
                case SUCCESS -> {
                    LOGGER.info("SATzilla step {} executed successfully", part);
                    String[] lines = result.output().lines().toArray(String[]::new);
                    String[] featureNames = lines[lines.length-2].split(",");
                    String[] featureValues = lines[lines.length-1].split(",");
                    Map<String, String> values = new HashMap<>();
                    for (int i = 0; i < featureNames.length; i++) {
                        if (featureNames[i].contains("c Unit prop probe...")) continue;
                        values.put(featureNames[i], featureValues[i]);
                    }
                    yield new IntraStepResult(values, result.status());
                }
                case TIMEOUT, MEMOUT -> {
                    LOGGER.info("SATzilla step {} {}", part, result.status());
                    yield new IntraStepResult(Map.of(), result.status());
                }
                case ERROR -> {
                    LOGGER.warn("SATzilla step {} error with output {}", part, result.output());
                    yield new IntraStepResult(Map.of(), result.status());
                }
            };
        }

        @Override
        public String[] getAnalysesNames() {
            return names;
        }

        private static String[] getCommand(String part, Path dimacsPath){
            String[] command = new String[3];
            command[0] = Path.of("external/revisiting_satzilla/SAT-features-competition2024/features").toAbsolutePath().toString();
            command[1] = part;
            command[2] = dimacsPath.toString();
            return command;
        }

        @Override
        public String toString() {
            return "SATzillaStep " + part;
        }
    }


}
