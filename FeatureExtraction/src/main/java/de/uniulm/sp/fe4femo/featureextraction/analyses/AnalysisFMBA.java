package de.uniulm.sp.fe4femo.featureextraction.analyses;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.collection.fm.analyses.*;
import org.collection.fm.util.AnalysisCacher;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * @see <a href="https://github.com/RSD6170/feature-model-batch-analysis">https://github.com/RSD6170/feature-model-batch-analysis</a>
 */
public class AnalysisFMBA extends Analysis {

    private static final Logger LOGGER = LogManager.getLogger();

    public AnalysisFMBA() {
        super("FMBA", Executors.newSingleThreadExecutor(), getAnalysisSteps());
    }

    @Override
    public List<Result> analyseFM(FMInstance instance, int perStepTimeout) throws InterruptedException {
        return super.analyseFM(instance, perStepTimeout, (perStepTimeout * 4)+4);
    }

    private static List<AnalysisStep> getAnalysisSteps() {
        List<AnalysisStep> list = new ArrayList<>();

        AnalysisCacher reusableCacher = new AnalysisCacher();
        List<IFMAnalysis> multiAnalysisStep = List.of(
                new NumberOfCoreFeatures(reusableCacher),
                new NumberOfDeadFeatures(reusableCacher),
                new RatioOfOptionalFeatures(reusableCacher),
                new NumberOfOptionalFeatures(reusableCacher)
        );
        list.add(new FMBASMultiStep(multiAnalysisStep));

        List<IFMAnalysis> singleAnalysisSteps = List.of(
                new NumberOfFeatures(),
                new NumberOfLeafFeatures(),
                new NumberOfTopFeatures(),
                new NumberOfConstraints(),
                new AverageConstraintSize(),
                new CtcDensity(),
                new FeaturesInConstraintsDensity(),
                new TreeDepth(),
                new AverageNumberOfChilden(),
                new NumberOfAlternatives(),
                new NumberOfOrs(),
                new NumberOfClauses(),
                new NumberOfLiterals(),
                new NumberOfUnitClauses(),
                new NumberOfTwoClauses(),
                new ClauseDensity(),
                new ConnectivityDensity(),
                new VoidModel(),
                new NumberOfValidConfigurationsLog(),
                new SimpleCyclomaticComplexity(),
                new IndependentCyclomaticComplexity(),

                new NumberOfTautologies(new AnalysisCacher()), // keep separate as unclear whether fully caches
                new NumberOfRedundantConstraints(new AnalysisCacher()),
                new NumberOfFalseOptionalFeatures(new AnalysisCacher())
        );
        for (IFMAnalysis analysisStep : singleAnalysisSteps) {
            list.add(new FMBASingleStep(analysisStep));
        }
        return list;
    }


    public static class FMBASMultiStep implements AnalysisStep {
        private static final Logger LOGGER = LogManager.getLogger();

        private final List<IFMAnalysis> analyses;

        public FMBASMultiStep(List<IFMAnalysis> analyses) {
            this.analyses = analyses;
        }

        @Override
        public String[] getAnalysesNames() {
            return analyses.stream().map(IFMAnalysis::getLabel).toArray(String[]::new);
        }

        @Override
        public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
            Map<String, String> results = new HashMap<>();
            for (IFMAnalysis analysisStep : analyses) {
                String result = analysisStep.getResult(fmInstance.featureModel(), fmInstance.fmFormula(), timeout, Path.of("external/feature-model-batch-analysis"));
                    if (result.equals("?")){
                        LOGGER.warn("Error or timeout in FMBA {}", analysisStep::getLabel);
                    }
                    else {
                        LOGGER.info("Analysed FMBA {} successfully", analysisStep::getLabel);
                        results.put(analysisStep.getLabel(), result);
                    }
            }
            if (results.isEmpty()) return new IntraStepResult(results, StatusEnum.ERROR);
            else return new IntraStepResult(results, StatusEnum.SUCCESS);

        }

    }

    public static class FMBASingleStep implements AnalysisStep {
        private static final Logger LOGGER = LogManager.getLogger();

        private final IFMAnalysis analysis;

        public FMBASingleStep(IFMAnalysis analysis) {
            this.analysis = analysis;
        }

        @Override
        public String[] getAnalysesNames() {
            return new String[]{analysis.getLabel()};
        }

        @Override
        public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
            String result = analysis.getResult(fmInstance.featureModel(), fmInstance.fmFormula(), timeout, Path.of("external/feature-model-batch-analysis"));
            if (result.equals("?")){
                LOGGER.warn("Error or timeout in FMBA {}", analysis::getLabel);
                return new IntraStepResult(Map.of(), StatusEnum.ERROR);
            }
            else {
                LOGGER.info("Analysed FMBA {} successfully", analysis::getLabel);
                return new IntraStepResult(Map.of(analysis.getLabel(), result), StatusEnum.SUCCESS);
            }
        }

    }
}
