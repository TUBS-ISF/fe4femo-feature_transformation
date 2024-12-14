package de.uniulm.sp.fe4femo.featureextraction.analyses;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.collection.fm.analyses.*;
import org.collection.fm.handler.SatzillaHandler;
import org.collection.fm.util.AnalysisCacher;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Executors;

public class AnalysisFMBA extends Analysis {

    private static final Logger LOGGER = LogManager.getLogger();

    public AnalysisFMBA() {
        super("FMBA", Executors.newSingleThreadExecutor(), getAnalysisSteps());
    }

    private static List<AnalysisStep> getAnalysisSteps() {
        AnalysisCacher analysisCacher = new AnalysisCacher();
        List<IFMAnalysis> analysisSteps = List.of(
                new NumberOfFeatures(),
                new NumberOfLeafFeatures(),
                new NumberOfTopFeatures(),
                new NumberOfConstraints(),
                new AverageConstraintSize(),
                new CtcDensity(),
                new FeaturesInConstraintsDensity(),
                new NumberOfTautologies(analysisCacher),
                new NumberOfRedundantConstraints(analysisCacher),
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
                new NumberOfCoreFeatures(analysisCacher),
                new NumberOfDeadFeatures(analysisCacher),
                new RatioOfOptionalFeatures(analysisCacher),
                new NumberOfFalseOptionalFeatures(analysisCacher),
                new NumberOfOptionalFeatures(analysisCacher),
                new NumberOfValidConfigurationsLog(),
                new SimpleCyclomaticComplexity(),
                new IndependentCyclomaticComplexity()
        );
        List<AnalysisStep> list = new ArrayList<>();
        for (IFMAnalysis analysisStep : analysisSteps) {
            list.add(new FMBAStep(analysisStep));
        }
        return list;
    }


    public static class FMBAStep implements AnalysisStep {
        private static final Logger LOGGER = LogManager.getLogger();

        private final IFMAnalysis analysis;

        public FMBAStep(IFMAnalysis analysis) {
            this.analysis = analysis;
        }

        @Override
        public String[] getAnalysesNames() {
            return new String[]{"FMBA/" + analysis.getLabel()};
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
