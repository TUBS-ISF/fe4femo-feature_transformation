package de.uniulm.sp.fe4femo.featureextraction.analyses;

import de.ovgu.featureide.fm.core.analysis.cnf.formula.FeatureModelFormula;
import de.ovgu.featureide.fm.core.base.IFeatureModel;
import de.ovgu.featureide.fm.core.io.manager.FeatureModelManager;
import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.AnalysisStep;
import de.uniulm.sp.fe4femo.featureextraction.analysis.IntraStepResult;
import de.uniulm.sp.fe4femo.featureextraction.analysis.StatusEnum;
import org.collection.fm.util.FMUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AnalysisESYESTest {

    private static FMInstance fmInstance;
    private static final int TIMEOUT = 60;


    @BeforeAll
    static void beforeAll() {
        FMUtils.installLibraries();
        Locale.setDefault(Locale.US);

        Path modelPath = Path.of("src/test/resources/test_ESYES.uvl");
        IFeatureModel featureModel = FeatureModelManager.load(modelPath);
        if (featureModel == null) throw new IllegalArgumentException("Feature model could not be loaded");
        fmInstance = new FMInstance(modelPath, null, null, null, featureModel, new FeatureModelFormula(featureModel));
    }

    @Test
    void testActualOptMand() throws InterruptedException {
        AnalysisStep step = AnalysisESYES.getActualOptMand();
        IntraStepResult result = step.analyze(fmInstance, TIMEOUT);
        assertEquals(StatusEnum.SUCCESS, result.statusEnum());
        assertEquals(Map.of(
                "RatioActualVariable", String.valueOf(5.0/19),
                "RatioActualCommonFeatures", String.valueOf(13.0/19),
                "RatioActualDeadFeatures", String.valueOf(1.0/19)
        ), result.featureValues());
    }

    @Test
    void testOptMand() throws InterruptedException {
        AnalysisStep step = AnalysisESYES.getOptionalMandatory();
        IntraStepResult result = step.analyze(fmInstance, TIMEOUT);
        assertEquals(StatusEnum.SUCCESS, result.statusEnum());
        assertEquals(Map.of(
                "pathMandatory", String.valueOf(3),
                "RatioPathMandatory", String.valueOf(3.0/19),
                "pathOptional", String.valueOf(15),
                "RatioPathOptional", String.valueOf(15.0/19),
                "localMandatory", String.valueOf(4),
                "RatioLocalMandatory", String.valueOf(4.0/19),
                "localOptional", String.valueOf(14),
                "RatioLocalOptional", String.valueOf(14.0/19)
        ), result.featureValues());
    }

    @Test
    void testVPCC() throws InterruptedException {
        AnalysisStep step = AnalysisESYES.getVPCC();
        IntraStepResult result = step.analyze(fmInstance, TIMEOUT);
        assertEquals(StatusEnum.SUCCESS, result.statusEnum());
        assertEquals(Map.of(
                "#Variability_points_and_cyclomatic_complexity", String.valueOf(Math.sqrt(Math.pow(5, 2) + Math.pow(4, 2)))
        ), result.featureValues());
    }

    @Test
    void testInterfaceComplexity() throws InterruptedException {
        AnalysisStep step = AnalysisESYES.getInterfaceComplexity();
        IntraStepResult result = step.analyze(fmInstance, TIMEOUT);
        assertEquals(StatusEnum.SUCCESS, result.statusEnum());
        assertEquals(Map.of(
                "Interface_Complexity", String.valueOf(9)
        ), result.featureValues());
    }

    @Test
    void testAtomicSet() throws InterruptedException {
        AnalysisStep step = AnalysisESYES.getAtomicSet();
        IntraStepResult result = step.analyze(fmInstance, TIMEOUT);
        assertEquals(StatusEnum.SUCCESS, result.statusEnum());
        assertEquals(Map.of(
                "#atomicSets", String.valueOf(11)
        ), result.featureValues());
    }


}