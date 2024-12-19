package de.uniulm.sp.fe4femo.featureextraction.analyses.dymmer;

import de.ovgu.featureide.fm.core.analysis.cnf.formula.FeatureModelFormula;
import de.ovgu.featureide.fm.core.base.IFeatureModel;
import de.ovgu.featureide.fm.core.io.manager.FeatureModelManager;
import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import org.collection.fm.util.FMUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

class DyMMerHelperTest {

    private DyMMerHelper dymmerHelper;
    private static FMInstance fmInstance;
    private static FMInstance fmInstance2;
    private static final int TIMEOUT = 60;


    @BeforeAll
    static void beforeAll() {
        FMUtils.installLibraries();
        Locale.setDefault(Locale.US);

        Path modelPath = Path.of("src/test/resources/test.uvl");
        IFeatureModel featureModel = FeatureModelManager.load(modelPath);
        if (featureModel == null) throw new IllegalArgumentException("Feature model could not be loaded");
        fmInstance = new FMInstance(modelPath, null, null, null, featureModel, new FeatureModelFormula(featureModel));
        Path modelPath2 = Path.of("src/test/resources/test2.uvl");
        IFeatureModel featureModel2 = FeatureModelManager.load(modelPath2);
        if (featureModel2 == null) throw new IllegalArgumentException("Feature model could not be loaded");
        fmInstance2 = new FMInstance(modelPath2, null, null, null, featureModel2, new FeatureModelFormula(featureModel2));
    }

    @BeforeEach
    void setUp() {
        dymmerHelper = new DyMMerHelper();
    }

    @Test
    void testReset()  {
        long beforeFeature = dymmerHelper.getFeatureCount(fmInstance);
        long beforeOr = dymmerHelper.getOrGroups(fmInstance);
        assertNotEquals(beforeFeature, dymmerHelper.getFeatureCount(fmInstance2));
        assertEquals(beforeOr, dymmerHelper.getOrGroups(fmInstance));
    }

    @Test
    void getNoMandatoryFeatures()  {
        assertEquals(4, dymmerHelper.getNoMandatoryFeatures(fmInstance));
    }

    @Test
    void getNoDeadFeatures() throws Exception {
        assertEquals(1, dymmerHelper.getNoDeadFeatures(fmInstance, TIMEOUT));
    }

    @Test
    void getNoCoreFeatures() throws Exception {
        assertEquals(4, dymmerHelper.getNoCoreFeatures(fmInstance, TIMEOUT));
    }

    @Test
    void getFeatureCount() {
        assertEquals(17, dymmerHelper.getFeatureCount(fmInstance));
    }

    @Test
    void getTopFeatures() {
        assertEquals(5, dymmerHelper.getTopFeatures(fmInstance));
    }

    @Test
    void getLeafFeatures() throws InterruptedException {
        assertEquals(12, dymmerHelper.getLeafFeatures(fmInstance));
    }

    @Test
    void getMaxDepth() throws InterruptedException {
        assertEquals(4, dymmerHelper.getMaxDepth(fmInstance));
    }

    @Test
    void getMedianDepth() throws InterruptedException {
        assertEquals(3, dymmerHelper.getMedianDepth(fmInstance));
    }

    @Test
    void getMeanDepth() throws InterruptedException {
        assertEquals(3.1666, dymmerHelper.getMeanDepth(fmInstance), 0.0001);
    }

    @Test
    void getOrGroups() {
        assertEquals(2, dymmerHelper.getOrGroups(fmInstance));
    }

    @Test
    void getXorGroups() {
        assertEquals(2, dymmerHelper.getXorGroups(fmInstance));
    }

    @Test
    void getBranchingFactorMedian() {
        assertEquals(0, dymmerHelper.getBranchingFactorMedian(fmInstance));
    }

    @Test
    void getBranchingFactorMean() {
        assertEquals(0.9411, dymmerHelper.getBranchingFactorMean(fmInstance), 0.0001);
    }

    @Test
    void getBranchingFactorMax() {
        assertEquals(5, dymmerHelper.getBranchingFactorMax(fmInstance));
    }

    @Test
    void getRootCount() {
        assertEquals(1, dymmerHelper.getRootCount(fmInstance));
    }

    @Test
    void getSingleCyclicDependentFeatures() {
        assertEquals(6, dymmerHelper.getSingleCyclicDependentFeatures(fmInstance));
    }

    @Test
    void getMultiCyclicDependentFeatures() {
        assertEquals(1, dymmerHelper.getMultiCyclicDependentFeatures(fmInstance));
    }

    @Test
    void getChildCountXor() {
        assertEquals(5, dymmerHelper.getChildCountXor(fmInstance));
    }

    @Test
    void getChildCountOr() {
        assertEquals(6, dymmerHelper.getChildCountOr(fmInstance));
    }

    @Test
    void getGroupedXor() {
        assertEquals(5, dymmerHelper.getGroupedXor(fmInstance));
    }

    @Test
    void getGroupedOr() {
        assertEquals(6, dymmerHelper.getGroupedOr(fmInstance));
    }

    @Test
    void getOptionalFeatures() {
        assertEquals(2, dymmerHelper.getOptionalFeatures(fmInstance));
    }

    @Test
    void getConstraintCount() {
        assertEquals(4, dymmerHelper.getConstraintCount(fmInstance));
    }

    @Test
    void getCompoundComplexity() {
        assertEquals(303.6666, dymmerHelper.getCompoundComplexity(fmInstance), 0.0001);
    }

    @Test
    void getConnectivityDensityCoefficent() throws InterruptedException {
        assertEquals(6.0/17, dymmerHelper.getConnectivityDensityCoefficent(fmInstance));
    }

    @Test
    void getCountGrouped() {
        assertEquals(11, dymmerHelper.getCountGrouped(fmInstance));
    }

    @Test
    void getCountGroups() {
        assertEquals(4, dymmerHelper.getCountGroups(fmInstance));
    }

    @Test
    void getVariabilityRatio() throws InterruptedException {
        assertEquals(16.0/(17-12), dymmerHelper.getVariabilityRatio(fmInstance));
    }

    @Test
    void getNoFeaturesConstraintRefExceptParents() throws InterruptedException {
        assertEquals(9, dymmerHelper.getNoFeaturesConstraintRefExceptParents(fmInstance));
    }

    @Test
    void getMeanRefdFeaturesInConstraintsExceptParentPerFeature() throws InterruptedException {
        assertEquals(1.3333, dymmerHelper.getMeanRefdFeaturesInConstraintsExceptParentPerFeature(fmInstance), 0.0001);
    }

    @Test
    void getNoFeaturesWithChildren() {
        assertEquals(5, dymmerHelper.getNoFeaturesWithChildren(fmInstance));
    }

    @Test
    void getNoConstrainedFeatures() {
        assertEquals(9, dymmerHelper.getNoConstrainedFeatures(fmInstance));
    }
}