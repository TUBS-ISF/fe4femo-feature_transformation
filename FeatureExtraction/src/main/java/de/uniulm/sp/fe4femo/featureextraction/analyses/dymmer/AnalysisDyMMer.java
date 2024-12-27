package de.uniulm.sp.fe4femo.featureextraction.analyses.dymmer;


import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.Analysis;
import de.uniulm.sp.fe4femo.featureextraction.analysis.AnalysisStep;
import de.uniulm.sp.fe4femo.featureextraction.analysis.IntraStepResult;
import de.uniulm.sp.fe4femo.featureextraction.analysis.StatusEnum;
import org.collection.fm.analyses.NumberOfValidConfigurationsLog;


import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executors;

/**
 * Reimplementation of context-free DyMMer metrics
 *
 * @see <a href="https://github.com/anderson-uchoa/DyMMer/blob/dev/src/br/ufc/lps/model/normal/MeasuresWithoutContextCalculat.java">https://github.com/anderson-uchoa/DyMMer</a>
 */
public class AnalysisDyMMer extends Analysis {

    public AnalysisDyMMer() {
        super("DyMMer", Executors.newSingleThreadExecutor(), getAnalysisSteps());
    }

    private static List<AnalysisStep> getAnalysisSteps() {
        DyMMerHelper dyMMerHelper = new DyMMerHelper();
        return List.of(

                new DyMMerStep("Number of features (NF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getFeatureCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },

                new DyMMerStep(new String[]{
                        "Number of Optional Features (NO)",
                        "Number of Mandatory Features (NM)",
                        "Non-Functional Commonality (NFC)",
                        "Ratio of Switch Features",
                        "Flexibility of configuration (FoC)",
                        "Number of variable features (NVF)",
                        "Compound Complexity (ComC)"
                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout)  {
                        LOGGER.info("Starting DyMMer feature-type based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf(helper.getOptionalFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getNoMandatoryFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[2], String.valueOf((double) helper.getNoMandatoryFeatures(fmInstance) / helper.getFeatureCount(fmInstance)));
                        results.put(this.getAnalysesNames()[3], String.valueOf(((double)helper.getFeatureCount(fmInstance) - helper.getNoMandatoryFeatures(fmInstance) -1) / helper.getFeatureCount(fmInstance))); //in computation but not export?
                        results.put(this.getAnalysesNames()[4], String.valueOf((double) helper.getOptionalFeatures(fmInstance) / helper.getFeatureCount(fmInstance)));
                        results.put(this.getAnalysesNames()[5], String.valueOf(helper.getOptionalFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[6], String.valueOf(helper.getCompoundComplexity(fmInstance)));
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },
                new DyMMerStep(new String[]{
                        "Number of top features (NTop)",
                        "Single Cyclic Dependent Features (SCDF)",
                        "Multiple Cyclic Dependent Features (MCDF)",
                        "Cross-tree constraints Variables (CTCV)",
                        "Cross-tree constraints Rate"
                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        LOGGER.info("Starting DyMMer constrained-features based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf(helper.getTopFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getSingleCyclicDependentFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[2], String.valueOf(helper.getMultiCyclicDependentFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[3], String.valueOf(helper.getNoConstrainedFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[4], String.valueOf((double) helper.getNoConstrainedFeatures(fmInstance) / helper.getFeatureCount(fmInstance))); //in computation but not export?
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },

                new DyMMerStep(new String[]{
                        "Number of leaf Features (NLeaf)",
                        "Depth of tree Max (DT Max)",
                        "Depth of tree Mean (DT Mean)",
                        "Depth of tree Median (DT Median)",
                        "Feature EXtendibility (FEX)"
                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        LOGGER.info("Starting DyMMer leaf-feature based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf(helper.getLeafFeatures(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getMaxDepth(fmInstance)));
                        results.put(this.getAnalysesNames()[2], String.valueOf(helper.getMeanDepth(fmInstance)));
                        results.put(this.getAnalysesNames()[3], String.valueOf(helper.getMedianDepth(fmInstance)));
                        results.put(this.getAnalysesNames()[4], String.valueOf(helper.getLeafFeatures(fmInstance) + helper.getSingleCyclicDependentFeatures(fmInstance) + helper.getMultiCyclicDependentFeatures(fmInstance)));
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },

                new DyMMerStep("Ratio of variability (RoV)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getVariabilityRatio(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },

                new DyMMerStep(new String[]{
                        "Cyclomatic complexity (CyC)",
                        "Cross-tree constraints (CTC)"
                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        LOGGER.info("Starting DyMMer constraint-count based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf(helper.getConstraintCount(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getConstraintCount(fmInstance)));
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },

                new DyMMerStep("Grouping Features (NGF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getNoFeaturesWithChildren(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },

                new DyMMerStep(new String[]{
                        "Connectivity of the Dependency Graph Rate (Rcon)",
                        "Number of Features Referenced in Constraints Mean (Rden)",
                        "Coeficient of connectivity-density (CoC)"
                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        LOGGER.info("Starting DyMMer connectivity based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf((double) helper.getNoFeaturesConstraintRefExceptParents(fmInstance) / helper.getFeatureCount(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getMeanRefdFeaturesInConstraintsExceptParentPerFeature(fmInstance)));
                        results.put(this.getAnalysesNames()[2], String.valueOf(helper.getConnectivityDensityCoefficent(fmInstance)));
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },

                new DyMMerStep(new String[]{
                        "Single Hotspot Features (SHoF)",
                        "Multiple Hotspot Features (MHoF)",
                        "Rigid Nohotspot Features (RNoF)"
                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        LOGGER.info("Starting DyMMer feature-grouping-children based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf(helper.getGroupedXor(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getGroupedOr(fmInstance)));
                        results.put(this.getAnalysesNames()[2], String.valueOf(helper.getFeatureCount(fmInstance) - helper.getCountGrouped(fmInstance)));
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },

                new DyMMerStep("Number of valid configurations (NVC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        NumberOfValidConfigurationsLog noConfigs = new NumberOfValidConfigurationsLog();
                        String retValue = noConfigs.getResult(fmInstance.featureModel(), fmInstance.fmFormula(), timeout, Path.of("external/feature-model-batch-analysis"));
                        if (Objects.equals(retValue, "?")) throw new Exception("Error or Timeout in computation of valid configuration count!");
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], retValue),
                                StatusEnum.SUCCESS
                        );
                    }
                },

                new DyMMerStep(new String[]{
                        "Branching Factor Max (BF Max)",
                        "Branching Factor Median",
                        "Branching Factor Mean",

                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        LOGGER.info("Starting DyMMer branching based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf(helper.getBranchingFactorMax(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getBranchingFactorMedian(fmInstance)));
                        results.put(this.getAnalysesNames()[2], String.valueOf(helper.getBranchingFactorMean(fmInstance)));
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },

                new DyMMerStep(new String[]{
                        "Number Groups Or (NGOr)",
                        "Number Groups XOR (NGXOr)",
                        "Number of Variation Points (NVP)",
                        "Cognitive Complexity of a Feature Model (CogC)"
                }, dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        LOGGER.info("Starting DyMMer group-count based analysis");
                        Map<String, String> results = new HashMap<>(getAnalysesNames().length);
                        results.put(this.getAnalysesNames()[0], String.valueOf(helper.getOrGroups(fmInstance)));
                        results.put(this.getAnalysesNames()[1], String.valueOf(helper.getXorGroups(fmInstance)));
                        results.put(this.getAnalysesNames()[2], Double.toString(helper.getCountGroups(fmInstance)));
                        results.put(this.getAnalysesNames()[3], String.valueOf(helper.getOrGroups(fmInstance) + helper.getXorGroups(fmInstance)));
                        return new IntraStepResult(results, StatusEnum.SUCCESS);
                    }
                },

                new DyMMerStep("Or Rate", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf((double) helper.getChildCountOr(fmInstance) / helper.getFeatureCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },

                new DyMMerStep("Xor Rate", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf((double) helper.getChildCountXor(fmInstance) / helper.getFeatureCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                }

        );
    }


}
