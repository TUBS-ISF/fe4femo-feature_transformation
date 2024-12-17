package de.uniulm.sp.fe4femo.featureextraction.analyses.dymmer;


import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.Analysis;
import de.uniulm.sp.fe4femo.featureextraction.analysis.AnalysisStep;
import de.uniulm.sp.fe4femo.featureextraction.analysis.IntraStepResult;
import de.uniulm.sp.fe4femo.featureextraction.analysis.StatusEnum;
import org.collection.fm.analyses.NumberOfValidConfigurationsLog;


import java.nio.file.Path;
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




                new DyMMerStep("Number of Optional Features (NO)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getOptionalFeatures(fmInstance, timeout))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Number of Mandatory Features (NM)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getNoMandatoryFeatures(fmInstance, timeout))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Non-Functional Commonality (NFC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        double value = (double) helper.getNoMandatoryFeatures(fmInstance, timeout) / helper.getFeatureCount(fmInstance);
                        return new IntraStepResult(Map.of(getAnalysesNames()[0], Double.toString(value)), StatusEnum.SUCCESS);
                    }
                },
                new DyMMerStep("Ratio of Switch Features", dyMMerHelper) { //in computation but not export?
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf((helper.getFeatureCount(fmInstance) - helper.getNoMandatoryFeatures(fmInstance, timeout) -1) / helper.getFeatureCount(fmInstance) )),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Flexibility of configuration (FoC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf((double) helper.getOptionalFeatures(fmInstance, timeout) / helper.getFeatureCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Number of variable features (NVF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getOptionalFeatures(fmInstance, timeout))),
                                StatusEnum.SUCCESS
                        );
                    }
                },

                new DyMMerStep("Compound Complexity (ComC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getCompoundComplexity(fmInstance, timeout))),
                                StatusEnum.SUCCESS
                        );
                    }
                },





                new DyMMerStep("Number of top features (NTop)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getTopFeatures(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },

                new DyMMerStep("Single Cyclic Dependent Features (SCDF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getSingleCyclicDependentFeatures(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Multiple Cyclic Dependent Features (MCDF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getMultiCyclicDependentFeatures(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Cross-tree constraints Variables (CTCV)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getNoConstrainedFeatures(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Cross-tree constraints Rate", dyMMerHelper) { //in computation but not export?
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf((double) helper.getNoConstrainedFeatures(fmInstance) / helper.getFeatureCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },



                new DyMMerStep("Number of leaf Features (NLeaf)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getLeafFeatures(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Depth of tree Max (DT Max)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getMaxDepth(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Depth of tree Mean (DT Mean)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getMeanDepth(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Depth of tree Median (DT Median)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getMedianDepth(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Feature EXtendibility (FEX)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getLeafFeatures(fmInstance) + helper.getSingleCyclicDependentFeatures(fmInstance) + helper.getMultiCyclicDependentFeatures(fmInstance))),
                                StatusEnum.SUCCESS
                        );
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







                new DyMMerStep("Cyclomatic complexity (CyC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getConstraintCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Cross-tree constraints (CTC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getConstraintCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
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








                new DyMMerStep("Connectivity of the Dependency Graph Rate (Rcon)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf((double) helper.getNoFeaturesConstraintRefExceptParents(fmInstance) / helper.getFeatureCount(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Number of Features Referenced in Constraints Mean (Rden)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getMeanRefdFeaturesInConstraintsExceptParentPerFeature(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Coeficient of connectivity-density (CoC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getConnectivityDensityCoefficent(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },



                new DyMMerStep("Single Hotspot Features (SHoF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getGroupedXor(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Multiple Hotspot Features (MHoF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getGroupedOr(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Rigid Nohotspot Features (RNoF)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getFeatureCount(fmInstance) - helper.getCountGrouped(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },




                new DyMMerStep("Number of valid configurations (NVC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        NumberOfValidConfigurationsLog noConfigs = new NumberOfValidConfigurationsLog();
                        String retValue = noConfigs.getResult(fmInstance.featureModel(), fmInstance.fmFormula(), timeout, Path.of("external/feature-model-batch-analysis"));
                        if (Objects.equals(retValue, "?")) throw new Exception("Error in computation of valid configuration count!");
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], retValue),
                                StatusEnum.SUCCESS
                        );
                    }
                },



                new DyMMerStep("Branching Factor Max (BF Max)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getBranchingFactorMax(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Branching Factor Median", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getBranchingFactorMedian(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Branching Factor Mean", dyMMerHelper) { //in computation but not export?
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getBranchingFactorMean(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },





                new DyMMerStep("Number Groups Or (NGOr)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getOrGroups(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Number Groups XOR (NGXOr)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], String.valueOf(helper.getXorGroups(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Number of Variation Points (NVP)", dyMMerHelper) { //in computation but not export?
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(getAnalysesNames()[0], Double.toString(helper.getCountGroups(fmInstance))),
                                StatusEnum.SUCCESS
                        );
                    }
                },
                new DyMMerStep("Cognitive Complexity of a Feature Model (CogC)", dyMMerHelper) {
                    @Override
                    protected IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception {
                        return new IntraStepResult(
                                Map.of(
                                        "Cognitive_Complexity_of_a_Feature_Model_(CogC)",
                                        String.valueOf(helper.getOrGroups(fmInstance) + helper.getXorGroups(fmInstance))
                                ),
                                StatusEnum.SUCCESS
                        );
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
