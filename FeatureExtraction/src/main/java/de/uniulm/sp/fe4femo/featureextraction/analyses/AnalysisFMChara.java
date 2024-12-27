package de.uniulm.sp.fe4femo.featureextraction.analyses;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * @see <a href="https://github.com/RSD6170/fm_characterization">https://github.com/RSD6170/fm_characterization</a>
 */
public class AnalysisFMChara extends Analysis {

    public AnalysisFMChara() {
        super("FM_Characterization", Executors.newSingleThreadExecutor(), getAnalysisSteps());
    }

    @Override
    public List<Result> analyseFM(FMInstance instance, int perStepTimeout) throws InterruptedException {
        return super.analyseFM(instance, perStepTimeout, (perStepTimeout * 2) + 2);
    }

    private static List<AnalysisStep> getAnalysisSteps() {

        String[] metrics = new String[]{
                "METRICS/Features/value",
                "METRICS/Features/Abstract_features/value",
                "METRICS/Features/Abstract_features/ratio",
                "METRICS/Features/Abstract_features/Abstract_leaf_features/value",
                "METRICS/Features/Abstract_features/Abstract_leaf_features/ratio",
                "METRICS/Features/Abstract_features/Abstract_compound_features/value",
                "METRICS/Features/Abstract_features/Abstract_compound_features/ratio",
                "METRICS/Features/Concrete_features/value",
                "METRICS/Features/Concrete_features/ratio",
                "METRICS/Features/Concrete_features/Concrete_leaf_features/value",
                "METRICS/Features/Concrete_features/Concrete_leaf_features/ratio",
                "METRICS/Features/Concrete_features/Concrete_compound_features/value",
                "METRICS/Features/Concrete_features/Concrete_compound_features/ratio",
                "METRICS/Features/Compound_features/value",
                "METRICS/Features/Compound_features/ratio",
                "METRICS/Features/Leaf_features/value",
                "METRICS/Features/Leaf_features/ratio",
                "METRICS/Features/Root_feature/value",
                "METRICS/Features/Root_feature/ratio",
                "METRICS/Features/Root_feature/Top_features/value",
                "METRICS/Features/Root_feature/Top_features/ratio",
                "METRICS/Features/Solitary_features/value",
                "METRICS/Features/Solitary_features/ratio",
                "METRICS/Features/Grouped_features/value",
                "METRICS/Features/Grouped_features/ratio",
                "METRICS/Tree_relationships/value",
                "METRICS/Tree_relationships/Mandatory_features/value",
                "METRICS/Tree_relationships/Mandatory_features/ratio",
                "METRICS/Tree_relationships/Optional_features/value",
                "METRICS/Tree_relationships/Optional_features/ratio",
                "METRICS/Tree_relationships/Feature_groups/value",
                "METRICS/Tree_relationships/Feature_groups/ratio",
                "METRICS/Tree_relationships/Feature_groups/Alternative_groups/value",
                "METRICS/Tree_relationships/Feature_groups/Alternative_groups/ratio",
                "METRICS/Tree_relationships/Feature_groups/Or_groups/value",
                "METRICS/Tree_relationships/Feature_groups/Or_groups/ratio",
                "METRICS/Tree_relationships/Feature_groups/Mutex_groups/value",
                "METRICS/Tree_relationships/Feature_groups/Mutex_groups/ratio",
                "METRICS/Tree_relationships/Feature_groups/Cardinality_groups/value",
                "METRICS/Tree_relationships/Feature_groups/Cardinality_groups/ratio",
                "METRICS/Depth_of_tree/value",
                "METRICS/Depth_of_tree/Mean_depth_of_tree/value",
                "METRICS/Branching_factor/value",
                "METRICS/Branching_factor/Min_children_per_feature/value",
                "METRICS/Branching_factor/Max_children_per_feature/value",
                "METRICS/Branching_factor/Avg_children_per_feature/value",
                "METRICS/Cross-tree_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/ratio",
                "METRICS/Cross-tree_constraints/Logical_constraints/Single_feature_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/Single_feature_constraints/ratio",
                "METRICS/Cross-tree_constraints/Logical_constraints/Simple_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/Simple_constraints/ratio",
                "METRICS/Cross-tree_constraints/Logical_constraints/Simple_constraints/Requires_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/Simple_constraints/Requires_constraints/ratio",
                "METRICS/Cross-tree_constraints/Logical_constraints/Simple_constraints/Excludes_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/Simple_constraints/Excludes_constraints/ratio",
                "METRICS/Cross-tree_constraints/Logical_constraints/Complex_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/Complex_constraints/ratio",
                "METRICS/Cross-tree_constraints/Logical_constraints/Complex_constraints/Pseudo-complex_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/Complex_constraints/Pseudo-complex_constraints/ratio",
                "METRICS/Cross-tree_constraints/Logical_constraints/Complex_constraints/Strict-complex_constraints/value",
                "METRICS/Cross-tree_constraints/Logical_constraints/Complex_constraints/Strict-complex_constraints/ratio",
                "METRICS/Cross-tree_constraints/Features_in_constraints/value",
                "METRICS/Cross-tree_constraints/Features_in_constraints/ratio",
                "METRICS/Cross-tree_constraints/Features_in_constraints/Min_features_per_constraint/value",
                "METRICS/Cross-tree_constraints/Features_in_constraints/Max_features_per_constraint/value",
                "METRICS/Cross-tree_constraints/Features_in_constraints/Avg_features_per_constraint/value",
                "METRICS/Cross-tree_constraints/Avg_constraints_per_feature/value",
                "METRICS/Cross-tree_constraints/Min_constraints_per_feature/value",
                "METRICS/Cross-tree_constraints/Max_constraints_per_feature/value",
                "METRICS/Attributes/value",
                "METRICS/Attributes/Features_with_attributes/value",
                "METRICS/Attributes/Features_with_attributes/ratio",
                "METRICS/Attributes/Min_attributes_per_feature/value",
                "METRICS/Attributes/Max_attributes_per_feature/value",
                "METRICS/Attributes/Avg_attributes_per_feature/value",
                "METRICS/Attributes/Avg_attributes_per_feature_w._attributes/value"
        };

        String[] analyses = new String[]{
                "ANALYSIS/Satisfiable_(valid)/value",
                "ANALYSIS/Core_features/value",
                "ANALYSIS/Core_features/ratio",
                "ANALYSIS/Core_features/False-optional_features/value",
                "ANALYSIS/Core_features/False-optional_features/ratio",
                "ANALYSIS/Dead_features/value",
                "ANALYSIS/Dead_features/ratio",
                "ANALYSIS/Variant_features/value",
                "ANALYSIS/Variant_features/ratio",
                "ANALYSIS/Variant_features/Unique_features/value",
                "ANALYSIS/Variant_features/Unique_features/ratio",
                "ANALYSIS/Variant_features/Pure_optional_features/value",
                "ANALYSIS/Variant_features/Pure_optional_features/ratio",
                "ANALYSIS/Configurations/value",
                "ANALYSIS/Total_variability/value",
                "ANALYSIS/Partial_variability/value",
                "ANALYSIS/Homogeneity/value",
                "ANALYSIS/Configuration_distribution/value",
                "ANALYSIS/Configuration_distribution/Mean/value",
                "ANALYSIS/Configuration_distribution/Standard_deviation/value",
                "ANALYSIS/Configuration_distribution/Median/value",
                "ANALYSIS/Configuration_distribution/Median_absolute_deviation/value",
                "ANALYSIS/Configuration_distribution/Mode/value",
                "ANALYSIS/Configuration_distribution/Min/value",
                "ANALYSIS/Configuration_distribution/Max/value",
                "ANALYSIS/Configuration_distribution/Range/value",
                "ANALYSIS/isApprox/value"
        };

        return List.of(new FMCharaMetricStep("FM_Characterization_Metrics", metrics), new FMCharaAnalysesStep("FM_Characterization_Analyses", analyses));


    }

    public static class FMCharaMetricStep implements AnalysisStep {
        private static final Logger LOGGER = LogManager.getLogger();

        protected final String part;
        private final String[] names;
        protected final Path subpath = Path.of("external/fm_characterization");

        public FMCharaMetricStep(String part, String[] names) {
            this.part = part;
            this.names = names;
        }

        @Override
        public String[] getAnalysesNames() {
            return names;
        }

        @Override
        public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
            ExecutableHelper.ExternalResult result = ExecutableHelper.executeExternal(getCommand(part, fmInstance.uvlPath(), fmInstance.xmlPath(), "metrics"), timeout, subpath);
            return switch (result.status()){
                case SUCCESS -> {
                    LOGGER.info("FM_Characterization metric step executed successfully");
                    yield new IntraStepResult(
                            parseOutput(result),
                            StatusEnum.SUCCESS);
                }
                case TIMEOUT, MEMOUT -> {
                    LOGGER.info("FM_Characterization metric step {}", result.status());
                    yield new IntraStepResult(Map.of(), result.status());
                }
                case ERROR -> {
                    LOGGER.warn("FM_Characterization metric step error with output {}", result.output());
                    yield new IntraStepResult(Map.of(), result.status());
                }
            };
        }



        protected static Map<String, String> parseOutput(ExecutableHelper.ExternalResult result) {
            try {
                Map<String, String> output = result.output().lines()
                        .dropWhile(e -> !Objects.equals("###---###", e)).skip(1)
                        .map(e -> e.split(" ", 2))
                        .collect(Collectors.toMap(e -> e[0], e -> e[1]));
                if (output.isEmpty()) {
                    LOGGER.error("Missing output in FMChara output parsing with input: {}", result.output());
                    throw new IllegalStateException("Missing Output in FMChara");
                }
                return output;
            } catch (Exception e) {
                LOGGER.error("Unexpected behaviour in FMChara output parsing with input: {}", result.output(), e);
                throw e;
            }
        }

        protected static String[] getCommand(String part, Path uvlPath, Path xmlPath, String mode){
            return new String[]{
                    Path.of("/venv_fm_chara/bin/python").toAbsolutePath().toString(),
                    Path.of("external/fm_characterization/main_characterization.py").toAbsolutePath().toString(),
                    "--mode",
                    mode,
                    uvlPath.toString(),
                    xmlPath.toString()
            };
        }

        @Override
        public String toString() {
            return "FMCharaMetricStep";
        }
    }

    public static class FMCharaAnalysesStep extends FMCharaMetricStep {

        private static final Logger LOGGER = LogManager.getLogger();

        public FMCharaAnalysesStep(String part, String[] names) {
            super(part, names);
        }

        @Override
        public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
            ExecutableHelper.ExternalResult result = ExecutableHelper.executeExternal(getCommand(part, fmInstance.uvlPath(), fmInstance.xmlPath(), "analysis_full"), timeout, subpath);
            return switch (result.status()){
                case SUCCESS -> {
                    LOGGER.info("Expensive FM_Characterization analyses step executed successfully");
                    yield new IntraStepResult(
                            parseOutput(result),
                            StatusEnum.SUCCESS);
                }
                case TIMEOUT, MEMOUT -> {
                    LOGGER.info("Expensive FM_Characterization step {}", result.status());
                    LOGGER.info("Trying easy FM_Characterization analyses step");
                    yield analyzeEasy(fmInstance, timeout);
                }
                case ERROR -> {
                    LOGGER.warn("Expensive FM_Characterization step error with output {}", result.output());
                    LOGGER.info("Trying easy FM_Characterization analyses step");
                    yield analyzeEasy(fmInstance, timeout);
                }
            };
        }

        private IntraStepResult analyzeEasy(FMInstance fmInstance, int timeout) throws InterruptedException{
            ExecutableHelper.ExternalResult result = ExecutableHelper.executeExternal(getCommand(part, fmInstance.uvlPath(), fmInstance.xmlPath(), "analysis_light"), timeout, subpath);
            return switch (result.status()){
                case SUCCESS -> {
                    LOGGER.info("Easy FM_Characterization analyses step executed successfully");
                    yield new IntraStepResult(
                            parseOutput(result),
                            StatusEnum.SUCCESS);
                }
                case TIMEOUT, MEMOUT -> {
                    LOGGER.info("Easy FM_Characterization analyses step {}", result.status());
                    yield new IntraStepResult(Map.of(), result.status());
                }
                case ERROR -> {
                    LOGGER.warn("Easy FM_Characterization analyses step error with output {}", result.output());
                    yield new IntraStepResult(Map.of(), result.status());
                }
            };
        }

        @Override
        public String toString() {
            return "FMCharaAnalysesStep";
        }
    }
}
