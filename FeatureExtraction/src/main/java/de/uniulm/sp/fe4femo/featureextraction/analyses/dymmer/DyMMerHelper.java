package de.uniulm.sp.fe4femo.featureextraction.analyses.dymmer;

import de.ovgu.featureide.fm.core.FeatureModelAnalyzer;
import de.ovgu.featureide.fm.core.analysis.cnf.LiteralSet;
import de.ovgu.featureide.fm.core.analysis.cnf.analysis.CoreDeadAnalysis;
import de.ovgu.featureide.fm.core.analysis.cnf.formula.FeatureModelFormula;
import de.ovgu.featureide.fm.core.base.IConstraint;
import de.ovgu.featureide.fm.core.base.IFeature;
import de.ovgu.featureide.fm.core.base.IFeatureModel;
import de.ovgu.featureide.fm.core.base.IFeatureStructure;
import de.ovgu.featureide.fm.core.job.monitor.NullMonitor;
import de.uniulm.sp.fe4femo.featureextraction.FMInstance;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DyMMerHelper {

    private FMInstance fmInstance;

    private FeatureModelAnalyzer featureModelAnalyzer;
    private long mandatoryFeatures;
    private long deadFeatures;
    private long featureCount;
    private long topFeatures;
    private long leafFeatures;
    private long maxDepth;
    private double medianDepth;
    private double meanDepth;
    private long constraintCount;
    private double variablityRatio;
    private long noFeaturesConstraintRefExceptParent;
    private double meanRefdFeaturesInConstraintsExceptParentPerFeature;
    private long noFeaturesWithChildren;

    private long orGroups;
    private long xorGroups;
    private ArrayList<Integer> childCounts = new ArrayList<>();
    private double branchingFactorMedian;
    private double branchingFactorMean;
    private int branchingFactorMax;
    private Set<IFeature> constrainedFeatures = null;
    private int rootCount;
    private long weirdMandatory;
    private long weirdOptional;
    private long singleCyclicDependentFeatures;
    private long multiCyclicDependentFeatures;
    private long childCountXor;
    private long childCountOr;
    private long groupedXor;
    private long groupedOr;


    protected DyMMerHelper() {
        invalidateAll();
    }

    private void invalidateAll() {
        mandatoryFeatures = -1;
        deadFeatures = -1;
        featureModelAnalyzer = null;
        featureCount = -1;
        topFeatures = -1;
        leafFeatures = -1;
        maxDepth = -1;
        medianDepth = -1;
        meanDepth = -1;
        constraintCount = -1;
        variablityRatio = -1;
        noFeaturesConstraintRefExceptParent = -1;
        meanRefdFeaturesInConstraintsExceptParentPerFeature = -1;
        noFeaturesWithChildren = -1;

        orGroups = -1;
        xorGroups = -1;
        childCounts.clear();
        branchingFactorMedian = -1;
        branchingFactorMean = -1;
        branchingFactorMax = -1;
        constrainedFeatures = null;
        rootCount = -1;
        weirdMandatory = -1;
        weirdOptional = -1;
        singleCyclicDependentFeatures = -1;
        multiCyclicDependentFeatures = -1;
        childCountXor = -1;
        childCountOr = -1;
        groupedXor = -1;
        groupedOr = -1;

    }

    private FeatureModelAnalyzer getFeatureModelAnalyzer(FeatureModelFormula formula) {
        if (Objects.isNull(featureModelAnalyzer) || !Objects.equals(fmInstance.fmFormula(), formula)) {
            featureModelAnalyzer = new FeatureModelAnalyzer(formula);
        }
        return featureModelAnalyzer;
    }

    private void checkInstance(FMInstance fmInstance) {
        if (! fmInstance.equals(this.fmInstance)) {
            invalidateAll();
            this.fmInstance = fmInstance;
        }
    }

    private void manDeadAnalysis(FMInstance fmInstance, int timeout) throws Exception {
        FeatureModelAnalyzer analyzer = getFeatureModelAnalyzer(fmInstance.fmFormula());
        CoreDeadAnalysis coreDeadAnalysis = new CoreDeadAnalysis(fmInstance.fmFormula().getCNF());
        coreDeadAnalysis.setTimeout(1000* timeout);
        LiteralSet result = coreDeadAnalysis.execute(new NullMonitor<>());
        mandatoryFeatures = fmInstance.fmFormula().getCNF().getVariables().convertToString(result, true, false, false).size();
        deadFeatures = fmInstance.fmFormula().getCNF().getVariables().convertToString(result, false, true, false).size();
    }

    private void generateLeafChildrenStats(IFeatureModel featureModel) throws InterruptedException{
        List<IFeature> leafFeatureList = featureModel.getFeatures().stream().filter(e -> ! e.getStructure().hasChildren()).toList();
        leafFeatures = leafFeatureList.size();
        long sum = 0;
        long count = 0;
        for (IFeature iFeature : leafFeatureList) {
            if (Thread.currentThread().isInterrupted()) throw new InterruptedException();
            long featureDepth = getFeatureDepth(iFeature.getStructure());
            sum += featureDepth;
            count++;
        }
        meanDepth = count > 0 ? (double) sum / count : 0.0;
    }

    private static long getFeatureDepth(IFeatureStructure feature) {
        if (Objects.isNull(feature.getParent())) return 1;
        return getFeatureDepth(feature.getParent()) + 1;
    }

    private static long getTreeDepthRecursive(IFeatureStructure structure) {
        List<Long> depthOfChildren = new ArrayList<>();
        if (structure.getChildrenCount() == 0) {
            return 1;
        }
        for(IFeatureStructure child : structure.getChildren()) {
            if (Thread.currentThread().isInterrupted()) break;
            depthOfChildren.add(getTreeDepthRecursive(child));
        }
        return Collections.max(depthOfChildren) + 1;
    }

    private void computeConstrainedFeatures(IFeatureModel featureModel) {
        constrainedFeatures = featureModel.getConstraints().stream().flatMap(e -> e.getContainedFeatures().stream()).collect(Collectors.toSet());
    }

    // https://github.com/anderson-uchoa/DyMMer/blob/cef913244aa2daeb3d41039b2595c7ff19f098a9/src/br/ufc/lps/splar/core/fm/FeatureModelStatistics.java#L59
    private void createStatistics(IFeatureModel featureModel) {
        childCounts.clear();
        rootCount = 0;
        weirdMandatory = 0;
        weirdOptional = 0;
        xorGroups = 0;
        orGroups = 0;
        singleCyclicDependentFeatures = 0;
        multiCyclicDependentFeatures = 0;
        childCountOr = 0;
        childCountXor = 0;
        groupedXor = 0;
        groupedOr = 0;
        if(Objects.isNull(constrainedFeatures)) computeConstrainedFeatures(featureModel);

        for (IFeature feature : featureModel.getFeatures()) {
            if (Thread.currentThread().isInterrupted()) break;
            IFeatureStructure featureNode = feature.getStructure();
            IFeatureStructure parent = featureNode.getParent();

            childCounts.add(featureNode.getChildrenCount());

            if (featureNode.isRoot()){
                rootCount++;
            } else if (featureNode.isOr() || featureNode.isAlternative() || parent.isOr() || parent.isAlternative()) {
                if (featureNode.hasChildren()) {
                    if (featureNode.isAlternative()) {
                        childCountXor += featureNode.getChildrenCount();
                        xorGroups++;
                        if (constrainedFeatures.contains(feature)) singleCyclicDependentFeatures++;
                    } else if (featureNode.isOr()) {
                        childCountOr += featureNode.getChildrenCount();
                        orGroups++;
                        if (constrainedFeatures.contains(feature)) multiCyclicDependentFeatures++;
                    }
                } else {
                    if (parent.isAlternative()) {
                        groupedXor++;
                        if (constrainedFeatures.contains(feature)) singleCyclicDependentFeatures++;
                    } else if (parent.isOr()) {
                        groupedOr++;
                        if (constrainedFeatures.contains(feature)) multiCyclicDependentFeatures++;
                    }
                }
            } else {
                if (featureNode.isMandatory()) weirdMandatory++;
                else weirdOptional++;
            }

        }


        IntSummaryStatistics childrenStatistics = childCounts.stream().mapToInt(i -> i ).summaryStatistics();
        branchingFactorMax = childrenStatistics.getMax();
        branchingFactorMean = childrenStatistics.getAverage();
        branchingFactorMedian = getMedian(childCounts);
    }

    private static double getMedian(List<Integer> input) {
        List<Integer> sortedList = input.stream().sorted().toList();
        if (sortedList.size() % 2 == 0) return sortedList.get(sortedList.size() / 2);
        else return (sortedList.get(sortedList.size() / 2 ) + sortedList.get(sortedList.size() / 2 +1))/2.0;
    }

    private void computeConnectivity(FMInstance fmInstance) throws InterruptedException {
        Map<IFeature, Set<IFeature>> connectedFeatures = new HashMap<>();
        for (IConstraint constraint : fmInstance.featureModel().getConstraints()) {
            for (IFeature feature : constraint.getContainedFeatures()) {
                if (Thread.currentThread().isInterrupted()) throw new InterruptedException();
                List<IFeature> constraintCopy = constraint.getContainedFeatures().stream()
                        .filter(e -> Objects.equals(e, feature))
                        .filter(e -> Objects.equals(feature.getStructure().getParent(), e.getStructure()))
                        .toList();
                connectedFeatures.putIfAbsent(feature, new HashSet<>());
                connectedFeatures.get(feature).addAll(constraintCopy);
            }
        }
        noFeaturesConstraintRefExceptParent = connectedFeatures.entrySet().stream().filter(e -> ! e.getValue().isEmpty()).count();
        meanRefdFeaturesInConstraintsExceptParentPerFeature = connectedFeatures.values().stream().filter(iFeatures -> !iFeatures.isEmpty()).mapToInt(Set::size).average().orElseGet(() -> 0.0);
    }

    // ------------------------------

    protected long getNoMandatoryFeatures(FMInstance fmInstance, int timeout) throws Exception {
        checkInstance(fmInstance);
        if (mandatoryFeatures == -1){
            manDeadAnalysis(fmInstance, timeout);
        }
        return mandatoryFeatures;
    }

    protected long getNoDeadFeatures(FMInstance fmInstance, int timeout) throws Exception {
        checkInstance(fmInstance);
        if (deadFeatures == -1){
            manDeadAnalysis(fmInstance, timeout);
        }
        return deadFeatures;
    }

    protected long getFeatureCount(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (featureCount == -1){
            featureCount = fmInstance.featureModel().getNumberOfFeatures();
        }
        return featureCount;
    }

    protected long getTopFeatures(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (topFeatures == -1){
            topFeatures = fmInstance.featureModel().getStructure().getRoot().getChildrenCount();
        }
        return topFeatures;
    }

    protected long getLeafFeatures(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (leafFeatures == -1){
            generateLeafChildrenStats(fmInstance.featureModel());
        }
        return leafFeatures;
    }

    protected long getMaxDepth(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (maxDepth == -1){
            maxDepth = getTreeDepthRecursive(fmInstance.featureModel().getStructure().getRoot());
        }
        return maxDepth;
    }

    protected double getMedianDepth(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (medianDepth < 0){
            List<IFeatureStructure> topLevel = fmInstance.featureModel().getStructure().getRoot().getChildren();
            if (topLevel.size() % 2 == 1){
                medianDepth = getTreeDepthRecursive(topLevel.get((topLevel.size() / 2) + 1));
            } else {
                medianDepth = (getTreeDepthRecursive(topLevel.get(topLevel.size() / 2))+ getTreeDepthRecursive(topLevel.get((topLevel.size() / 2) + 1))) / 2.0;
            }
        }
        return medianDepth;
    }

    protected double getMeanDepth(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (meanDepth < 0) {
            generateLeafChildrenStats(fmInstance.featureModel());
        }
        return meanDepth;
    }

    protected long getOrGroups(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (orGroups == -1){
            createStatistics(fmInstance.featureModel());
        }
        return orGroups;
    }

    protected long getXorGroups(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (xorGroups == -1){
            createStatistics(fmInstance.featureModel());
        }
        return xorGroups;
    }

    protected double getBranchingFactorMedian(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (branchingFactorMedian < 0){
            createStatistics(fmInstance.featureModel());
        }
        return branchingFactorMedian;
    }
    protected double getBranchingFactorMean(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (branchingFactorMean < 0){
            createStatistics(fmInstance.featureModel());
        }
        return branchingFactorMean;
    }
    protected int getBranchingFactorMax(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (branchingFactorMax < 0){
            createStatistics(fmInstance.featureModel());
        }
        return branchingFactorMax;
    }
    protected int getRootCount(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (rootCount < 0){
            createStatistics(fmInstance.featureModel());
        }
        return rootCount;
    }
    protected long getWeirdMandatory(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (weirdMandatory < 0){
            createStatistics(fmInstance.featureModel());
        }
        return weirdMandatory;
    }
    protected long getWeirdOptional(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (weirdOptional < 0){
            createStatistics(fmInstance.featureModel());
        }
        return weirdOptional;
    }
    protected long getSingleCyclicDependentFeatures(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (singleCyclicDependentFeatures < 0){
            createStatistics(fmInstance.featureModel());
        }
        return singleCyclicDependentFeatures;
    }
    protected long getMultiCyclicDependentFeatures(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (multiCyclicDependentFeatures < 0){
            createStatistics(fmInstance.featureModel());
        }
        return multiCyclicDependentFeatures;
    }
    protected long getChildCountXor(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (childCountXor < 0){
            createStatistics(fmInstance.featureModel());
        }
        return childCountXor;
    }
    protected long getChildCountOr(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (childCountOr < 0){
            createStatistics(fmInstance.featureModel());
        }
        return childCountOr;
    }
    protected long getGroupedXor(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (groupedXor < 0){
            createStatistics(fmInstance.featureModel());
        }
        return groupedXor;
    }
    protected long getGroupedOr(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (groupedOr < 0){
            createStatistics(fmInstance.featureModel());
        }
        return groupedOr;
    }

    protected long getOptionalFeatures(FMInstance fmInstance, int timeout) throws Exception{
        checkInstance(fmInstance);
        return getFeatureCount(fmInstance) - getNoMandatoryFeatures(fmInstance, timeout) - getNoDeadFeatures(fmInstance, timeout);
    }

    protected long getConstraintCount(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (constraintCount < 1){
            constraintCount = fmInstance.featureModel().getConstraintCount();
        }
        return constraintCount;
    }

    protected double getCompoundComplexity(FMInstance fmInstance, int timeout) throws Exception {
        checkInstance(fmInstance);
        return Math.pow(getFeatureCount(fmInstance), 2)
                + (Math.pow(getNoMandatoryFeatures(fmInstance, timeout), 2)
                    + 2 * Math.pow(getOrGroups(fmInstance), 2)
                    + 3 * Math.pow(getXorGroups(fmInstance), 2)
                    + 3 * Math.pow(getOrGroups(fmInstance) + getXorGroups(fmInstance), 2)
                    + 3 * Math.pow(getConstraintCount(fmInstance), 2)
                  ) / 9;
    }

    protected double getConnectivityDensityCoefficent(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (getFeatureCount(fmInstance) -1 <= 0) return 0;
        else return (double) (getFeatureCount(fmInstance) -1) / getFeatureCount(fmInstance);
    }

    protected long getCountGrouped(FMInstance fmInstance) {
        checkInstance(fmInstance);
        return getGroupedOr(fmInstance) + getGroupedXor(fmInstance);
    }

    protected long getCountGroups(FMInstance fmInstance) {
        checkInstance(fmInstance);
        return getOrGroups(fmInstance) + getXorGroups(fmInstance);
    }


    public double getVariabilityRation(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (variablityRatio < 0) {
            long sum = fmInstance.featureModel().getFeatures().stream().mapToInt(e -> e.getStructure().getChildrenCount()).sum();
            variablityRatio = sum / ((double) getFeatureCount(fmInstance) - getLeafFeatures(fmInstance));
        }
        return variablityRatio;
    }


    public long getNoFeaturesConstraintRefExceptParents(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (noFeaturesConstraintRefExceptParent < 0) computeConnectivity(fmInstance);
        return noFeaturesConstraintRefExceptParent;
    }

    public double getMeanRefdFeaturesInConstraintsExceptParentPerFeature(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (meanRefdFeaturesInConstraintsExceptParentPerFeature < 0) computeConnectivity(fmInstance);
        return meanRefdFeaturesInConstraintsExceptParentPerFeature;
    }

    public long getNoFeaturesWithChildren(FMInstance fmInstance)  {
        checkInstance(fmInstance);
        if (noFeaturesWithChildren < 0) {
            noFeaturesWithChildren = fmInstance.featureModel().getFeatures().stream().filter(e -> e.getStructure().getChildrenCount() > 0).count();
        }
        return noFeaturesWithChildren;
    }

    public long getNoConstrainedFeatures(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if(Objects.isNull(constrainedFeatures)) computeConstrainedFeatures(fmInstance.featureModel());
        return constrainedFeatures.size();
    }




}
