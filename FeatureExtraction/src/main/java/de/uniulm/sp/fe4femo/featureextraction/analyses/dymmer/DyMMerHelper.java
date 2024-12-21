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
import java.util.stream.Collectors;

public class DyMMerHelper {

    private FMInstance fmInstance;

    private FeatureModelAnalyzer featureModelAnalyzer;
    private long mandatoryFeatures;
    private long optionalFeatures;
    private long coreFeatures;
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
    private long noEdgesConnectivity;
    private long noFeaturesWithChildren;

    private long orGroups;
    private long xorGroups;
    private double branchingFactorMedian;
    private double branchingFactorMean;
    private int branchingFactorMax;
    private Set<IFeature> constrainedFeatures = null;
    private int rootCount;
    private long singleCyclicDependentFeatures;
    private long multiCyclicDependentFeatures;
    private long childCountXor;
    private long childCountOr;
    private long groupedXor;
    private long groupedOr;


    public DyMMerHelper() {
        invalidateAll();
    }

    private void invalidateAll() {
        mandatoryFeatures = -1;
        optionalFeatures = -1;
        coreFeatures = -1;
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
        noEdgesConnectivity = -1;
        noFeaturesWithChildren = -1;

        orGroups = -1;
        xorGroups = -1;
        branchingFactorMedian = -1;
        branchingFactorMean = -1;
        branchingFactorMax = -1;
        constrainedFeatures = null;
        rootCount = -1;
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
        coreFeatures = fmInstance.fmFormula().getCNF().getVariables().convertToString(result, true, false, false).size();
        deadFeatures = fmInstance.fmFormula().getCNF().getVariables().convertToString(result, false, true, false).size();
    }

    private void generateLeafChildrenStats(IFeatureModel featureModel) throws InterruptedException{
        List<IFeature> leafFeatureList = featureModel.getFeatures().stream().filter(e -> ! e.getStructure().hasChildren()).toList();
        leafFeatures = leafFeatureList.size();
        List<Long> treeDepths = new ArrayList<>(leafFeatureList.size());
        for (IFeature iFeature : leafFeatureList) {
            if (Thread.currentThread().isInterrupted()) throw new InterruptedException();
            treeDepths.add(getFeatureDepth(iFeature.getStructure()));
        }
        meanDepth = treeDepths.stream().mapToLong(i -> i).average().orElse(0.0);
        maxDepth = treeDepths.stream().mapToLong(i -> i).max().orElse(0);
        medianDepth = getMedian(treeDepths);
    }

    private static long getFeatureDepth(IFeatureStructure feature) {
        if (Objects.isNull(feature.getParent())) return 1;
        return getFeatureDepth(feature.getParent()) + 1;
    }

    private void computeConstrainedFeatures(IFeatureModel featureModel) {
        constrainedFeatures = featureModel.getConstraints().stream().flatMap(e -> e.getContainedFeatures().stream()).collect(Collectors.toSet());
    }

    private void generateBranchingFactors(IFeatureModel featureModel) {
        List<Integer> childCounts = featureModel.getFeatures().stream().map(IFeature::getStructure).mapToInt(IFeatureStructure::getChildrenCount).boxed().toList();
        IntSummaryStatistics childrenStatistics = childCounts.stream().mapToInt(i -> i ).summaryStatistics();
        branchingFactorMax = childrenStatistics.getMax();
        branchingFactorMean = childrenStatistics.getAverage();
        branchingFactorMedian = getMedian(childCounts.stream().mapToLong(i -> i).boxed().toList());
    }

    private static double getMedian(List<Long> input) {
        List<Long> sortedList = input.stream().sorted().toList();
        if (sortedList.size() % 2 == 0) return sortedList.get(sortedList.size() / 2);
        else return (sortedList.get(sortedList.size() / 2 ) + sortedList.get(sortedList.size() / 2 +1))/2.0;
    }

    private void computeConnectivity(FMInstance fmInstance) throws InterruptedException {
        Map<IFeature, Set<IFeature>> connectedFeatures = new HashMap<>();
        for (IConstraint constraint : fmInstance.featureModel().getConstraints()) {
            for (IFeature feature : constraint.getContainedFeatures()) {
                if (Thread.currentThread().isInterrupted()) throw new InterruptedException();
                List<IFeature> constraintCopy = constraint.getContainedFeatures().stream()
                        .filter(e -> ! Objects.equals(e, feature))
                        .filter(e -> ! Objects.equals(feature.getStructure().getParent(), e.getStructure()))
                        .toList();
                connectedFeatures.putIfAbsent(feature, new HashSet<>());
                connectedFeatures.get(feature).addAll(constraintCopy);
            }
        }
        noFeaturesConstraintRefExceptParent = connectedFeatures.entrySet().stream().filter(e -> ! e.getValue().isEmpty()).count();
        noEdgesConnectivity = connectedFeatures.values().stream().mapToLong(Set::size).sum() / 2;
        meanRefdFeaturesInConstraintsExceptParentPerFeature = connectedFeatures.values().stream().filter(iFeatures -> !iFeatures.isEmpty()).mapToInt(Set::size).average().orElse(0.0);
    }

    // ------------------------------

    public long getNoMandatoryFeatures(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (mandatoryFeatures == -1){
            mandatoryFeatures = fmInstance.featureModel().getFeatures().stream().filter(e -> e.getStructure().isMandatory()).count();
        }
        return mandatoryFeatures;
    }

    public long getNoCoreFeatures(FMInstance fmInstance, int timeout) throws Exception {
        checkInstance(fmInstance);
        if (coreFeatures == -1){
            manDeadAnalysis(fmInstance, timeout);
        }
        return coreFeatures;
    }

    public long getNoDeadFeatures(FMInstance fmInstance, int timeout) throws Exception {
        checkInstance(fmInstance);
        if (deadFeatures == -1){
            manDeadAnalysis(fmInstance, timeout);
        }
        return deadFeatures;
    }

    public long getFeatureCount(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (featureCount == -1){
            featureCount = fmInstance.featureModel().getNumberOfFeatures();
        }
        return featureCount;
    }

    public long getTopFeatures(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (topFeatures == -1){
            topFeatures = fmInstance.featureModel().getStructure().getRoot().getChildrenCount();
        }
        return topFeatures;
    }

    public long getLeafFeatures(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (leafFeatures == -1){
            generateLeafChildrenStats(fmInstance.featureModel());
        }
        return leafFeatures;
    }

    public long getMaxDepth(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (maxDepth == -1){
            generateLeafChildrenStats(fmInstance.featureModel());
        }
        return maxDepth;
    }

    public double getMedianDepth(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (medianDepth < 0){
            generateLeafChildrenStats(fmInstance.featureModel());
        }
        return medianDepth;
    }

    public double getMeanDepth(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (meanDepth < 0) {
            generateLeafChildrenStats(fmInstance.featureModel());
        }
        return meanDepth;
    }

    public long getOrGroups(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (orGroups == -1){
            orGroups = fmInstance.featureModel().getFeatures().stream().filter(e -> e.getStructure().isOr()).count();
        }
        return orGroups;
    }

    public long getXorGroups(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (xorGroups == -1){
            xorGroups = fmInstance.featureModel().getFeatures().stream().filter(e -> e.getStructure().isAlternative()).count();
        }
        return xorGroups;
    }

    public double getBranchingFactorMedian(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (branchingFactorMedian < 0){
            generateBranchingFactors(fmInstance.featureModel());
        }
        return branchingFactorMedian;
    }
    public double getBranchingFactorMean(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (branchingFactorMean < 0){
            generateBranchingFactors(fmInstance.featureModel());
        }
        return branchingFactorMean;
    }
    public int getBranchingFactorMax(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (branchingFactorMax < 0){
            generateBranchingFactors(fmInstance.featureModel());
        }
        return branchingFactorMax;
    }
    public int getRootCount(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (rootCount < 0){
            rootCount = 1;
        }
        return rootCount;
    }
    public long getSingleCyclicDependentFeatures(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (singleCyclicDependentFeatures < 0){
            if (constrainedFeatures == null) computeConstrainedFeatures(fmInstance.featureModel());
            singleCyclicDependentFeatures = constrainedFeatures.stream()
                    .filter(e -> e.getStructure().isAlternative() || (e.getStructure().getParent() != null && e.getStructure().getParent().isAlternative()))
                    .count();
        }
        return singleCyclicDependentFeatures;
    }
    public long getMultiCyclicDependentFeatures(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (multiCyclicDependentFeatures < 0){
            if (constrainedFeatures == null) computeConstrainedFeatures(fmInstance.featureModel());
            multiCyclicDependentFeatures = constrainedFeatures.stream()
                    .filter(e -> e.getStructure().isOr() || (e.getStructure().getParent() != null && e.getStructure().getParent().isOr()))
                    .count();
        }
        return multiCyclicDependentFeatures;
    }
    public long getChildCountXor(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (childCountXor < 0){
            childCountXor = fmInstance.featureModel().getFeatures().stream().map(IFeature::getStructure).filter(IFeatureStructure::isAlternative).mapToLong(IFeatureStructure::getChildrenCount).sum();
        }
        return childCountXor;
    }
    public long getChildCountOr(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (childCountOr < 0){
            childCountOr = fmInstance.featureModel().getFeatures().stream().map(IFeature::getStructure).filter(IFeatureStructure::isOr).mapToLong(IFeatureStructure::getChildrenCount).sum();
        }
        return childCountOr;
    }
    public long getGroupedXor(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (groupedXor < 0){
            groupedXor = fmInstance.featureModel().getFeatures().stream().filter(e -> (e.getStructure().getParent() != null) && e.getStructure().getParent().isAlternative()).count();
        }
        return groupedXor;
    }
    public long getGroupedOr(FMInstance fmInstance){
        checkInstance(fmInstance);
        if (groupedOr < 0){
            groupedOr = fmInstance.featureModel().getFeatures().stream().filter(e -> (e.getStructure().getParent() != null) && e.getStructure().getParent().isOr()).count();
        }
        return groupedOr;
    }

    public long getOptionalFeatures(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if(optionalFeatures < 0) {
            optionalFeatures = fmInstance.featureModel().getFeatures().stream().map(IFeature::getStructure)
                    .filter(e -> ! e.isMandatory())
                    .filter(e -> ! e.getParent().isAlternative())
                    .filter(e -> ! e.getParent().isOr())
                    .count();
        }
        return optionalFeatures;
    }

    public long getConstraintCount(FMInstance fmInstance) {
        checkInstance(fmInstance);
        if (constraintCount < 1){
            constraintCount = fmInstance.featureModel().getConstraintCount();
        }
        return constraintCount;
    }

    public double getCompoundComplexity(FMInstance fmInstance)  {
        checkInstance(fmInstance);
        return Math.pow(getFeatureCount(fmInstance), 2)
                + (Math.pow(getNoMandatoryFeatures(fmInstance), 2)
                    + 2 * Math.pow(getOrGroups(fmInstance), 2)
                    + 3 * Math.pow(getXorGroups(fmInstance), 2)
                    + 3 * Math.pow( getOrGroups(fmInstance) + getXorGroups(fmInstance), 2)
                    + 3 * Math.pow(getConstraintCount(fmInstance), 2)
                  ) / 9;
    }

    public double getConnectivityDensityCoefficent(FMInstance fmInstance) throws InterruptedException {
        checkInstance(fmInstance);
        if (noEdgesConnectivity < 0) computeConnectivity(fmInstance);
        return (double) noEdgesConnectivity / getFeatureCount(fmInstance);
    }

    public long getCountGrouped(FMInstance fmInstance) {
        checkInstance(fmInstance);
        return getGroupedOr(fmInstance) + getGroupedXor(fmInstance);
    }

    public long getCountGroups(FMInstance fmInstance) {
        checkInstance(fmInstance);
        return getOrGroups(fmInstance) + getXorGroups(fmInstance);
    }


    public double getVariabilityRatio(FMInstance fmInstance) throws InterruptedException {
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
