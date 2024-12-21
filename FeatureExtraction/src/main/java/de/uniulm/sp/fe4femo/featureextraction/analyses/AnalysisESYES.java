package de.uniulm.sp.fe4femo.featureextraction.analyses;

import de.ovgu.featureide.fm.core.FeatureModelAnalyzer;
import de.ovgu.featureide.fm.core.analysis.cnf.LiteralSet;
import de.ovgu.featureide.fm.core.analysis.cnf.analysis.AtomicSetAnalysis;
import de.ovgu.featureide.fm.core.analysis.cnf.analysis.CoreDeadAnalysis;
import de.ovgu.featureide.fm.core.base.IFeature;
import de.ovgu.featureide.fm.core.base.IFeatureStructure;
import de.ovgu.featureide.fm.core.job.monitor.NullMonitor;
import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analyses.dymmer.DyMMerHelper;
import de.uniulm.sp.fe4femo.featureextraction.analysis.Analysis;
import de.uniulm.sp.fe4femo.featureextraction.analysis.AnalysisStep;
import de.uniulm.sp.fe4femo.featureextraction.analysis.IntraStepResult;
import de.uniulm.sp.fe4femo.featureextraction.analysis.StatusEnum;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;

// https://www.sciencedirect.com/science/article/pii/S0950584918301873
public class AnalysisESYES extends Analysis {

    protected static final Logger LOGGER = LogManager.getLogger();


    public AnalysisESYES() {
        super("ESYES", Executors.newSingleThreadExecutor(), getAnalysisSteps());
    }

    protected static List<AnalysisStep> getAnalysisSteps() {
        return List.of(
                getAtomicSet(),
                getInterfaceComplexity(),
                getVPCC(),
                getOptionalMandatory(),
                getActualOptMand()
                //can't implement
                // - Proportion of features violating hierarchy rules
                // - Overall complexity
        );
    }

    protected static AnalysisStep getActualOptMand() {
        return new AnalysisStep() {

            @Override
            public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
                long noFeatures = fmInstance.featureModel().getNumberOfFeatures();
                List<IFeature> reallyMandatory = fmInstance.featureModel().getFeatures().stream().filter(e -> !e.getStructure().isRoot()).filter(e -> e.getStructure().isMandatory()).toList();
                List<IFeature> reallyOptional = fmInstance.featureModel().getFeatures().stream().filter(e -> !e.getStructure().isRoot()).filter(e -> !e.getStructure().isMandatory()).toList();

                FeatureModelAnalyzer analyzer = new FeatureModelAnalyzer(fmInstance.fmFormula());
                try {
                    List<IFeature> actualMandatory = analyzer.getFalseOptionalFeatures(new NullMonitor<>());

                    CoreDeadAnalysis coreDeadAnalysis = new CoreDeadAnalysis(fmInstance.fmFormula().getCNF());
                    coreDeadAnalysis.setTimeout(1000 * timeout);
                    LiteralSet result = coreDeadAnalysis.execute(new NullMonitor<>());
                    List<IFeature> actualDead = result == null ? List.of() : fmInstance.fmFormula().getCNF().getVariables().convertToString(result, false, true, false).stream().map(e -> fmInstance.featureModel().getFeature(e)).toList();

                    Set<IFeature> cract = new HashSet<>(reallyMandatory);
                    cract.addAll(actualMandatory);
                    Set<IFeature> vract = new HashSet<>(reallyOptional);
                    actualMandatory.forEach(vract::remove);

                    return new IntraStepResult(Map.of(
                            getAnalysesNames()[0], String.valueOf((double) cract.size() / noFeatures),
                            getAnalysesNames()[1], String.valueOf((double) vract.size() / noFeatures),
                            getAnalysesNames()[2], String.valueOf((double) actualDead.size() / noFeatures)
                    ), StatusEnum.SUCCESS);
                } catch (Exception e) {
                    LOGGER.warn("Error while computing ESYES false optional or dead features", e);
                    return new IntraStepResult(Map.of(), StatusEnum.ERROR);
                }
            }

            @Override
            public String[] getAnalysesNames() {
                return new String[]{"RatioActualVariable", "RatioActualCommonFeatures", "RatioActualDeadFeatures"};
            }
        };
    }

    protected static AnalysisStep getOptionalMandatory() {
        return new AnalysisStep() {

            private boolean checkMandatory(IFeatureStructure e) throws InterruptedException {
                if (Thread.currentThread().isInterrupted()) throw new InterruptedException();

                if (e.getParent() == null) return true;
                else if (e.isMandatory()) return checkMandatory(e.getParent());
                else return false;
            }

            @Override
            public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
                long pathMandatoryCount = 0;
                long pathOptionalCount = 0;
                long localMandatoryCount = 0;
                long localOptionalCount = 0;
                for (IFeature iFeature : fmInstance.featureModel().getFeatures()) {
                    if (Thread.currentThread().isInterrupted()) throw new InterruptedException();

                    IFeatureStructure e = iFeature.getStructure();
                    if (e.isRoot()) continue;

                    if (e.isMandatory()) localMandatoryCount++;
                    else localOptionalCount++;

                    if (checkMandatory(e)) pathMandatoryCount++;
                    else pathOptionalCount++;
                }
                DyMMerHelper dyMMerHelper = new DyMMerHelper();
                return new IntraStepResult(Map.of(
                        getAnalysesNames()[0], String.valueOf(pathMandatoryCount),
                        getAnalysesNames()[1], String.valueOf((double) pathMandatoryCount / dyMMerHelper.getFeatureCount(fmInstance)),
                        getAnalysesNames()[2], String.valueOf(pathOptionalCount),
                        getAnalysesNames()[3], String.valueOf((double) pathOptionalCount / dyMMerHelper.getFeatureCount(fmInstance)),
                        getAnalysesNames()[4], String.valueOf(localMandatoryCount),
                        getAnalysesNames()[5], String.valueOf((double) localMandatoryCount / dyMMerHelper.getFeatureCount(fmInstance)),
                        getAnalysesNames()[6], String.valueOf(localOptionalCount),
                        getAnalysesNames()[7], String.valueOf((double) localOptionalCount / dyMMerHelper.getFeatureCount(fmInstance))
                ), StatusEnum.SUCCESS);
            }

            @Override
            public String[] getAnalysesNames() {
                return new String[]{"pathMandatory", "RatioPathMandatory", "pathOptional", "RatioPathOptional", "localMandatory", "RatioLocalMandatory", "localOptional", "RatioLocalOptional"};
            }
        };
    }

    protected static AnalysisStep getVPCC() {
        return new AnalysisStep() {

            @Override
            public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
                DyMMerHelper dyMMerHelper = new DyMMerHelper();
                double value = Math.sqrt(Math.pow(dyMMerHelper.getConstraintCount(fmInstance), 2) + Math.pow(dyMMerHelper.getCountGroups(fmInstance), 2));
                return new IntraStepResult(Map.of(getAnalysesNames()[0], String.valueOf(value)), StatusEnum.SUCCESS);
            }

            @Override
            public String[] getAnalysesNames() {
                return new String[]{"#Variability_points_and_cyclomatic_complexity"};
            }
        };
    }

    protected static AnalysisStep getInterfaceComplexity() {
        return new AnalysisStep() {

            @Override
            public String[] getAnalysesNames() {
                return new String[]{"Interface_Complexity"};
            }

            @Override
            public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
                DyMMerHelper dyMMerHelper = new DyMMerHelper();
                long value = dyMMerHelper.getConstraintCount(fmInstance) + dyMMerHelper.getOrGroups(fmInstance) + dyMMerHelper.getXorGroups(fmInstance);
                return new IntraStepResult(Map.of(getAnalysesNames()[0], String.valueOf(value)), StatusEnum.SUCCESS);
            }
        };
    }

    protected static AnalysisStep getAtomicSet() {
        return new AnalysisStep() {

            @Override
            public String[] getAnalysesNames() {
                return new String[]{"#atomicSets"};
            }

            @Override
            public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
                AtomicSetAnalysis atomicSetAnalysis = new AtomicSetAnalysis(fmInstance.fmFormula().getCNF());
                atomicSetAnalysis.setTimeout(1000 * timeout);
                try {
                    List<LiteralSet> atomicSets = atomicSetAnalysis.analyze(new NullMonitor<>());
                    LOGGER.info("Successfully computed ESYES atomic sets");
                    return new IntraStepResult(Map.of(getAnalysesNames()[0], String.valueOf(atomicSets.size())), StatusEnum.SUCCESS);
                } catch (Exception e) {
                    LOGGER.warn("Error in ESYES atomic sets", e);
                    return new IntraStepResult(Map.of(), StatusEnum.ERROR);
                }
            }
        };
    }


}
