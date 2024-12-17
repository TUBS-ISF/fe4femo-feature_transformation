package de.uniulm.sp.fe4femo.featureextraction.analyses.dymmer;

import de.uniulm.sp.fe4femo.featureextraction.FMInstance;
import de.uniulm.sp.fe4femo.featureextraction.analysis.AnalysisStep;
import de.uniulm.sp.fe4femo.featureextraction.analysis.IntraStepResult;
import de.uniulm.sp.fe4femo.featureextraction.analysis.StatusEnum;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Map;

public abstract class DyMMerStep implements AnalysisStep {
    protected static final Logger LOGGER = LogManager.getLogger();

    private final String[] names;
    protected final DyMMerHelper helper;

    protected DyMMerStep(String name, DyMMerHelper helper) {
        String fixedName = name.replace(" ", "_");
        this.names = new String[]{fixedName};
        this.helper = helper;
    }

    protected DyMMerStep(String[] names, DyMMerHelper helper) {
        this.names = Arrays.stream(names).map(e -> e.replace(" ", "_")).toArray(String[]::new);
        this.helper = helper;
    }

    protected abstract IntraStepResult doComputation(FMInstance fmInstance, int timeout) throws Exception;

    @Override
    public IntraStepResult analyze(FMInstance fmInstance, int timeout) throws InterruptedException {
        try {
            IntraStepResult result = doComputation(fmInstance, timeout);
            LOGGER.info("Successfully computed DyMMerStep {}", this::getAnalysesNames);
            return result;
        } catch (InterruptedException e){
            LOGGER.info("Interrupted DyMMerStep {}", this.getAnalysesNames(), e);
            throw new InterruptedException();
        } catch (Exception e){
            LOGGER.info("Error in DyMMerStep {}", this.getAnalysesNames(), e);
            return new IntraStepResult(Map.of(), StatusEnum.ERROR);
        }
    }

    @Override
    public String[] getAnalysesNames() {
        return names;
    }


}
