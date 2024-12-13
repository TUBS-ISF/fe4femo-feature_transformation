package de.uniulm.sp.fe4femo.featureextraction;

import de.ovgu.featureide.fm.core.analysis.cnf.formula.FeatureModelFormula;
import de.ovgu.featureide.fm.core.base.IFeatureModel;

import java.nio.file.Path;

public record FMInstance(Path fmPath, Path dimacsPath, IFeatureModel featureModel, FeatureModelFormula fmFormula) {
    @Override
    public String toString() {
        return fmPath.toString();
    }
}
