package de.uniulm.sp.fe4femo.featureextraction;

import de.ovgu.featureide.fm.core.analysis.cnf.formula.FeatureModelFormula;
import de.ovgu.featureide.fm.core.base.IFeatureModel;

import java.nio.file.Path;
import java.util.Objects;

public record FMInstance(Path originalPath, Path dimacsPath, Path uvlPath, Path xmlPath, IFeatureModel featureModel, FeatureModelFormula fmFormula) {
    @Override
    public String toString() {
        return originalPath.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;

        FMInstance that = (FMInstance) o;
        return Objects.equals(originalPath(), that.originalPath());
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(originalPath());
    }
}
