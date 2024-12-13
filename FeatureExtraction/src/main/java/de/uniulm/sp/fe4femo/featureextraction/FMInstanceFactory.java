package de.uniulm.sp.fe4femo.featureextraction;


import de.ovgu.featureide.fm.core.analysis.cnf.formula.FeatureModelFormula;
import de.ovgu.featureide.fm.core.base.IFeatureModel;
import de.ovgu.featureide.fm.core.io.dimacs.DimacsWriter;
import de.ovgu.featureide.fm.core.io.manager.FeatureModelManager;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.util.Optional;

public class FMInstanceFactory {

    private static final Logger LOGGER = LogManager.getLogger();

    public static Optional<FMInstance> createFMInstance(Path pathFM) {
        //TODO handle if dimacs use dimacs, else export dimacs
        try {
            IFeatureModel featureModel = FeatureModelManager.load(pathFM);
            if (featureModel == null) {
                LOGGER.error("Could not load feature model {}", pathFM);
                return Optional.empty();
            }
            Path pathDimacs = Files.createTempFile("fm_exp", ".dimacs");
            FeatureModelFormula formula = new FeatureModelFormula(featureModel);
            formula.getCNFNode();

            final DimacsWriter dWriter = new DimacsWriter(formula.getCNF());
            Files.writeString(pathDimacs, dWriter.write());

            return Optional.of(new FMInstance(pathFM, pathDimacs, featureModel, formula));
        } catch (IOException e) {
            LOGGER.error("Could not initialise feature model {}", pathFM, e);
            return Optional.empty();
        }
    }

}
