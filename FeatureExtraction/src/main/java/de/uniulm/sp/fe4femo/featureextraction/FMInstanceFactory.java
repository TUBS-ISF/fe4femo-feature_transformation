package de.uniulm.sp.fe4femo.featureextraction;


import de.ovgu.featureide.fm.attributes.format.XmlExtendedFeatureModelFormat;
import de.ovgu.featureide.fm.core.analysis.cnf.formula.FeatureModelFormula;
import de.ovgu.featureide.fm.core.base.IFeatureModel;
import de.ovgu.featureide.fm.core.configuration.FeatureIDEFormat;
import de.ovgu.featureide.fm.core.io.dimacs.DimacsWriter;
import de.ovgu.featureide.fm.core.io.manager.FeatureModelManager;
import de.ovgu.featureide.fm.core.io.manager.IFeatureModelManager;
import de.ovgu.featureide.fm.core.io.uvl.UVLFeatureModelFormat;
import de.ovgu.featureide.fm.core.io.xml.XmlFeatureModelFormat;
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
            Path pathDimacs = Files.createTempFile("fm_dimacs", ".dimacs");
            FeatureModelFormula formula = new FeatureModelFormula(featureModel);
            formula.getCNFNode();

            final DimacsWriter dWriter = new DimacsWriter(formula.getCNF());
            Files.writeString(pathDimacs, dWriter.write());

            Path pathUVL = Files.createTempFile("fm_uvl", ".uvl");
            Files.writeString(pathUVL, new UVLFeatureModelFormat().write(featureModel));

            Path pathXML = Files.createTempFile("fm_xml", ".xml");
            FeatureModelManager.save(featureModel, pathXML, new XmlFeatureModelFormat());

            return Optional.of(new FMInstance(pathFM, pathDimacs, pathUVL, pathXML, featureModel, formula));
        } catch (IOException e) {
            LOGGER.error("Could not initialise feature model {}", pathFM, e);
            return Optional.empty();
        }
    }

}
