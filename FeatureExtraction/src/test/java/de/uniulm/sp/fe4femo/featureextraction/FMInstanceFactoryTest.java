package de.uniulm.sp.fe4femo.featureextraction;

import org.collection.fm.util.FMUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

class FMInstanceFactoryTest {

    @BeforeAll
    static void beforeAll() {
        FMUtils.installLibraries();
        Locale.setDefault(Locale.US);
    }

    @Test
    void testSolettaProblem() {
        Path modelPath = Path.of("src/test/resources/test_soletta.uvl");
        FMInstance featureModel = FMInstanceFactory.createFMInstance(modelPath).orElseThrow();
        assertNotNull(featureModel.featureModel());
    }
}