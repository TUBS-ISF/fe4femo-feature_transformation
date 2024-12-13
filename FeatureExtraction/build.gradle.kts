plugins {
    id("java")
    application
    id("com.gradleup.shadow") version "8.3.5"
    id("com.coditory.manifest") version "1.1.0"
}

group = "de.uniulm.sp.fe4femo.featureextraction"
version = "1.0-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_21

repositories {
    mavenCentral()
    gradlePluginPortal()
}


val jackson = "2.18.2"

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")


    // https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-core
    implementation("com.fasterxml.jackson.core:jackson-core:$jackson")
    // https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind
    implementation("com.fasterxml.jackson.core:jackson-databind:$jackson")
    // https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-annotations
    implementation("com.fasterxml.jackson.core:jackson-annotations:$jackson")
    // https://mvnrepository.com/artifact/com.fasterxml.jackson.datatype/jackson-datatype-jsr310
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:$jackson")


    //implementation(files("external/featjar/featjar-1.0-SNAPSHOT-all.jar"))
    implementation(project(":feature-model-batch-analysis")) //also exposes FeatureIDE + log4j

}

application {
    mainClass = "de.uniulm.sp.fe4femo.featureextraction.ExtractionHandler"
    //TODO applicationDefaultJvmArgs = listOf("-Djava.io.tmpdir=\$TMPDIR")
}

manifest{
    attributes = mapOf("Multi-Release" to "true")
}

tasks.named("shadowJar", com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar::class.java) {
    transform(com.github.jengelman.gradle.plugins.shadow.transformers.Log4j2PluginsCacheFileTransformer::class.java)
}

tasks.test {
    useJUnitPlatform()
}