plugins {
    id("java")
    application
    id("com.gradleup.shadow") version "8.3.5"
}

group = "de.uniulm.sp.fe4femo.featureextraction"
version = "1.0-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_21

repositories {
    mavenCentral()
    gradlePluginPortal()
}

val log4j = "2.24.2"
val jackson = "2.18.2"

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")

    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-api
    implementation("org.apache.logging.log4j:log4j-api:$log4j")
    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-core
    runtimeOnly("org.apache.logging.log4j:log4j-core:$log4j")
    runtimeOnly("org.apache.logging.log4j:log4j-layout-template-json:$log4j")

    // https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-core
    implementation("com.fasterxml.jackson.core:jackson-core:$jackson")
    // https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind
    implementation("com.fasterxml.jackson.core:jackson-databind:$jackson")
    // https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-annotations
    implementation("com.fasterxml.jackson.core:jackson-annotations:$jackson")

    implementation(files("external/featjar/featjar-1.0-SNAPSHOT-all.jar"))
    implementation(project(":feature-model-batch-analysis"))

}

application {
    mainClass = "de.uniulm.sp.fe4femo.featureextraction.ExtractionHandler"
    //TODO applicationDefaultJvmArgs = listOf("-Djava.io.tmpdir=\$TMPDIR")
}

tasks.test {
    useJUnitPlatform()
}