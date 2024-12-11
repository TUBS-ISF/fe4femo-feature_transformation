plugins {
    id("java")
    application
}

group = "de.uniulm.sp.fe4femo.featureextraction"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

val log4j = "2.24.2"

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")

    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-api
    implementation("org.apache.logging.log4j:log4j-api:$log4j")
    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-core
    runtimeOnly("org.apache.logging.log4j:log4j-core:$log4j")
    runtimeOnly("org.apache.logging.log4j:log4j-layout-template-json:$log4j")

}

application {
    mainClass = "de.uniulm.sp.fe4femo.featureextraction.ExtractionHandler"
    //TODO applicationDefaultJvmArgs = listOf("-Djava.io.tmpdir=\$TMPDIR")
}

tasks.test {
    useJUnitPlatform()
}