plugins {
    id("java")
}

group = "de.uniulm.sp.fe4femo.featureextraction"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")

    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-api
    implementation("org.apache.logging.log4j:log4j-api:2.24.2")
    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-core
    runtimeOnly("org.apache.logging.log4j:log4j-core:2.24.2")
    runtimeOnly("org.apache.logging.log4j:log4j-layout-template-json")

}

tasks.test {
    useJUnitPlatform()
}