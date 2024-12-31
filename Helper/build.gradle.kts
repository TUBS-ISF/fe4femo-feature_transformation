plugins {
    id("java")
    id("application")
}

group = "de.uniulm.sp.fe4femo.helper"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

application {
    mainClass = "de.uniulm.sp.fe4femo.helper.Main"
}

val log4j="2.24.2"

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")

    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-api
    implementation("org.apache.logging.log4j:log4j-api:$log4j")
    // https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-core
    runtimeOnly("org.apache.logging.log4j:log4j-core:$log4j")
    runtimeOnly("org.apache.logging.log4j:log4j-layout-template-json:$log4j")

    // https://mvnrepository.com/artifact/org.apache.commons/commons-csv
    implementation("org.apache.commons:commons-csv:1.12.0")
}

tasks.test {
    useJUnitPlatform()
}
