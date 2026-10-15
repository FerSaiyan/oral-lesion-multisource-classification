plugins {
    id("com.android.application") version "8.12.1"
    id("org.jetbrains.kotlin.android") version "2.3.10"
}

android {
    namespace = "org.bpex.benchmark"
    compileSdk = 35
    defaultConfig {
        applicationId = "org.bpex.benchmark"
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging {
        jniLibs.useLegacyPackaging = true
        resources.excludes += setOf("META-INF/LICENSE*", "META-INF/NOTICE*")
    }
}
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

val runtimeRoot = System.getProperty("user.home") + "/nvme_tmp/npu-litert230"
dependencies {
    implementation(files("$runtimeRoot/aar/litert-npu-qualcomm-2.3.0-qairt2.50.0.260828-v75.aar"))
    implementation("com.google.ai.edge.litert:litert-gpu:2.3.0") { isTransitive = false }
}
