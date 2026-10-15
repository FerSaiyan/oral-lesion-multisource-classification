package org.bpex.benchmark

import android.app.Activity
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.util.Log
import android.view.WindowManager
import android.widget.TextView
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.math.sqrt

class BenchmarkActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val label = TextView(this).apply { text = "BPEX benchmark running. Results remain on this device." }
        setContentView(label)
        if (intent.getBooleanExtra("prepare", false)) {
            File(getExternalFilesDir(null), "bpex-deployment").mkdirs()
            label.text = "Ready to receive BPEX models and frozen inputs."
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            return
        }
        val modelName = requireNotNull(intent.getStringExtra("model"))
        val backend = requireNotNull(intent.getStringExtra("backend"))
        val profile = intent.getStringExtra("npu_profile") ?: "runtime-default"
        val runId = requireNotNull(intent.getStringExtra("run_id"))
        require(modelName in listOf("effnet_b3", "effnet_b7", "vit_base", "vit_large"))
        require(backend in listOf("GPU", "NPU"))
        require(profile in listOf("runtime-default", "default-quiet", "burst"))
        label.text = "BPEX: $modelName $backend $profile. The app closes during cooldowns."
        require(runId.matches(Regex("[a-zA-Z0-9_-]+")))
        val root = File(getExternalFilesDir(null), "bpex-deployment")
        val result = File(root, "results/$runId").apply { check(mkdirs()) { "Run directory already exists" } }
        val power = getSystemService(POWER_SERVICE) as PowerManager
        val wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "bpex:benchmark")
        wakeLock.acquire(60 * 60 * 1000L)
        Thread {
            fun state(value: JSONObject) {
                val staging = File(result, "status.json.tmp")
                staging.writeText(value.toString(2))
                check(staging.renameTo(File(result, "status.json")))
            }
            try {
                state(JSONObject().put("status", "running").put("backend", backend).put("model", modelName))
                benchmark(root, result, modelName, backend, power)
                state(JSONObject().put("status", "completed").put("backend", backend).put("model", modelName))
                runOnUiThread { label.text = "Benchmark complete: $modelName $backend" }
            } catch (error: Throwable) {
                Log.e("BPEX_BENCH", "Benchmark failed", error)
                state(JSONObject().put("status", "failed").put("error", error.stackTraceToString()))
                runOnUiThread { label.text = "Benchmark failed: ${error.message}" }
            } finally {
                if (wakeLock.isHeld) wakeLock.release()
                runOnUiThread { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
        }.start()
    }

    private fun benchmark(root: File, result: File, name: String, backend: String, power: PowerManager) {
        if (backend == "NPU") check(Build.SOC_MODEL.uppercase() == "SM8650") { "Expected SM8650, found ${Build.SOC_MODEL}" }
        val size = when (name) { "effnet_b3" -> 384; "effnet_b7" -> 512; else -> 224 }
        val fixture = File(root, "fixture/$size")
        val manifest = JSONObject(File(fixture, "manifest.json").readText())
        check(manifest.getInt("n") == 1209)
        val selection = intent.getStringExtra("timing_indices")
        val timingIndices = when (selection) {
            null -> (0 until 1209).toList()
            "none" -> emptyList()
            else -> selection.split(",").map { it.toInt() }
        }
        check(timingIndices.distinct().size == timingIndices.size && timingIndices.all { it in 0 until 1209 })
        val file = if (backend == "NPU") File(root, "aot/${name}_SM8650.tflite") else File(root, "models/${name}_float.tflite")
        val expected = requireNotNull(intent.getStringExtra("model_sha256"))
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) { val n = stream.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
        }
        check(digest.digest().joinToString("") { "%02x".format(it) } == expected) { "Transferred model checksum mismatch" }
        val profile = intent.getStringExtra("npu_profile") ?: "runtime-default"
        val options = CompiledModel.Options(if (backend == "NPU") Accelerator.NPU else Accelerator.GPU)
        if (backend == "NPU" && profile != "runtime-default") {
            options.qualcommOptions = CompiledModel.QualcommOptions(
                htpPerformanceMode = if (profile == "burst") CompiledModel.QualcommOptions.HtpPerformanceMode.BURST else CompiledModel.QualcommOptions.HtpPerformanceMode.DEFAULT,
                logLevel = CompiledModel.QualcommOptions.LogLevel.ERROR,
                profiling = CompiledModel.QualcommOptions.Profiling.OFF
            )
            Log.i("BPEX_BENCH", "QNN profile=$profile; logLevel=ERROR; profiling=OFF; accelerator=NPU only")
        }
        val environment = Environment.create(mapOf(Environment.Option.DispatchLibraryDir to applicationInfo.nativeLibraryDir))
        val start = System.nanoTime()
        val model = try {
            CompiledModel.create(file.absolutePath, options, environment)
        } catch (error: Throwable) { environment.close(); throw error }
        val loadMs = (System.nanoTime() - start) / 1e6
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()
        val observations = mutableListOf<Double>()
        val pipelineObservations = mutableListOf<Double>()
        val logits = JSONArray()
        val telemetry = JSONArray()
        var maximumPss = Debug.getPss()
        fun input(index: Int): FloatArray {
            val pixels = File(fixture, "%04d.rgb".format(index)).readBytes()
            check(pixels.size == size * size * 3)
            return FloatArray(pixels.size) { (pixels[it].toInt() and 255) / 255f }
        }
        fun run(values: FloatArray): Pair<FloatArray, Double> {
            inputs[0].writeFloat(values)
            val started = System.nanoTime()
            model.run(inputs, outputs)
            val output = outputs[0].readFloat()
            val ms = (System.nanoTime() - started) / 1e6
            check(output.size == 4 && output.all { it.isFinite() })
            return output to ms
        }
        fun snapshot(stage: String) {
            maximumPss = maxOf(maximumPss, Debug.getPss())
            val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            telemetry.put(JSONObject().put("stage", stage).put("unix_ms", System.currentTimeMillis())
                .put("pss_kib", Debug.getPss()).put("thermal_status", power.currentThermalStatus)
                .put("battery_temperature_tenths_c", battery?.getIntExtra("temperature", -1)))
        }
        try {
            check(inputs.size == 1 && outputs.size == 1)
            val probe = input(0)
            repeat(intent.getIntExtra("warmups", 30)) { run(probe) }
            snapshot("after_warmup")
            File(result, "latency.csv").bufferedWriter().use { writer ->
                writer.write("pass,index,inference_ms,preprocessed_file_pipeline_ms\n")
                repeat(if (timingIndices.isEmpty()) 0 else intent.getIntExtra("passes", 3)) { pass ->
                    for (index in timingIndices) {
                        val pipelineStart = System.nanoTime()
                        val (output, ms) = run(input(index))
                        val pipelineMs = (System.nanoTime() - pipelineStart) / 1e6
                        observations += ms
                        pipelineObservations += pipelineMs
                        writer.write("${pass + 1},$index,$ms,$pipelineMs\n")
                        if (index % 200 == 0) snapshot("pass_${pass + 1}_image_$index")
                        check(power.currentThermalStatus < PowerManager.THERMAL_STATUS_CRITICAL) { "Critical device thermal status" }
                    }
                    writer.flush()
                    if (pass < intent.getIntExtra("passes", 3) - 1 && intent.getBooleanExtra("cool_between_passes", false)) {
                        snapshot("pass_${pass + 1}_cooldown_start")
                        val deadline = System.nanoTime() + 900_000_000_000L
                        while (power.currentThermalStatus != PowerManager.THERMAL_STATUS_NONE) {
                            check(System.nanoTime() < deadline) { "Device did not cool between timing passes" }
                            Thread.sleep(5000)
                        }
                        repeat(intent.getIntExtra("warmups", 30)) { run(probe) }
                        snapshot("pass_${pass + 2}_after_rewarm")
                    }
                }
            }
            val soakSeconds = intent.getIntExtra("soak_seconds", 90)
            val soak = JSONArray()
            val report = JSONObject().put("model", name).put("requested_backend", backend)
                .put("npu_profile", if (backend == "NPU") profile else "not_applicable")
                .put("qnn_log_level", if (backend == "NPU" && profile != "runtime-default") "ERROR" else "runtime_default")
                .put("fallback_policy", "No explicit CPU retry; inspect native logcat partition/delegation details")
                .put("model_sha256", expected).put("soc", Build.SOC_MODEL).put("device", Build.MODEL)
                .put("android", Build.VERSION.RELEASE).put("model_load_ms", loadMs)
                .put("inference", if (observations.isEmpty()) JSONObject.NULL else stats(observations))
                .put("preprocessed_file_pipeline", if (pipelineObservations.isEmpty()) JSONObject.NULL else stats(pipelineObservations))
                .put("pipeline_scope", "Cached letterboxed RGB file read, uint8-to-float, input upload, inference and output read; excludes JPEG decode and camera/UI")
                .put("observed_max_pss_kib", maximumPss).put("telemetry", telemetry).put("soak_windows", soak)
                .put("timing_status", if (timingIndices.isEmpty()) "not_requested" else "completed").put("soak_status", "pending")
                .put("timing_indices", JSONArray(timingIndices)).put("prediction_status", "pending")
                .put("soak_requested_seconds", soakSeconds)
                .put("cool_between_passes", intent.getBooleanExtra("cool_between_passes", false))
            File(result, "summary.json").writeText(report.toString(2))
            if (!intent.getBooleanExtra("prediction_evaluation", true)) {
                check(soakSeconds == 0) { "Timing-only runs exclude the soak" }
                report.put("prediction_status", "not_requested").put("prediction_images", 0)
                    .put("soak_status", "not_requested")
                File(result, "summary.json").writeText(report.toString(2))
                return
            }
            // Fidelity is a separate, untimed full-set pass, with idle gaps and thermal pauses.
            runOnUiThread { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            for (index in 0 until 1209) {
                if (power.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
                    snapshot("prediction_${index}_cooldown_start")
                    File(result, "logits_partial.json").writeText(logits.toString())
                    val deadline = System.nanoTime() + 900_000_000_000L
                    while (power.currentThermalStatus != PowerManager.THERMAL_STATUS_NONE) {
                        check(System.nanoTime() < deadline) { "Device did not cool during prediction evaluation" }
                        Thread.sleep(5000)
                    }
                }
                logits.put(JSONArray(run(input(index)).first.toList()))
                if ((index + 1) % 100 == 0) {
                    File(result, "logits_partial.json").writeText(logits.toString())
                    snapshot("prediction_${index + 1}")
                }
                Thread.sleep(20)
            }
            File(result, "logits.json").writeText(logits.toString())
            report.put("prediction_status", "completed").put("prediction_images", 1209)
            File(result, "summary.json").writeText(report.toString(2))
            val soakStart = System.nanoTime()
            var windowStart = soakStart
            var window = mutableListOf<Double>()
            var soakStatus = if (soakSeconds == 0) "not_requested" else "completed"
            while (System.nanoTime() - soakStart < soakSeconds * 1_000_000_000L) {
                if (power.currentThermalStatus >= PowerManager.THERMAL_STATUS_CRITICAL) {
                    soakStatus = "thermal_stopped"
                    snapshot("soak_thermal_stop")
                    break
                }
                window += run(probe).second
                if (System.nanoTime() - windowStart >= 10_000_000_000L) {
                    soak.put(stats(window).put("elapsed_seconds", (System.nanoTime() - soakStart) / 1e9))
                    snapshot("soak")
                    window = mutableListOf()
                    windowStart = System.nanoTime()
                }
            }
            if (window.isNotEmpty()) soak.put(stats(window).put("elapsed_seconds", (System.nanoTime() - soakStart) / 1e9).put("partial_window", true))
            snapshot("finished")
            report.put("soak_status", soakStatus).put("soak_elapsed_seconds", (System.nanoTime() - soakStart) / 1e9)
                .put("observed_max_pss_kib", maximumPss)
            File(result, "summary.json").writeText(report.toString(2))
        } finally {
            inputs.forEach { it.close() }; outputs.forEach { it.close() }
            model.close(); environment.close()
        }
    }

    private fun stats(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        val mean = values.average()
        val sd = sqrt(values.sumOf { (it - mean) * (it - mean) } / (values.size - 1).coerceAtLeast(1))
        val position = (sorted.size - 1) * 0.95
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.size - 1)
        val p95 = sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
        val median = if (sorted.size % 2 == 0) (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2 else sorted[sorted.size / 2]
        return JSONObject().put("n", values.size).put("mean_ms", mean).put("sd_ms", sd)
            .put("median_ms", median).put("p95_ms", p95)
    }
}
