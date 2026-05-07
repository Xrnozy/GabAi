package com.example.gabai.modules

import android.content.Context
import com.example.gabai.WebRTCClient
import com.example.gabai.services.TtsService
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class OcrModule(
    context: Context,
    private val tts: TtsService,
) : WebRTCClient.NavigationListener {
    private val modelManager = ModelManager()
    private val serverManager = ServerManager(context)
    private val started = AtomicBoolean(false)

    var onStatusText: ((String) -> Unit)? = null

    fun start() {
        if (!started.compareAndSet(false, true)) return

        // Lazy init: OCR-only pipeline.
        modelManager.ensureOcrModelsInitialized()

        val rtc = serverManager.getOrCreateWebRtcClient(mode = "ocr")
        rtc.navigationListener = this
        rtc.startCamera()
        rtc.createOffer()
    }

    fun stop() {
        if (!started.compareAndSet(true, false)) return
        modelManager.disposeOcrModels()
        serverManager.releaseWebRtc()
    }

    override fun onNavigationCommand(command: String, fullData: JSONObject) {
        onStatusText?.invoke(command)
        val shouldSpeak = fullData.optBoolean("should_speak", false)
        if (shouldSpeak) {
            tts.speak(command)
        }
    }
}
