package com.example.gabai.modules

import android.content.Context
import com.example.gabai.WebRTCClient
import com.example.gabai.services.TtsService
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class NavigationModule(
    context: Context,
    private val tts: TtsService,
) : WebRTCClient.NavigationListener {

    private val modelManager = ModelManager()
    private val serverManager = ServerManager(context)
    private val started = AtomicBoolean(false)

    var onStatusText: ((String) -> Unit)? = null

    fun start() {
        if (!started.compareAndSet(false, true)) return

        // Lazy init: only now load navigation pipeline & connect.
        modelManager.ensureNavigationModelsInitialized()

        val rtc = serverManager.getOrCreateWebRtcClient(mode = "navigation")
        rtc.navigationListener = this
        rtc.startCamera()
        rtc.createOffer()
    }

    fun stop() {
        if (!started.compareAndSet(true, false)) return
        serverManager.releaseWebRtc()
        modelManager.disposeNavigationModels()
    }

    override fun onNavigationCommand(command: String, fullData: JSONObject) {
        onStatusText?.invoke(command)

        val shouldSpeak = fullData.optBoolean("should_speak", false)
        if (shouldSpeak) {
            tts.speak(command)
        }
    }
}

