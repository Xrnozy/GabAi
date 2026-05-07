package com.example.gabai.services

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import java.util.Locale

class TtsService(
    context: Context,
    private val locale: Locale = Locale.US,
) : TextToSpeech.OnInitListener {

    private val mainHandler = Handler(Looper.getMainLooper())

    private var tts: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var initialized = false
    private var pending: String? = null

    private var lastSpokenNormalized: String = ""
    private var lastSpokenAtMs: Long = 0L

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) return
        val engine = tts ?: return

        val result = engine.setLanguage(locale)
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            return
        }

        initialized = true
        pending?.let {
            pending = null
            speak(it)
        }
    }

    fun speak(text: String) {
        val trimmed = text.trim()
        if (trimmed.isBlank()) return

        if (!initialized) {
            pending = trimmed
            return
        }

        val engine = tts ?: return
        val normalized = trimmed.lowercase(locale)
        val now = android.os.SystemClock.elapsedRealtime()

        // Avoid TTS spam (duplicate focus events / repeated voice command echoes).
        val tooSoon = (now - lastSpokenAtMs) < 700
        if (normalized == lastSpokenNormalized && (engine.isSpeaking || tooSoon)) {
            return
        }

        lastSpokenNormalized = normalized
        lastSpokenAtMs = now

        // Flush keeps the UX crisp for blind-friendly prompts.
        mainHandler.post {
            engine.speak(trimmed, TextToSpeech.QUEUE_FLUSH, null, "gabai-tts")
        }
    }

    fun shutdown() {
        val engine = tts ?: return
        tts = null
        mainHandler.post {
            engine.stop()
            engine.shutdown()
        }
    }
}

