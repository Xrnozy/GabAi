package com.example.gabai.services

import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

class VoiceCommandService(
    private val context: Context,
    private val locale: Locale = Locale.US,
    private val onCommand: (String) -> Boolean,
    private val onError: (String?) -> Unit = {},
) {

    private val mainHandler = Handler(Looper.getMainLooper())
    private val enabled = AtomicBoolean(false)
    private val listening = AtomicBoolean(false)

    private var recognizer: SpeechRecognizer? = null

    fun start() {
        if (!enabled.compareAndSet(false, true)) return
        ensureRecognizer()
        listen()
    }

    fun stop() {
        enabled.set(false)
        listening.set(false)
        recognizer?.cancel()
    }

    fun destroy() {
        stop()
        recognizer?.destroy()
        recognizer = null
    }

    private fun ensureRecognizer() {
        if (recognizer != null) return
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            onError("Speech recognition is not available on this device.")
            return
        }

        recognizer = SpeechRecognizer.createSpeechRecognizer(context.applicationContext).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: android.os.Bundle?) {
                    listening.set(true)
                }

                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}

                override fun onError(error: Int) {
                    listening.set(false)
                    if (!enabled.get()) return

                    val message = when (error) {
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS ->
                            "Microphone permission is required for voice commands."
                        SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT ->
                            "Network error. Voice commands may be unavailable."
                        SpeechRecognizer.ERROR_NO_MATCH ->
                            null // Common while idle; don't spam.
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT ->
                            null // Normal; just restart listening.
                        else -> null
                    }

                    if (message != null) onError(message)
                    scheduleRestart()
                }

                override fun onResults(results: android.os.Bundle?) {
                    listening.set(false)
                    if (!enabled.get()) return

                    val matches = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        .orEmpty()
                    // Try all recognition hypotheses; stop at first accepted command.
                    for (candidate in matches) {
                        val phrase = candidate.orEmpty().trim()
                        if (phrase.isNotBlank() && onCommand(phrase)) break
                    }

                    scheduleRestart()
                }

                override fun onPartialResults(partialResults: android.os.Bundle?) {}
                override fun onEvent(eventType: Int, params: android.os.Bundle?) {}
            })
        }
    }

    private fun listen() {
        if (!enabled.get()) return
        if (listening.get()) return

        val sr = recognizer ?: return

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, locale)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
        }

        try {
            sr.startListening(intent)
        } catch (_: SecurityException) {
            onError("Microphone permission is required for voice commands.")
        } catch (_: Exception) {
            onError("Voice command system error.")
        }
    }

    private fun scheduleRestart() {
        if (!enabled.get()) return
        mainHandler.postDelayed({ listen() }, 250)
    }
}

