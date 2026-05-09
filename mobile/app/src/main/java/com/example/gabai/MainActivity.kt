package com.example.gabai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.ViewCompat
import androidx.core.content.ContextCompat
import com.example.gabai.services.Haptics
import com.example.gabai.services.TtsService
import com.example.gabai.services.VoiceCommandService
import java.util.Calendar
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

class MainActivity : AppCompatActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var voiceCommands: VoiceCommandService

    private val openingMode = AtomicReference<Mode?>(null)

    private enum class Mode { NAVIGATION, ASL, OCR, EASY_OCR, COLOR }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val micGranted = permissions[Manifest.permission.RECORD_AUDIO] == true
        if (micGranted) {
            voiceCommands.start()
        } else {
            tts.speak("Microphone permission denied. You can still use the buttons.")
            Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)


        // Bind cards (now LinearLayouts instead of Buttons)
        val btnNavigation = findViewById<LinearLayout>(R.id.btn_navigation)
        val btnAsl = findViewById<LinearLayout>(R.id.btn_asl)
        val btnOcr = findViewById<LinearLayout>(R.id.btn_ocr)

        val btnColor = findViewById<LinearLayout>(R.id.btn_color)

        // Set dynamic greeting based on time of day
        val tvTitle = findViewById<TextView>(R.id.tv_title)
        tvTitle.text = getGreeting()


        // Ensure TalkBack focuses the primary actions early.
        ViewCompat.setAccessibilityHeading(tvTitle, true)
        btnNavigation.requestFocus()

        val speakButtonNameOnFocus: (android.view.View, Boolean, String) -> Unit =
            { _, hasFocus, label ->
                if (hasFocus) {
                    tts.speak(label)
                }
            }

        btnNavigation.setOnFocusChangeListener { v, hasFocus ->
            speakButtonNameOnFocus(v, hasFocus, "Navigation")
        }
        btnAsl.setOnFocusChangeListener { v, hasFocus ->
            speakButtonNameOnFocus(v, hasFocus, "A S L")
        }
        btnOcr.setOnFocusChangeListener { v, hasFocus ->
            speakButtonNameOnFocus(v, hasFocus, "O C R")
        }

        btnColor.setOnFocusChangeListener { v, hasFocus ->
            speakButtonNameOnFocus(v, hasFocus, "Color")
        }

        btnNavigation.setOnClickListener {
            haptics.click()
            tts.speak("Navigation")
            openMode(Mode.NAVIGATION)
        }

        btnAsl.setOnClickListener {
            haptics.click()
            tts.speak("A S L")
            openMode(Mode.ASL)
        }

        btnOcr.setOnClickListener {
            haptics.click()
            tts.speak("O C R")
            openMode(Mode.OCR)
        }


        btnColor.setOnClickListener {
            haptics.click()
            tts.speak("Color")
            openMode(Mode.COLOR)
        }

        voiceCommands = VoiceCommandService(
            context = this,
            onCommand = { phrase ->
                when (parseHomeCommand(phrase)) {
                    Mode.NAVIGATION -> {
                        openMode(Mode.NAVIGATION)
                        true
                    }
                    Mode.ASL -> {
                        openMode(Mode.ASL)
                        true
                    }
                    Mode.OCR -> {
                        openMode(Mode.OCR)
                        true
                    }
                    Mode.EASY_OCR -> {
                        openMode(Mode.EASY_OCR)
                        true
                    }
                    Mode.COLOR -> {
                        openMode(Mode.COLOR)
                        true
                    }
                    null -> false
                }
            },
            onError = { userMessage ->
                // Keep the homepage usable even if speech recognition fails.
                if (!userMessage.isNullOrBlank()) {
                    tts.speak(userMessage)
                }
            },
        )
    }

    override fun onResume() {
        super.onResume()
        openingMode.set(null)

        // Update greeting on resume (time may have changed)
        findViewById<TextView>(R.id.tv_title).text = getGreeting()

        // Immediate voice guidance (blind-friendly).
        tts.speak("Welcome. Say Navigation, A S L, O C R, Easy O C R, or Color.")

        ensureMicPermissionAndStartListening()
    }

    override fun onPause() {
        super.onPause()
        voiceCommands.stop()
    }

    private fun getGreeting(): String {
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        return when {
            hour < 12 -> "Good morning"
            hour < 17 -> "Good afternoon"
            else -> "Good evening"
        }
    }

    private fun ensureMicPermissionAndStartListening() {
        val micGranted =
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

        if (micGranted) {
            voiceCommands.start()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
        }
    }

    private fun openMode(mode: Mode) {
        val existing = openingMode.get()
        if (existing == mode) return

        if (!openingMode.compareAndSet(existing, mode)) return

        voiceCommands.stop()

        when (mode) {
            Mode.NAVIGATION -> {
                tts.speak("Opening Navigation mode")
                startActivity(Intent(this, NavigationActivity::class.java))
            }
            Mode.ASL -> {
                tts.speak("Opening A S L mode")
                startActivity(Intent(this, AslActivity::class.java))
            }
            Mode.OCR -> {
                tts.speak("Opening O C R mode")
                startActivity(Intent(this, OcrActivity::class.java))
            }
            Mode.EASY_OCR -> {
                tts.speak("Opening Easy O C R mode")
                startActivity(Intent(this, EasyOcrActivity::class.java))
            }
            Mode.COLOR -> {
                tts.speak("Opening Color mode")
                startActivity(Intent(this, ColorActivity::class.java))
            }
        }
    }

    private fun parseHomeCommand(raw: String): Mode? {
        val text = raw.trim().lowercase(Locale.ROOT)
        if (text.isBlank()) return null
        val normalized = text
            .replace(Regex("[^a-z0-9 ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
        if (normalized.isBlank()) return null

        val isAsl =
            normalized == "asl" ||
                normalized == "a s l" ||
                normalized == "open asl" ||
                normalized == "open a s l" ||
                normalized.contains(" asl ") ||
                normalized.startsWith("asl ") ||
                normalized.endsWith(" asl") ||
                normalized.contains(" a s l ") ||
                normalized.startsWith("a s l ") ||
                normalized.endsWith(" a s l") ||
                // Common recognizer mis-hear for "ASL"
                normalized.contains("as well")

        val isOcr =
            normalized == "ocr" ||
                normalized == "o c r" ||
                normalized == "open ocr" ||
                normalized == "open o c r" ||
                normalized == "text" ||
                normalized == "read text" ||
                normalized.contains(" ocr ") ||
                normalized.startsWith("ocr ") ||
                normalized.endsWith(" ocr") ||
                normalized.contains(" o c r ") ||
                normalized.startsWith("o c r ") ||
                normalized.endsWith(" o c r")

        val isEasyOcr =
            normalized == "easy ocr" ||
                normalized == "easy o c r" ||
                normalized == "open easy ocr" ||
                normalized == "open easy o c r" ||
                normalized.contains(" easy ocr ") ||
                normalized.startsWith("easy ocr ") ||
                normalized.endsWith(" easy ocr")

        val isNav =
            normalized == "navigation" ||
                normalized == "open navigation" ||
                normalized.contains(" navigation ") ||
                normalized.startsWith("navigation ") ||
                normalized.endsWith(" navigation")

        val isColor =
            normalized == "color" ||
                normalized == "colour" ||
                normalized == "open color" ||
                normalized == "open colour" ||
                normalized.contains(" color ") ||
                normalized.startsWith("color ") ||
                normalized.endsWith(" color") ||
                normalized.contains(" colour ") ||
                normalized.startsWith("colour ") ||
                normalized.endsWith(" colour")

        return when {
            // Prefer ASL when both keywords are present in the same noisy phrase.
            isAsl -> Mode.ASL
            isEasyOcr -> Mode.EASY_OCR
            isOcr -> Mode.OCR
            isNav -> Mode.NAVIGATION
            isColor -> Mode.COLOR
            else -> null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        voiceCommands.destroy()
        tts.shutdown()
    }
}