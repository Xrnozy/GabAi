package com.example.gabai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.gabai.modules.ColorModule
import com.example.gabai.services.Haptics
import com.example.gabai.services.TtsService
import java.util.Locale

class ColorActivity : AppCompatActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var module: ColorModule

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        if (cameraGranted) {
            startColor()
        } else {
            tts.speak("Camera permission denied. Cannot start color mode.")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_color)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)

        val tvStatus = findViewById<TextView>(R.id.tv_color_status)

        module = ColorModule(this, tts).apply {
            onStatusText = { text ->
                runOnUiThread { tvStatus.text = text }
            }
        }

        // Back button is now an ImageView
        findViewById<ImageView>(R.id.btn_color_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }

        // Announce color button
        findViewById<android.widget.Button>(R.id.btn_color_announce).setOnClickListener {
            haptics.click()
            val color = tvStatus.text?.toString().orEmpty()
            if (color.isNotBlank()) {
                tts.speak(color)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        tts.speak("Color mode")
        ensurePermissionsAndStart()
    }

    override fun onPause() {
        super.onPause()
        module.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        module.stop()
        tts.shutdown()
    }

    private fun ensurePermissionsAndStart() {
        val cameraGranted =
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

        if (cameraGranted) {
            startColor()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun startColor() {
        module.start()
    }
}
