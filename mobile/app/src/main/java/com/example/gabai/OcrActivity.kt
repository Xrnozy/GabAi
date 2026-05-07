package com.example.gabai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.gabai.modules.OcrModule
import com.example.gabai.services.Haptics
import com.example.gabai.services.TtsService
import java.util.Locale

class OcrActivity : AppCompatActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var module: OcrModule

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        if (cameraGranted) {
            startOcr()
        } else {
            tts.speak("Camera permission denied. Cannot start O C R.")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ocr)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)

        val tvStatus = findViewById<android.widget.TextView>(R.id.tv_ocr_status)

        module = OcrModule(this, tts).apply {
            onStatusText = { text ->
                runOnUiThread { tvStatus.text = text }
            }
        }

        findViewById<android.widget.Button>(R.id.btn_ocr_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        tts.speak("O C R mode")
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
            startOcr()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun startOcr() {
        module.start()
    }
}
