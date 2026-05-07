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
import com.example.gabai.modules.NavigationModule
import com.example.gabai.services.Haptics
import com.example.gabai.services.TtsService
import java.util.Locale

class NavigationActivity : AppCompatActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var module: NavigationModule

    private lateinit var tvStatus: TextView

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        if (cameraGranted) {
            startNavigation()
        } else {
            tts.speak("Camera permission denied. Cannot start navigation.")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_navigation)

        // Navigation page uses dark theme — set status bar color
        window.statusBarColor = getColor(R.color.surface_dark)
        window.navigationBarColor = getColor(R.color.surface_dark)

        tvStatus = findViewById(R.id.tv_nav_status)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)

        module = NavigationModule(this, tts).apply {
            onStatusText = { text ->
                runOnUiThread { tvStatus.text = text }
            }
        }

        // Back button is now an ImageView
        findViewById<ImageView>(R.id.btn_nav_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        tts.speak("Navigation mode")
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
            startNavigation()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun startNavigation() {
        tvStatus.text = getString(R.string.nav_status_ready)
        module.start()
    }
}
