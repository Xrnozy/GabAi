package com.example.gabai

import com.example.lumenguide.FramePreprocessor


import android.content.Context
import android.util.Log
import org.webrtc.*
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.charset.Charset

class WebRTCClient(
    private val context: Context,
    private val mode: String = "navigation",
) {

    interface NavigationListener {
        fun onNavigationCommand(command: String, fullData: JSONObject)
    }

    var navigationListener: NavigationListener? = null

    private val serverUrl = "http://100.124.181.67:8080"

    private val eglBase = EglBase.create()

    private var peerConnectionFactory: PeerConnectionFactory
    private var peerConnection: PeerConnection? = null

    private var videoCapturer: CameraVideoCapturer? = null
    private var videoSource: VideoSource? = null
    private var localVideoTrack: VideoTrack? = null
    private var navigationChannel: DataChannel? = null

    companion object {
        private const val TAG = "WebRTCClient"
    }

    init {
        PeerConnectionFactory.initialize(
            PeerConnectionFactory.InitializationOptions.builder(context)
                .createInitializationOptions()
        )

        peerConnectionFactory = PeerConnectionFactory.builder()
            .setVideoEncoderFactory(
                DefaultVideoEncoderFactory(
                    eglBase.eglBaseContext,
                    true,
                    true
                )
            )
            .setVideoDecoderFactory(
                DefaultVideoDecoderFactory(eglBase.eglBaseContext)
            )
            .createPeerConnectionFactory()

        initPeerConnection()
    }

    private fun initPeerConnection() {

        val iceServers = listOf(
            PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer()
        )

        val config = PeerConnection.RTCConfiguration(iceServers).apply {
            sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
            bundlePolicy = PeerConnection.BundlePolicy.MAXBUNDLE
            rtcpMuxPolicy = PeerConnection.RtcpMuxPolicy.REQUIRE
            continualGatheringPolicy =
                PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
        }

        peerConnection = peerConnectionFactory.createPeerConnection(
            config,
            object : PeerConnection.Observer {

                override fun onIceCandidate(candidate: IceCandidate) {
                    println("[ICE] Candidate: ${candidate.sdp}")
                }

                override fun onConnectionChange(newState: PeerConnection.PeerConnectionState) {
                    println("[PC] State: $newState")
                }

                override fun onTrack(transceiver: RtpTransceiver) {
                    println("[TRACK] Remote: ${transceiver.receiver.track()?.kind()}")
                }

                override fun onIceConnectionChange(p0: PeerConnection.IceConnectionState) {}
                override fun onIceGatheringChange(p0: PeerConnection.IceGatheringState) {}
                override fun onSignalingChange(p0: PeerConnection.SignalingState) {}
                override fun onAddStream(p0: MediaStream) {}
                override fun onRemoveStream(p0: MediaStream) {}
                override fun onIceConnectionReceivingChange(p0: Boolean) {}
                override fun onIceCandidatesRemoved(p0: Array<out IceCandidate>) {}
                override fun onDataChannel(dc: DataChannel) {
                    Log.d(TAG, "[DC] onDataChannel: ${dc.label()} id=${dc.id()}")
                }
                override fun onRenegotiationNeeded() {}
            }
        )

        // Create negotiated navigation channel (must match server id=2)
        val init = DataChannel.Init().apply {
            negotiated = true
            id = 2
        }
        navigationChannel = peerConnection?.createDataChannel("navigation", init)
        Log.d(TAG, "[DC] Created negotiated navigation channel (id=2)")

        navigationChannel?.registerObserver(object : DataChannel.Observer {
            override fun onBufferedAmountChange(previous: Long) {}

            override fun onStateChange() {
                Log.d(TAG, "[DC] Navigation channel state: ${navigationChannel?.state()}")
            }

            override fun onMessage(buffer: DataChannel.Buffer) {
                val data = buffer.data
                val bytes = ByteArray(data.remaining())
                data.get(bytes)
                val json = String(bytes, Charset.forName("UTF-8"))
                try {
                    val obj = JSONObject(json)
                    val command = obj.optString("nav_command", "Forward")
                    Log.d(TAG, "[NAV] Received: $command")
                    navigationListener?.onNavigationCommand(command, obj)
                } catch (e: Exception) {
                    Log.e(TAG, "[NAV] Parse error: ${e.message}")
                }
            }
        })
    }

    // -------------------------
    // CAMERA
    // -------------------------
    fun startCamera() {
        val cameraEnumerator = Camera2Enumerator(context)
        val deviceName = cameraEnumerator.deviceNames.first()

        videoCapturer = cameraEnumerator.createCapturer(deviceName, null)
        videoSource = peerConnectionFactory.createVideoSource(false)
        videoSource?.setVideoProcessor(CustomVideoProcessor())

        val surfaceTextureHelper =
            SurfaceTextureHelper.create("CaptureThread", eglBase.eglBaseContext)

        videoCapturer?.initialize(
            surfaceTextureHelper,
            context,
            videoSource?.capturerObserver
        )

        // 🔥 OPTIMIZED FOR LOW INTERNET / MOBILE DATA
        videoCapturer?.startCapture(320, 240, 10)

        localVideoTrack =
            peerConnectionFactory.createVideoTrack("video", videoSource)

        val sender = peerConnection?.addTrack(localVideoTrack)

        // 🔥 FORCE STABLE QUALITY & LOW DELAY (MAIN FIX)
        val parameters = sender?.parameters
        parameters?.encodings?.forEach {
            it.maxBitrateBps = 250_000   // 250 kbps (mobile data friendly)
            it.minBitrateBps = 100_000   // prevent blur
            it.maxFramerate = 10         // smooth enough for nav, low bandwidth
            it.scaleResolutionDownBy = 1.0
        }
        sender?.parameters = parameters
    }
    // -------------------------
    // OFFER
    // -------------------------
    fun createOffer() {

        val constraints = MediaConstraints().apply {
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "false"))

            // Prefer H264
            optional.add(MediaConstraints.KeyValuePair("preferredVideoCodec", "H264"))
        }

        peerConnection?.createOffer(object : SdpObserver {

            override fun onCreateSuccess(desc: SessionDescription) {

                peerConnection?.setLocalDescription(object : SdpObserver {

                    override fun onSetSuccess() {
                        sendOfferToServer(desc)
                    }

                    override fun onSetFailure(p0: String?) {}
                    override fun onCreateSuccess(p0: SessionDescription?) {}
                    override fun onCreateFailure(p0: String?) {}

                }, desc)
            }

            override fun onCreateFailure(p0: String?) {}
            override fun onSetSuccess() {}
            override fun onSetFailure(p0: String?) {}

        }, constraints)
    }

    private fun sendOfferToServer(offer: SessionDescription) {
        Thread {
            try {
                val json = JSONObject().apply {
                    put("type", offer.type.canonicalForm())
                    put("sdp", offer.description)
                    put("mode", mode)
                }

                val url = java.net.URL("$serverUrl/offer")
                val conn = url.openConnection() as java.net.HttpURLConnection

                conn.requestMethod = "POST"
                conn.setRequestProperty("Content-Type", "application/json")
                conn.doOutput = true

                conn.outputStream.bufferedWriter().use {
                    it.write(json.toString())
                }

                val response =
                    conn.inputStream.bufferedReader().use { it.readText() }

                val answerJson = JSONObject(response)

                val answer = SessionDescription(
                    SessionDescription.Type.fromCanonicalForm(
                        answerJson.getString("type")
                    ),
                    answerJson.getString("sdp")
                )

                peerConnection?.setRemoteDescription(
                    object : SdpObserver {
                        override fun onSetSuccess() {
                            println("[RTC] Connected")
                        }

                        override fun onSetFailure(p0: String?) {}
                        override fun onCreateSuccess(p0: SessionDescription?) {}
                        override fun onCreateFailure(p0: String?) {}
                    },
                    answer
                )

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()
    }

    fun release() {
        videoCapturer?.stopCapture()
        videoCapturer?.dispose()
        videoSource?.dispose()
        peerConnection?.close()
        peerConnectionFactory.dispose()
        eglBase.release()
    }
    private class CustomVideoProcessor : VideoProcessor {

        private var sink: VideoSink? = null

        override fun onCapturerStarted(success: Boolean) {}
        override fun onCapturerStopped() {}

        override fun setSink(videoSink: VideoSink?) {
            sink = videoSink
        }

        override fun onFrameCaptured(frame: VideoFrame) {
            val processed = FramePreprocessor.processFrame(frame)
            sink?.onFrame(processed)
        }
    }
}