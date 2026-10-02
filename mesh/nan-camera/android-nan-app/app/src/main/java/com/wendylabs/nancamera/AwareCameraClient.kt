package com.wendylabs.nancamera

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.IntentFilter
import android.content.BroadcastReceiver
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.aware.AttachCallback
import android.net.wifi.aware.DiscoverySessionCallback
import android.net.wifi.aware.PeerHandle
import android.net.wifi.aware.ServiceDiscoveryInfo
import android.net.wifi.aware.SubscribeConfig
import android.net.wifi.aware.SubscribeDiscoverySession
import android.net.wifi.aware.WifiAwareManager
import android.net.wifi.aware.WifiAwareNetworkInfo
import android.net.wifi.aware.WifiAwareNetworkSpecifier
import android.net.wifi.aware.WifiAwareSession
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

data class CameraOffer(val peer: PeerHandle, val port: Int, val label: String)

internal fun cameraPortFromServiceInfo(info: ByteArray?): Int? {
    // This service uses a fixed port. Some NAN implementations reject the
    // service-specific-info extension, so the publisher may omit it entirely.
    if (info == null || info.isEmpty() || info.contentEquals(byteArrayOf(1, 0x23, 0x83.toByte(), 0))) {
        return 9091
    }
    return null
}

private sealed interface CameraAction {
    data object ListPhotos : CameraAction
    data object Capture : CameraAction
    data class Thumbnail(val id: UUID) : CameraAction
    data class Delete(val id: UUID) : CameraAction
    data class Download(val id: UUID) : CameraAction
}

@SuppressLint("MissingPermission") // The activity checks NEARBY_WIFI_DEVICES before discovery.
class AwareCameraClient(private val context: Context) {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val aware = context.getSystemService(WifiAwareManager::class.java)
    private val connectivity = context.getSystemService(ConnectivityManager::class.java)
    private var awareSession: WifiAwareSession? = null
    private var subscription: SubscribeDiscoverySession? = null
    private var networkCallback: ConnectivityManager.NetworkCallback? = null
    private val actions = LinkedBlockingQueue<CameraAction>()
    @Volatile private var socket: Socket? = null
    @Volatile private var generation = 0
    private var discoveryGeneration = 0
    private var rateTicket = 0
    private var closed = false

    val offers = mutableStateListOf<CameraOffer>()
    val photos = mutableStateListOf<PhotoRecord>()
    var status by mutableStateOf("Tap Discover to find a camera")
        private set
    var discovering by mutableStateOf(false)
        private set
    var connecting by mutableStateOf(false)
        private set
    var connected by mutableStateOf(false)
        private set
    var busy by mutableStateOf(false)
        private set
    var preview by mutableStateOf<Bitmap?>(null)
        private set
    var thumbnail by mutableStateOf<Bitmap?>(null)
        private set
    var selectedPhoto by mutableStateOf<UUID?>(null)
        private set
    var downloadRate by mutableStateOf<String?>(null)
        private set
    var downloadProgress by mutableStateOf<String?>(null)
        private set

    private val availability = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == WifiAwareManager.ACTION_WIFI_AWARE_STATE_CHANGED && aware?.isAvailable != true) {
                stopDiscovery()
                status = "Wi-Fi Aware became unavailable. Check Wi-Fi and try again."
            }
        }
    }

    init {
        context.registerReceiver(availability, IntentFilter(WifiAwareManager.ACTION_WIFI_AWARE_STATE_CHANGED))
    }

    fun discover() {
        if (closed || discovering) return
        if (!context.packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI_AWARE) || aware == null) {
            status = "This phone does not support Wi-Fi Aware"
            return
        }
        if (!aware.isAvailable) {
            status = "Wi-Fi Aware unavailable. Enable Wi-Fi and Location."
            return
        }
        stopDiscovery()
        val token = discoveryGeneration
        discovering = true
        status = "Joining Wi-Fi Aware and looking for cameras…"
        try {
            aware.attach(object : AttachCallback() {
                override fun onAttached(session: WifiAwareSession) {
                    if (token != discoveryGeneration || closed) { session.close(); return }
                    awareSession = session
                    val config = SubscribeConfig.Builder()
                        .setServiceName("wendy.nan.camera.v1")
                        .setSubscribeType(SubscribeConfig.SUBSCRIBE_TYPE_PASSIVE)
                        .build()
                    session.subscribe(config, object : DiscoverySessionCallback() {
                        override fun onSubscribeStarted(session: SubscribeDiscoverySession) {
                            if (token != discoveryGeneration || closed) { session.close(); return }
                            subscription = session
                            status = "Looking for cameras nearby…"
                        }

                        override fun onServiceDiscovered(peer: PeerHandle, info: ByteArray?, filters: MutableList<ByteArray>?) {
                            traceDiscovery("onServiceDiscovered", peer, info, filters, token)
                            recordOffer(peer, info, token)
                        }

                        override fun onServiceDiscovered(info: ServiceDiscoveryInfo) {
                            traceDiscovery("onServiceDiscovered(info)", info.peerHandle, info.serviceSpecificInfo, info.matchFilters, token)
                            recordOffer(info.peerHandle, info.serviceSpecificInfo, token)
                        }

                        override fun onServiceDiscoveredWithinRange(
                            peer: PeerHandle, info: ByteArray?, filters: MutableList<ByteArray>?, distanceMm: Int
                        ) {
                            traceDiscovery("onServiceDiscoveredWithinRange($distanceMm)", peer, info, filters, token)
                            recordOffer(peer, info, token)
                        }

                        override fun onServiceDiscoveredWithinRange(info: ServiceDiscoveryInfo, distanceMm: Int) {
                            traceDiscovery("onServiceDiscoveredWithinRange(info,$distanceMm)", info.peerHandle, info.serviceSpecificInfo, info.matchFilters, token)
                            recordOffer(info.peerHandle, info.serviceSpecificInfo, token)
                        }

                        private fun recordOffer(peer: PeerHandle, info: ByteArray?, token: Int) {
                            if (token != discoveryGeneration || closed) return
                            val port = cameraPortFromServiceInfo(info) ?: return
                            if (offers.none { it.peer == peer }) {
                                offers.add(CameraOffer(peer, port, "Camera ${peer.hashCode().toUInt().toString(16)}"))
                            }
                            if (!connected && !connecting) status = "${offers.size} camera(s) nearby"
                        }

                        private fun traceDiscovery(
                            callback: String, peer: PeerHandle, info: ByteArray?, filters: List<ByteArray>?, token: Int
                        ) {
                            if (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
                            Log.i("WendyNANCamera", "$callback peer=${peer.hashCode()} ssiLen=${info?.size ?: -1} " +
                                "ssi=${info?.joinToString("") { "%02x".format(it.toInt() and 0xff) } ?: "null"} " +
                                "filters=${filters?.size ?: -1} generation=$token/$discoveryGeneration")
                        }

                        override fun onServiceLost(peer: PeerHandle, reason: Int) {
                            if (token != discoveryGeneration || closed) return
                            offers.removeAll { it.peer == peer }
                            if (!connected && !connecting) status = "${offers.size} camera(s) nearby"
                        }

                        override fun onSessionConfigFailed() {
                            if (token == discoveryGeneration) failDiscovery("Camera discovery failed")
                        }

                        override fun onSessionTerminated() {
                            if (token == discoveryGeneration) failDiscovery("Camera discovery ended")
                        }
                    }, main)
                }

                override fun onAttachFailed() {
                    if (token == discoveryGeneration) failDiscovery("Wi-Fi Aware attach failed")
                }
            }, main)
        } catch (e: Exception) { failDiscovery("Discovery: ${e.message}") }
    }

    fun permissionDenied() { status = "Nearby devices permission is required" }

    private fun failDiscovery(message: String) {
        stopDiscovery()
        status = message
    }

    fun connect(offer: CameraOffer) {
        val sub = subscription ?: run { status = "Discover a camera first"; return }
        if (closed) return
        disconnect()
        val token = generation
        var launched = false
        connecting = true
        status = "Opening NAN data path to ${offer.label}…"
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(network: Network, capabilities: NetworkCapabilities) {
                val address = (capabilities.transportInfo as? WifiAwareNetworkInfo)?.peerIpv6Addr ?: return
                main.post {
                    if (token != generation || networkCallback !== this || launched) return@post
                    launched = true
                    worker.execute { runCamera(network, InetSocketAddress(address, offer.port), token) }
                }
            }

            override fun onUnavailable() {
                main.post {
                    if (token == generation && networkCallback === this) {
                        // A replaced publisher may leave its old PeerHandle in
                        // discovery until Android reports service loss. Do not
                        // keep offering a handle this request could not reach.
                        // A later discovery callback may legitimately add it again.
                        offers.removeAll { it.peer == offer.peer }
                        linkFailed("Camera unavailable. Choose another camera or stop the scan and discover again.")
                    }
                }
            }

            override fun onLost(network: Network) {
                main.post {
                    if (token == generation && networkCallback === this) linkFailed("NAN data path lost")
                }
            }
        }
        networkCallback = callback
        try {
            // An unset security config requests an open NDP. The camera uses a fixed TCP port.
            val specifier = WifiAwareNetworkSpecifier.Builder(sub, offer.peer).build()
            val request = NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_WIFI_AWARE)
                .removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .setNetworkSpecifier(specifier).build()
            connectivity.requestNetwork(request, callback, 30_000)
        } catch (e: Exception) { linkFailed("NAN connection: ${e.message}") }
    }

    private fun runCamera(network: Network, address: InetSocketAddress, token: Int) {
        try {
            openInitialCameraSocket(
                socketFactory = { network.socketFactory.createSocket() },
                address = address,
                active = { token == generation && !closed },
                onAttemptSocket = { socket = it },
            ).use { tcp ->
                tcp.tcpNoDelay = true
                tcp.soTimeout = 15_000
                val wire = CameraWire(tcp.getInputStream(), tcp.getOutputStream())
                main.post {
                    if (token == generation) {
                        connecting = false
                        connected = true
                        status = "Connected to camera"
                    }
                }
                actions.offer(CameraAction.ListPhotos)
                while (token == generation && !closed) {
                    try {
                        when (val action = actions.poll(20, TimeUnit.MILLISECONDS)) {
                            null -> {
                                val image = decode(wire.snapshot())
                                main.post { if (token == generation) preview = image }
                            }
                            CameraAction.ListPhotos -> {
                                val list = wire.list()
                                main.post {
                                    if (token == generation) { photos.clear(); photos.addAll(list) }
                                }
                            }
                            CameraAction.Capture -> {
                                wire.capture()
                                main.post { if (token == generation) status = "Photo saved on camera" }
                                actions.offer(CameraAction.ListPhotos)
                            }
                            is CameraAction.Thumbnail -> {
                                val image = decode(wire.thumbnail(action.id))
                                main.post { if (token == generation && selectedPhoto == action.id) thumbnail = image }
                            }
                            is CameraAction.Delete -> {
                                wire.delete(action.id)
                                main.post {
                                    if (token == generation) {
                                        status = "Photo deleted"
                                        if (selectedPhoto == action.id) { selectedPhoto = null; thumbnail = null }
                                    }
                                }
                                actions.offer(CameraAction.ListPhotos)
                            }
                            is CameraAction.Download -> saveDownload(wire, action.id, token)
                        }
                    } catch (e: CameraOperationException) {
                        main.post { if (token == generation) status = e.message ?: "Camera operation failed" }
                        // For example, an unplugged webcam may reject each live
                        // snapshot. Keep the NDP while bounding retry traffic.
                        Thread.sleep(250)
                    }
                }
            }
        } catch (e: Exception) {
            main.post { if (token == generation) linkFailed("Camera connection: ${e.message ?: e.javaClass.simpleName}") }
        } finally {
            socket = null
        }
    }

    private fun decode(bytes: ByteArray): Bitmap {
        return CameraImageDecoder.decode(bytes)
    }

    private fun saveDownload(wire: CameraWire, id: UUID, token: Int) {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "wendy-nan-$id.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Wendy NAN Camera")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IOException("Cannot create a photo in MediaStore")
        var sampleElapsed = 0L
        var sampleBytes = 0L
        var saved = false
        try {
            val stream = resolver.openOutputStream(uri) ?: throw IOException("Cannot open photo destination")
            stream.use { output ->
                val expectedSize = photos.firstOrNull { it.id == id }?.jpegBytes
                    ?: throw IOException("Photo is no longer in the camera catalog")
                val download = wire.downloadInMemory(id, expectedSize) { bytes, total, elapsed ->
                    if (elapsed - sampleElapsed >= 200_000_000 || bytes == total) {
                        val rate = (bytes - sampleBytes) * 1e9 / (elapsed - sampleElapsed).coerceAtLeast(1L) / 1024
                        main.post {
                            if (token == generation) {
                                downloadRate = String.format("%.1f KiB/s", rate)
                                downloadProgress = "$bytes / $total bytes"
                            }
                        }
                        sampleElapsed = elapsed
                        sampleBytes = bytes
                    }
                }
                val count = download.jpeg.size
                val average = count * 1e9 / download.elapsedNanos / 1024
                // The measurement has ended. Android storage work is outside it.
                output.write(download.jpeg)
                output.flush()
                Log.i("WendyNANCamera", "downloadReceived id=$id bytes=$count elapsedNanos=${download.elapsedNanos} window=request-flush-to-RAM")
                main.post {
                    if (token == generation) {
                        downloadRate = String.format("%.1f KiB/s average", average)
                        downloadProgress = "$count bytes saved"
                        status = "Downloaded photo to Pictures/Wendy NAN Camera"
                        val ticket = ++rateTicket
                        main.postDelayed({
                            if (ticket == rateTicket && token == generation) {
                                downloadRate = null
                                downloadProgress = null
                            }
                        }, 10_000)
                    }
                }
            }
            resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            saved = true
        } finally {
            if (!saved) resolver.delete(uri, null, null)
            main.post {
                if (token == generation) {
                    busy = false
                    if (!saved) {
                        downloadRate = null
                        downloadProgress = null
                    }
                }
            }
        }
    }

    fun capture() { if (connected) actions.offer(CameraAction.Capture) }
    fun refresh() { if (connected) actions.offer(CameraAction.ListPhotos) }
    fun select(photo: PhotoRecord) {
        selectedPhoto = photo.id
        thumbnail = null
        if (connected) actions.offer(CameraAction.Thumbnail(photo.id))
    }
    fun deleteSelected() { selectedPhoto?.let { if (connected) actions.offer(CameraAction.Delete(it)) } }
    fun downloadSelected() {
        val id = selectedPhoto ?: return
        if (connected && !busy) {
            busy = true
            downloadRate = "Starting download…"
            downloadProgress = null
            rateTicket++
            actions.offer(CameraAction.Download(id))
        }
    }

    private fun linkFailed(message: String) {
        disconnect()
        status = message
    }

    fun disconnect() {
        generation++
        socket?.close()
        socket = null
        networkCallback?.let { runCatching { connectivity.unregisterNetworkCallback(it) } }
        networkCallback = null
        actions.clear()
        connecting = false
        connected = false
        busy = false
        preview = null
        photos.clear()
        selectedPhoto = null
        thumbnail = null
        downloadRate = null
        downloadProgress = null
        rateTicket++
    }

    fun stopDiscovery() {
        discoveryGeneration++
        disconnect()
        subscription?.close(); subscription = null
        awareSession?.close(); awareSession = null
        offers.clear()
        discovering = false
    }

    fun close() {
        if (closed) return
        closed = true
        stopDiscovery()
        context.unregisterReceiver(availability)
        worker.shutdownNow()
    }
}
