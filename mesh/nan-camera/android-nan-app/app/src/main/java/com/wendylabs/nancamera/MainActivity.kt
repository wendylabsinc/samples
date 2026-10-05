package com.wendylabs.nancamera

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import java.text.DateFormat
import java.util.Date

private val Ink = Color(0xFF080D12)
private val Panel = Color(0xFF141D26)
private val Line = Color(0xFF293744)
private val Cyan = Color(0xFF79E4E0)
private val Muted = Color(0xFF9DAEBC)
private val Lime = Color(0xFFD6F58A)
private val CameraColors = darkColorScheme(
    primary = Cyan, onPrimary = Color(0xFF062E30), primaryContainer = Color(0xFF183A3D),
    secondary = Lime, background = Ink, onBackground = Color(0xFFF0F5F8),
    surface = Panel, onSurface = Color(0xFFF0F5F8), onSurfaceVariant = Muted,
    outline = Line, error = Color(0xFFFFB4AB), surfaceContainer = Panel,
)

/** The radio session outlives activity recreation, but closes when this screen finishes. */
class CameraViewModel(application: Application) : AndroidViewModel(application) {
    val camera = AwareCameraClient(application.applicationContext)
    override fun onCleared() {
        camera.close()
        super.onCleared()
    }
}

class MainActivity : ComponentActivity() {
    private lateinit var camera: AwareCameraClient
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) camera.discover() else camera.permissionDenied()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
        camera = ViewModelProvider(this)[CameraViewModel::class.java].camera
        setContent {
            MaterialTheme(colorScheme = CameraColors) {
                Surface(modifier = Modifier.fillMaxSize(), color = Ink) {
                    CameraScreen(camera, ::startDiscovery)
                }
            }
        }
    }

    private fun startDiscovery() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.NEARBY_WIFI_DEVICES) == PackageManager.PERMISSION_GRANTED) {
            camera.discover()
        } else permission.launch(Manifest.permission.NEARBY_WIFI_DEVICES)
    }

}

@Composable
private fun CameraScreen(camera: AwareCameraClient, discover: () -> Unit) {
    var deletePhoto by remember { mutableStateOf<java.util.UUID?>(null) }
    var cameraName by rememberSaveable { mutableStateOf("Camera") }
    BoxWithConstraints(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing)) {
        val wide = maxWidth >= 650.dp
        Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = if (wide) 6.dp else 12.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.size(if (wide) 34.dp else 42.dp).clip(RoundedCornerShape(13.dp)).background(Cyan), contentAlignment = Alignment.Center) {
                Glyph("camera", Color(0xFF062E30), Modifier.size(25.dp))
            }
            Column(Modifier.weight(1f)) {
                Text("Wendy", fontSize = if (wide) 19.sp else 23.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.5).sp)
                Text("NAN CAMERA", fontSize = 10.sp, letterSpacing = 2.sp, color = Muted, fontWeight = FontWeight.SemiBold)
            }
            ConnectionBadge(camera)
        }
        HorizontalDivider(color = Line.copy(alpha = .6f))
        BoxWithConstraints(Modifier.weight(1f)) {
            if (wide) {
                // Reserve the title, capture control and spacing; letterbox the
                // complete image into the remaining height. Scroll remains an
                // accessibility fallback when enlarged fonts need more room.
                val previewHeight = (maxHeight - 120.dp).coerceAtLeast(64.dp)
                Row(Modifier.fillMaxSize().padding(horizontal = 20.dp), horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                    Column(Modifier.weight(1.1f).verticalScroll(rememberScrollState()).padding(vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        LiveView(camera, cameraName, previewHeight)
                        CaptureButton(camera, compact = true)
                    }
                    Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        ConnectionSection(camera, discover) { cameraName = it.label; camera.connect(it) }
                        if (camera.connected) Gallery(camera) { deletePhoto = camera.selectedPhoto }
                    }
                }
            } else {
                Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
                    LiveView(camera, cameraName)
                    if (camera.connected) CaptureButton(camera)
                    ConnectionSection(camera, discover) { cameraName = it.label; camera.connect(it) }
                    if (camera.connected) Gallery(camera) { deletePhoto = camera.selectedPhoto }
                }
            }
        }
        // Keep current and recent transfer speed visible even while browsing a long gallery.
        if (camera.downloadRate != null) TransferPanel(camera, compact = wide)
        }
    }
    if (deletePhoto != null) {
        AlertDialog(onDismissRequest = { deletePhoto = null },
            title = { Text("Delete this photo?") },
            text = { Text("Remove the original from the camera. Any copy already downloaded to this phone will be kept.") },
            confirmButton = {
                TextButton(onClick = {
                    // A disconnect or selection change must not delete a different photo.
                    if (camera.connected && camera.selectedPhoto == deletePhoto) camera.deleteSelected()
                    deletePhoto = null
                }) { Text("Delete photo", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { deletePhoto = null }) { Text("Keep photo") } })
    }
}

@Composable
private fun ConnectionBadge(camera: AwareCameraClient) {
    val label = when { camera.connected -> "CONNECTED"; camera.connecting -> "CONNECTING"; camera.discovering -> "SEARCHING"; else -> "OFFLINE" }
    val color = if (camera.connected) Lime else if (camera.connecting || camera.discovering) Cyan else Muted
    Row(Modifier.clip(CircleShape).background(color.copy(alpha = .09f)).border(1.dp, color.copy(alpha = .22f), CircleShape).padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Box(Modifier.size(5.dp).background(color, CircleShape))
        Text(label, fontSize = 9.sp, fontWeight = FontWeight.Bold, color = color, letterSpacing = .7.sp)
    }
}

@Composable
private fun LiveView(camera: AwareCameraClient, name: String, previewHeight: Dp? = null) {
    val compact = previewHeight != null
    Column(verticalArrangement = Arrangement.spacedBy(if (compact) 8.dp else 10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Live view", style = if (compact) MaterialTheme.typography.titleMedium else MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            Text(if (camera.connected) name else "DIRECT · WI-FI AWARE", color = Muted, fontSize = 10.sp, letterSpacing = .8.sp)
        }
        val frameSize = if (previewHeight != null) Modifier.height(previewHeight) else Modifier.aspectRatio(4f / 3f)
        Box(Modifier.fillMaxWidth().then(frameSize).clip(RoundedCornerShape(22.dp)).background(Color(0xFF0C141C)).border(1.dp, Line, RoundedCornerShape(22.dp)),
            contentAlignment = Alignment.Center) {
            val frame = camera.preview
            if (frame != null && camera.connected) {
                Image(frame.asImageBitmap(), "Live camera preview", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                Row(Modifier.align(Alignment.TopStart).padding(14.dp).clip(CircleShape).background(Color.Black.copy(alpha = .65f)).padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Box(Modifier.size(6.dp).background(if (camera.busy) Cyan else Lime, CircleShape))
                    Text(if (camera.busy) "PAUSED FOR TRANSFER" else "LIVE", fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                }
                Text("${frame.width} × ${frame.height}", Modifier.align(Alignment.BottomEnd).padding(14.dp).clip(CircleShape).background(Color.Black.copy(alpha = .6f)).padding(horizontal = 10.dp, vertical = 5.dp), fontSize = 10.sp, color = Color.White)
            } else if (compact) {
                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    if (camera.connecting || camera.connected) CircularProgressIndicator(Modifier.size(20.dp), color = Cyan, strokeWidth = 2.dp)
                    else Glyph("camera", Muted)
                    Text(if (camera.connecting) "Opening camera…" else if (camera.connected) "Waiting for first frame…" else "Connect a nearby camera", fontSize = 12.sp, color = Muted)
                }
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(24.dp)) {
                    Box(Modifier.size(64.dp).background(Cyan.copy(alpha = .07f), CircleShape), contentAlignment = Alignment.Center) {
                        if (camera.connecting || camera.connected) CircularProgressIndicator(Modifier.size(28.dp), color = Cyan, strokeWidth = 2.dp)
                        else Glyph("camera", Muted, Modifier.size(30.dp))
                    }
                    Text(if (camera.connecting) "Opening camera…" else if (camera.connected) "Waiting for first frame…" else "Your next point of view", fontWeight = FontWeight.Medium)
                    Text(if (camera.connected || camera.connecting) "Establishing a direct camera connection" else "Find a nearby camera to start a live view.", color = Muted, fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun CaptureButton(camera: AwareCameraClient, compact: Boolean = false) {
    Button(onClick = camera::capture, enabled = camera.connected && !camera.busy,
        modifier = Modifier.fillMaxWidth().heightIn(min = if (compact) 48.dp else 60.dp), shape = RoundedCornerShape(18.dp),
        colors = ButtonDefaults.buttonColors(containerColor = Cyan)) {
        Glyph("shutter", MaterialTheme.colorScheme.onPrimary, Modifier.size(27.dp))
        Spacer(Modifier.width(12.dp))
        Column {
            Text("Take photo", fontSize = 17.sp, fontWeight = FontWeight.Bold)
            if (!compact) Text("Save the full-resolution original", fontSize = 11.sp)
        }
    }
}

@Composable
private fun ConnectionSection(camera: AwareCameraClient, discover: () -> Unit, connect: (CameraOffer) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(if (camera.connected) "Camera connection" else "Nearby cameras", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            if (camera.connected || camera.connecting) TextButton(onClick = camera::disconnect) { Text(if (camera.connecting) "Cancel" else "Disconnect") }
            else if (camera.discovering) TextButton(onClick = camera::stopDiscovery) { Text("Stop scan") }
            else TextButton(onClick = discover) { Glyph("search", Cyan); Spacer(Modifier.width(6.dp)); Text("Discover") }
        }
        Text(camera.status, color = Muted, style = MaterialTheme.typography.bodySmall)
        if (!camera.connected && !camera.connecting) {
            if (camera.offers.isEmpty()) {
                if (camera.discovering) LinearProgressIndicator(Modifier.fillMaxWidth().height(2.dp), color = Cyan, trackColor = Line)
                else OutlinedButton(onClick = discover, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp)) {
                    Text("Find a camera")
                }
            }
            camera.offers.forEach { offer ->
                Surface(onClick = { connect(offer) }, shape = RoundedCornerShape(16.dp), color = Panel, modifier = Modifier.fillMaxWidth()) {
                    Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Glyph("camera", Cyan)
                        Column(Modifier.weight(1f)) {
                            Text(offer.label, fontWeight = FontWeight.SemiBold)
                            Text("Nearby · ready to connect", fontSize = 11.sp, color = Muted)
                        }
                        Text("Connect", color = Cyan, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                        Glyph("arrow", Cyan, Modifier.size(16.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun Gallery(camera: AwareCameraClient, delete: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        HorizontalDivider(color = Line)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("On the camera", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            Text("${camera.photos.size}", color = Muted, modifier = Modifier.padding(end = 4.dp))
            IconButton(onClick = camera::refresh, enabled = !camera.busy, modifier = Modifier.semantics { contentDescription = "Refresh saved photos" }) { Glyph("refresh", Cyan) }
        }
        val selected = camera.photos.firstOrNull { it.id == camera.selectedPhoto }
        if (selected != null) {
            Surface(shape = RoundedCornerShape(18.dp), color = Panel, modifier = Modifier.border(1.dp, Cyan.copy(alpha = .45f), RoundedCornerShape(18.dp))) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(104.dp, 78.dp).clip(RoundedCornerShape(10.dp)).background(Ink), contentAlignment = Alignment.Center) {
                            camera.thumbnail?.let { Image(it.asImageBitmap(), "Selected saved photo thumbnail", Modifier.fillMaxSize(), contentScale = ContentScale.Fit) }
                                ?: CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                        }
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text("SELECTED PHOTO", color = Cyan, fontSize = 9.sp, letterSpacing = 1.sp, fontWeight = FontWeight.Bold)
                            Text(photoDate(selected), style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                            Text("${selected.jpegBytes / 1024} KiB · JPEG original", fontSize = 11.sp, color = Muted)
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        Button(onClick = camera::downloadSelected, enabled = !camera.busy, modifier = Modifier.weight(1f), shape = RoundedCornerShape(12.dp)) {
                            Glyph("download", MaterialTheme.colorScheme.onPrimary, Modifier.size(18.dp)); Spacer(Modifier.width(8.dp)); Text("Download JPEG")
                        }
                        OutlinedButton(onClick = delete, enabled = !camera.busy, contentPadding = PaddingValues(horizontal = 12.dp), shape = RoundedCornerShape(12.dp)) {
                            Glyph("trash", MaterialTheme.colorScheme.error, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("Delete", color = MaterialTheme.colorScheme.error)
                        }
                    }
                }
            }
        }
        if (camera.photos.isEmpty()) {
            Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Panel).padding(20.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Make the first frame", fontWeight = FontWeight.Medium)
                Text("Take a photo and it will appear here. Originals stay on the camera until you delete them.", color = Muted, fontSize = 12.sp)
            }
        }
        camera.photos.forEach { photo ->
            val isSelected = photo.id == camera.selectedPhoto
            Surface(onClick = { camera.select(photo) }, enabled = !camera.busy, shape = RoundedCornerShape(14.dp), color = if (isSelected) Cyan.copy(alpha = .09f) else Panel, modifier = Modifier.fillMaxWidth()) {
                Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Glyph("photo", if (isSelected) Cyan else Muted)
                    Column(Modifier.weight(1f)) {
                        Text(photoDate(photo), fontSize = 13.sp, fontWeight = FontWeight.Medium)
                        Text("${photo.jpegBytes / 1024} KiB · ${photo.id.toString().take(8)}", color = Muted, fontSize = 11.sp)
                    }
                    Text(if (isSelected) "Selected" else "View", color = if (isSelected) Cyan else Muted, fontSize = 11.sp)
                }
            }
        }
    }
}

private fun photoDate(photo: PhotoRecord): String = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(photo.capturedAtUnixMs))

@Composable
private fun TransferPanel(camera: AwareCameraClient, compact: Boolean = false) {
    val rate = camera.downloadRate ?: return
    val active = camera.busy
    Surface(color = Color(0xFF122A2D), tonalElevation = 0.dp) {
        Column(Modifier.fillMaxWidth()) {
            HorizontalDivider(color = Cyan.copy(alpha = .3f))
            Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = if (compact) 6.dp else 12.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Glyph("download", Cyan, Modifier.size(22.dp))
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Text(if (active) "DOWNLOADING ORIGINAL" else "SAVED TO YOUR PHONE", color = Cyan, fontSize = 9.sp, letterSpacing = 1.sp, fontWeight = FontWeight.SemiBold)
                    if (!compact) {
                        Text(rate, fontSize = 19.sp, fontWeight = FontWeight.Bold)
                    }
                    camera.downloadProgress?.let { Text(it, color = Muted, fontSize = 10.sp) }
                }
                if (compact) Text(rate, fontSize = 17.sp, fontWeight = FontWeight.Bold)
                if (active) CircularProgressIndicator(Modifier.size(20.dp), color = Cyan, strokeWidth = 2.dp)
                else Text("✓", color = Lime, fontSize = 22.sp)
            }
        }
    }
}

/** Small code-native line icons; their controls provide the accessible labels. */
@Composable
private fun Glyph(kind: String, color: Color, modifier: Modifier = Modifier.size(22.dp)) {
    Canvas(modifier) {
        val u = size.minDimension / 24f
        val stroke = 1.7f * u
        fun line(x1: Float, y1: Float, x2: Float, y2: Float) = drawLine(color, Offset(x1*u,y1*u), Offset(x2*u,y2*u), stroke, StrokeCap.Round)
        fun circle(x: Float, y: Float, r: Float) = drawCircle(color, r*u, Offset(x*u,y*u), style = Stroke(stroke))
        when (kind) {
            "camera" -> { drawRoundRect(color, Offset(2*u,6*u), Size(20*u,15*u), androidx.compose.ui.geometry.CornerRadius(3*u), style = Stroke(stroke)); circle(12f,13f,4f); line(7f,6f,9f,3f); line(9f,3f,15f,3f); line(15f,3f,17f,6f) }
            "shutter" -> { circle(12f,12f,10f); drawCircle(color, 6*u, Offset(12*u,12*u)) }
            "download" -> { line(12f,2f,12f,15f); line(7f,10f,12f,15f); line(17f,10f,12f,15f); line(3f,16f,3f,21f); line(3f,21f,21f,21f); line(21f,21f,21f,16f) }
            "photo" -> { drawRoundRect(color, Offset(2*u,3*u), Size(20*u,18*u), androidx.compose.ui.geometry.CornerRadius(2*u), style = Stroke(stroke)); circle(8f,8f,2f); line(3f,19f,10f,12f); line(10f,12f,14f,16f); line(14f,16f,18f,12f); line(18f,12f,21f,15f) }
            "trash" -> { line(3f,6f,21f,6f); line(9f,3f,15f,3f); line(5f,6f,6f,21f); line(6f,21f,18f,21f); line(18f,21f,19f,6f); line(10f,10f,10f,17f); line(14f,10f,14f,17f) }
            "search" -> { circle(10f,10f,7f); line(15f,15f,22f,22f) }
            "arrow" -> { line(4f,12f,20f,12f); line(14f,6f,20f,12f); line(14f,18f,20f,12f) }
            "refresh" -> { drawArc(color, 40f, 290f, false, Offset(4*u,4*u), Size(16*u,16*u), style = Stroke(stroke, cap = StrokeCap.Round)); line(20f,3f,20f,9f); line(20f,9f,14f,9f) }
        }
    }
}
