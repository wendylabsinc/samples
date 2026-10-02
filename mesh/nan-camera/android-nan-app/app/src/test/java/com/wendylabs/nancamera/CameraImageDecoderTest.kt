package com.wendylabs.nancamera

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.io.IOException
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], manifest = Config.NONE)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class CameraImageDecoderTest {
    private fun jpeg(width: Int, height: Int): ByteArray {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        return ByteArrayOutputStream().use { out ->
            check(bitmap.compress(Bitmap.CompressFormat.JPEG, 70, out))
            bitmap.recycle()
            out.toByteArray()
        }
    }
    @Test fun normalPreviewAndThumbnailDecode() {
        for ((width, height) in listOf(320 to 240, 160 to 120, 1920 to 1080)) {
            val image = CameraImageDecoder.decode(jpeg(width, height))
            assertEquals(width, image.width); assertEquals(height, image.height)
            image.recycle()
        }
    }
    @Test fun compressedSmallButPixelLargeIsRejected() {
        val data = jpeg(2049, 2048)
        assertTrue(data.size < 100_000)
        assertThrows(IOException::class.java) { CameraImageDecoder.decode(data) }
    }
    @Test fun dimensionLimitIsIndependentOfPixelCount() {
        assertThrows(IOException::class.java) { CameraImageDecoder.decode(jpeg(4097, 1)) }
        val allowed = CameraImageDecoder.decode(jpeg(4096, 1))
        assertEquals(4096, allowed.width); allowed.recycle()
    }
    @Test fun exactPixelLimitAcceptedAndInvalidPayloadRejected() {
        val image = CameraImageDecoder.decode(jpeg(2048, 2048))
        assertEquals(2048, image.width); image.recycle()
        for (data in listOf(byteArrayOf(), byteArrayOf(0, 1, 2, 3))) {
            assertThrows(IOException::class.java) { CameraImageDecoder.decode(data) }
        }
    }
    @Test fun nonJpegImageRejected() {
        val bitmap = Bitmap.createBitmap(16, 16, Bitmap.Config.ARGB_8888)
        val data = ByteArrayOutputStream().use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out); out.toByteArray()
        }
        bitmap.recycle()
        assertThrows(IOException::class.java) { CameraImageDecoder.decode(data) }
    }
}
