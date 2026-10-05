package com.wendylabs.nancamera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.IOException

/** Bounds decoded pixels independently of the compressed wire-body limit. */
internal object CameraImageDecoder {
    const val MAX_DIMENSION = 4096
    const val MAX_PIXELS = 4L * 1024 * 1024

    fun decode(bytes: ByteArray): Bitmap {
        if (bytes.isEmpty()) throw IOException("Empty camera JPEG")
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outMimeType != "image/jpeg" || bounds.outWidth <= 0 || bounds.outHeight <= 0 ||
            bounds.outWidth > MAX_DIMENSION || bounds.outHeight > MAX_DIMENSION ||
            bounds.outWidth.toLong() * bounds.outHeight > MAX_PIXELS) {
            throw IOException("Invalid or oversized camera JPEG")
        }
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val image = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
            ?: throw IOException("Invalid camera JPEG")
        if (image.width != bounds.outWidth || image.height != bounds.outHeight) {
            image.recycle()
            throw IOException("Camera JPEG dimensions changed during decode")
        }
        return image
    }
}
