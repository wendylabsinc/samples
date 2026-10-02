package com.wendylabs.nancamera

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.util.UUID

data class PhotoRecord(val id: UUID, val capturedAtUnixMs: Long, val jpegBytes: Long)
data class CameraDownload(val jpeg: ByteArray, val elapsedNanos: Long)

/** A complete error frame leaves the connection aligned for the next request. */
class CameraOperationException(message: String) : IOException(message)

/** The framing in ../PROTOCOL.md. One owner serializes requests on each TCP connection. */
class CameraWire(input: InputStream, output: OutputStream) {
    private val fromPeer = DataInputStream(input.buffered(64 * 1024))
    private val toPeer = DataOutputStream(output.buffered())
    private var nextId = 1

    companion object {
        const val MAX_PAYLOAD = 32 * 1024 * 1024
        private const val MAGIC = 0x574e4341 // WNCA
    }

    private fun begin(op: Int, payload: ByteArray): Int {
        require(payload.size <= MAX_PAYLOAD)
        val id = nextId++
        toPeer.writeInt(MAGIC)
        toPeer.writeByte(1)
        toPeer.writeByte(op)
        toPeer.writeInt(id)
        toPeer.writeInt(payload.size)
        toPeer.write(payload)
        toPeer.flush()
        return id
    }

    private fun reply(op: Int, id: Int): Int {
        if (fromPeer.readInt() != MAGIC) throw IOException("Bad camera reply magic")
        if (fromPeer.readUnsignedByte() != 1) throw IOException("Unsupported camera protocol version")
        val replyOp = fromPeer.readUnsignedByte()
        if (fromPeer.readInt() != id) throw IOException("Camera reply ID mismatch")
        val size = fromPeer.readInt()
        if (size < 0 || size > MAX_PAYLOAD) throw IOException("Camera reply is too large")
        if (replyOp == 0x7f) {
            if (size > 4096) throw IOException("Camera error is too large")
            val message = ByteArray(size)
            fromPeer.readFully(message)
            throw CameraOperationException("Camera: ${message.toString(Charsets.UTF_8)}")
        }
        if (replyOp != (op or 0x80)) throw IOException("Unexpected camera reply operation")
        return size
    }

    private fun exchange(op: Int, payload: ByteArray = byteArrayOf()): ByteArray {
        val id = begin(op, payload)
        val size = reply(op, id)
        return ByteArray(size).also(fromPeer::readFully)
    }

    fun snapshot(): ByteArray = exchange(0x01)

    fun capture(): PhotoRecord = parseRecord(exchange(0x02))

    fun list(): List<PhotoRecord> {
        val bytes = exchange(0x03)
        if (bytes.size < 2) throw IOException("Truncated camera list")
        val count = ((bytes[0].toInt() and 0xff) shl 8) or (bytes[1].toInt() and 0xff)
        if (bytes.size != 2 + count * 32) throw IOException("Bad camera list size")
        return (0 until count).map { parseRecord(bytes.copyOfRange(2 + it * 32, 2 + (it + 1) * 32)) }
    }

    fun thumbnail(id: UUID): ByteArray = exchange(0x04, id.bytes())

    fun delete(id: UUID) {
        if (exchange(0x06, id.bytes()).isNotEmpty()) throw IOException("Unexpected delete reply body")
    }

    /** One bounded buffer is allocated before sending. No destination I/O is timed.
     * The serial owner has consumed the previous reply before entering this method.
     * Start after the request flush returns; stop immediately after the final RAM read.
     * A socket flush hands bytes to TCP, rather than proving an on-air transmit time.
     */
    fun downloadInMemory(
        id: UUID,
        expectedSize: Long,
        nanoTime: () -> Long = System::nanoTime,
        progress: (Long, Long, Long) -> Unit = { _, _, _ -> },
    ): CameraDownload {
        if (expectedSize !in 0..MAX_PAYLOAD.toLong()) throw IOException("Camera JPEG is too large")
        val jpeg = ByteArray(expectedSize.toInt())
        val requestId = begin(0x05, id.bytes())
        val started = nanoTime()
        val size = reply(0x05, requestId)
        if (size != jpeg.size) throw IOException("Camera JPEG size differs from its catalog record")
        var received = 0
        var finished = started
        while (received < size) {
            val amount = fromPeer.read(jpeg, received, minOf(size - received, 32 * 1024))
            if (amount < 0) throw IOException("Truncated camera download")
            received += amount
            finished = nanoTime()
            progress(received.toLong(), size.toLong(), (finished - started).coerceAtLeast(1L))
        }
        if (size == 0) finished = nanoTime()
        return CameraDownload(jpeg, (finished - started).coerceAtLeast(1L))
    }

    private fun parseRecord(bytes: ByteArray): PhotoRecord {
        if (bytes.size != 32) throw IOException("Bad camera photo record")
        val input = DataInputStream(bytes.inputStream())
        val record = PhotoRecord(UUID(input.readLong(), input.readLong()), input.readLong(), input.readLong())
        if (record.jpegBytes < 0) throw IOException("Invalid camera JPEG size")
        return record
    }

    private fun UUID.bytes(): ByteArray = java.io.ByteArrayOutputStream(16).also {
        DataOutputStream(it).apply { writeLong(mostSignificantBits); writeLong(leastSignificantBits) }
    }.toByteArray()
}
