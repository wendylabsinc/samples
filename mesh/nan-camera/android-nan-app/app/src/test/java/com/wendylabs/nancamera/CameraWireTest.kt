package com.wendylabs.nancamera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.util.UUID

class CameraWireTest {
    private fun reply(op: Int, id: Int = 1, payload: ByteArray = byteArrayOf(), length: Int = payload.size): ByteArray {
        return ByteArrayOutputStream().also { output ->
            DataOutputStream(output).apply {
                writeInt(0x574e4341)
                writeByte(1)
                writeByte(op)
                writeInt(id)
                writeInt(length)
                write(payload)
            }
        }.toByteArray()
    }

    @Test fun snapshotUsesExactHeader() {
        val output = ByteArrayOutputStream()
        val body = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 0xff.toByte(), 0xd9.toByte())
        val wire = CameraWire(ByteArrayInputStream(reply(0x81, payload = body)), output)
        assertArrayEquals(body, wire.snapshot())
        DataInputStream(output.toByteArray().inputStream()).apply {
            assertEquals(0x574e4341, readInt())
            assertEquals(1, readUnsignedByte())
            assertEquals(1, readUnsignedByte())
            assertEquals(1, readInt())
            assertEquals(0, readInt())
            assertEquals(-1, read())
        }
    }

    @Test fun rejectsOversizedReplyBeforeAllocation() {
        val wire = CameraWire(ByteArrayInputStream(reply(0x81, length = CameraWire.MAX_PAYLOAD + 1)), ByteArrayOutputStream())
        assertThrows(IOException::class.java) { wire.snapshot() }
    }

    @Test fun verifiesRequestIdAndError() {
        assertThrows(IOException::class.java) {
            CameraWire(ByteArrayInputStream(reply(0x81, id = 2)), ByteArrayOutputStream()).snapshot()
        }
        val error = assertThrows(IOException::class.java) {
            CameraWire(ByteArrayInputStream(reply(0x7f, payload = "No camera".toByteArray())), ByteArrayOutputStream()).snapshot()
        }
        assertEquals("Camera: No camera", error.message)
    }

    @Test fun completeOperationErrorPreservesNextReplyAlignment() {
        val image = byteArrayOf(1, 2, 3)
        val incoming = reply(0x7f, payload = "Webcam unavailable".toByteArray()) + reply(0x81, id = 2, payload = image)
        val wire = CameraWire(ByteArrayInputStream(incoming), ByteArrayOutputStream())
        val error = assertThrows(CameraOperationException::class.java) { wire.snapshot() }
        assertEquals("Camera: Webcam unavailable", error.message)
        assertArrayEquals(image, wire.snapshot())
    }

    @Test fun incompleteErrorIsTransportFailure() {
        val wire = CameraWire(ByteArrayInputStream(reply(0x7f, payload = byteArrayOf(1), length = 2)), ByteArrayOutputStream())
        val error = assertThrows(IOException::class.java) { wire.snapshot() }
        assertEquals(false, error is CameraOperationException)
    }

    @Test fun parsesListRecordsAndReceivesDownloadInMemory() {
        val id = UUID.fromString("11223344-5566-7788-99aa-bbccddeeff00")
        val record = ByteArrayOutputStream().also { output ->
            DataOutputStream(output).apply {
                writeLong(id.mostSignificantBits); writeLong(id.leastSignificantBits)
                writeLong(1_700_000_000_000); writeLong(80_000)
            }
        }.toByteArray()
        val listBody = byteArrayOf(0, 1) + record
        val image = ByteArray(80_000) { (it and 255).toByte() }
        val input = ByteArrayInputStream(reply(0x83, payload = listBody) + reply(0x85, id = 2, payload = image))
        val outgoing = ByteArrayOutputStream()
        val wire = CameraWire(input, outgoing)
        assertEquals(listOf(PhotoRecord(id, 1_700_000_000_000, 80_000)), wire.list())
        var lastProgress = 0L
        val download = wire.downloadInMemory(id, image.size.toLong()) { bytes, total, elapsed ->
            assertEquals(image.size.toLong(), total)
            assertEquals(true, elapsed > 0)
            lastProgress = bytes
        }
        assertEquals(image.size.toLong(), lastProgress)
        assertArrayEquals(image, download.jpeg)
        DataInputStream(outgoing.toByteArray().inputStream()).apply {
            assertEquals(0x574e4341, readInt()); assertEquals(1, readUnsignedByte())
            assertEquals(3, readUnsignedByte()); assertEquals(1, readInt()); assertEquals(0, readInt())
            assertEquals(0x574e4341, readInt()); assertEquals(1, readUnsignedByte())
            assertEquals(5, readUnsignedByte()); assertEquals(2, readInt()); assertEquals(16, readInt())
            assertArrayEquals(record.copyOfRange(0, 16), ByteArray(16).also { readFully(it) })
        }
    }

    @Test fun malformedListAndTruncatedDownloadFail() {
        assertThrows(IOException::class.java) {
            CameraWire(ByteArrayInputStream(reply(0x83, payload = byteArrayOf(0, 1))), ByteArrayOutputStream()).list()
        }
        val wire = CameraWire(ByteArrayInputStream(reply(0x85, payload = byteArrayOf(1, 2), length = 3)), ByteArrayOutputStream())
        assertThrows(IOException::class.java) { wire.downloadInMemory(UUID.randomUUID(), 3) }
    }

    @Test fun downloadTimerStartsAfterRequestFlushAndEndsBeforeCompletionCallback() {
        var now = 100L
        val outgoing = object : ByteArrayOutputStream() {
            override fun flush() { super.flush(); now = 1_000L }
        }
        val image = byteArrayOf(1, 2, 3, 4)
        val input = object : ByteArrayInputStream(reply(0x85, payload = image)) {
            override fun read(bytes: ByteArray, offset: Int, length: Int): Int {
                now += 200L
                return super.read(bytes, offset, length)
            }
        }
        val clockReads = mutableListOf<Long>()
        val wire = CameraWire(input, outgoing)
        val download = wire.downloadInMemory(UUID.randomUUID(), image.size.toLong(),
            nanoTime = { now.also { clockReads.add(it) } },
            progress = { bytes, total, elapsed ->
                assertEquals(4L, bytes); assertEquals(4L, total); assertEquals(200L, elapsed)
                // Saving/UI work after the last read must not extend the result.
                now += 50_000L
            })
        assertEquals(listOf(1_000L, 1_200L), clockReads)
        assertEquals(200L, download.elapsedNanos)
        assertArrayEquals(image, download.jpeg)
    }
}
