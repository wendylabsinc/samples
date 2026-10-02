package com.wendylabs.nancamera

import java.io.IOException
import java.io.InterruptedIOException
import java.net.Socket
import java.net.SocketAddress

/** The NDP can report available before the publisher has bound its TCP listener. */
internal fun openInitialCameraSocket(
    socketFactory: () -> Socket,
    address: SocketAddress,
    active: () -> Boolean,
    onAttemptSocket: (Socket?) -> Unit,
    timeoutMillis: Long = 10_000,
    nowNanos: () -> Long = System::nanoTime,
    pauseMillis: (Long) -> Unit = Thread::sleep,
): Socket {
    require(timeoutMillis > 0)
    val deadline = nowNanos() + timeoutMillis * 1_000_000
    var lastFailure: IOException? = null
    while (active()) {
        val remainingNanos = deadline - nowNanos()
        if (remainingNanos <= 0) break
        var candidate: Socket? = null
        try {
            candidate = socketFactory()
            onAttemptSocket(candidate)
            // Initial IPv6 discovery and NAN availability windows can consume
            // multiple seconds before a SYN is delivered. Let each socket survive
            // that setup while retaining the overall connection deadline.
            val dialMillis = minOf(5_000L, maxOf(1L, remainingNanos / 1_000_000)).toInt()
            candidate.connect(address, dialMillis)
            if (!active()) throw InterruptedIOException("Camera connection cancelled")
            return candidate
        } catch (e: IOException) {
            lastFailure = e
            runCatching { candidate?.close() }
            onAttemptSocket(null)
            if (!active()) throw InterruptedIOException("Camera connection cancelled")
        }
        val waitMillis = minOf(200L, maxOf(0L, (deadline - nowNanos()) / 1_000_000))
        if (waitMillis > 0) pauseMillis(waitMillis)
    }
    if (!active()) throw InterruptedIOException("Camera connection cancelled")
    throw IOException("Camera TCP listener did not become ready within ${timeoutMillis}ms", lastFailure)
}
