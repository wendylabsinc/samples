package com.wendylabs.nancamera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.io.InterruptedIOException
import java.net.ConnectException
import java.net.InetSocketAddress
import java.net.Socket
import java.net.SocketAddress
import java.net.SocketTimeoutException

class InitialTcpConnectTest {
    private class ProbeSocket(private val refuse: Boolean) : Socket() {
        var closed = false
        override fun connect(endpoint: SocketAddress?, timeout: Int) {
            if (refuse) throw ConnectException("Connection refused")
        }
        override fun close() {
            closed = true
            super.close()
        }
    }

    private val address = InetSocketAddress("127.0.0.1", 9091)

    @Test fun retriesRefusedDialWithFreshSockets() {
        val made = mutableListOf<ProbeSocket>()
        var clock = 0L
        var current: Socket? = null
        val connected = openInitialCameraSocket(
            socketFactory = { ProbeSocket(made.size < 2).also(made::add) },
            address = address,
            active = { true },
            onAttemptSocket = { current = it },
            nowNanos = { clock },
            pauseMillis = { clock += it * 1_000_000 },
        )
        assertEquals(3, made.size)
        assertTrue(made[0].closed)
        assertTrue(made[1].closed)
        assertFalse(made[2].closed)
        assertSame(made[2], connected)
        assertSame(connected, current)
        connected.close()
    }

    @Test fun cancellationStopsBeforeAnotherDial() {
        var active = true
        var attempts = 0
        assertThrows(InterruptedIOException::class.java) {
            openInitialCameraSocket(
                socketFactory = { attempts++; ProbeSocket(true) },
                address = address,
                active = { active },
                onAttemptSocket = { if (it == null) active = false },
                pauseMillis = { throw AssertionError("Cancelled connection must not wait") },
            )
        }
        assertEquals(1, attempts)
    }

    @Test fun stopsAtDeadlineInsteadOfRetryingForever() {
        var clock = 0L
        var attempts = 0
        val error = assertThrows(IOException::class.java) {
            openInitialCameraSocket(
                socketFactory = { attempts++; ProbeSocket(true) },
                address = address,
                active = { true },
                onAttemptSocket = {},
                timeoutMillis = 450,
                nowNanos = { clock },
                pauseMillis = { clock += it * 1_000_000 },
            )
        }
        assertTrue(attempts in 2..3)
        assertTrue(error.message!!.contains("450ms"))
        assertTrue(error.cause is ConnectException)
    }

    @Test fun slowNanSetupCanCompleteOnFirstSocket() {
        var clock = 0L
        var attempts = 0
        val candidate = object : Socket() {
            override fun connect(endpoint: SocketAddress?, timeout: Int) {
                assertTrue("NAN setup must survive a 3-second initial delay", timeout >= 3_000)
                clock += 3_000_000_000L
            }
        }
        val connected = openInitialCameraSocket(
            socketFactory = { attempts++; candidate }, address = address,
            active = { true }, onAttemptSocket = {}, nowNanos = { clock },
            pauseMillis = { throw AssertionError("Successful first dial must not retry") },
        )
        assertEquals(1, attempts)
        assertSame(candidate, connected)
        connected.close()
    }

    @Test fun timedOutDialsShareTheOverallTenSecondBudget() {
        var clock = 0L
        val budgets = mutableListOf<Int>()
        var closed = 0
        assertThrows(IOException::class.java) {
            openInitialCameraSocket(
                socketFactory = { object : Socket() {
                    override fun connect(endpoint: SocketAddress?, timeout: Int) {
                        budgets += timeout
                        clock += timeout * 1_000_000L
                        throw SocketTimeoutException("Timed out")
                    }
                    override fun close() { closed++; super.close() }
                } },
                address = address, active = { true }, onAttemptSocket = {},
                nowNanos = { clock }, pauseMillis = { clock += it * 1_000_000 },
            )
        }
        assertEquals(listOf(5_000, 4_800), budgets)
        assertEquals(2, closed)
        assertEquals(10_000_000_000L, clock)
    }

}
