package com.wendylabs.nancamera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class CameraAdvertisementTest {
    @Test fun acceptsFixedPortWithNoServiceInfo() {
        assertEquals(9091, cameraPortFromServiceInfo(null))
        assertEquals(9091, cameraPortFromServiceInfo(byteArrayOf()))
    }

    @Test fun acceptsLegacyServiceInfoButRejectsUnknownPayload() {
        assertEquals(9091, cameraPortFromServiceInfo(byteArrayOf(1, 0x23, 0x83.toByte(), 0)))
        assertNull(cameraPortFromServiceInfo(byteArrayOf(1, 0x23, 0x83.toByte(), 1)))
        assertNull(cameraPortFromServiceInfo(byteArrayOf(1)))
    }
}
