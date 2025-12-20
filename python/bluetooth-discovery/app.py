#!/usr/bin/env python3
import asyncio
import json
import re
import subprocess
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

app = FastAPI()


class BluetoothDevice(BaseModel):
    name: str | None
    address: str
    rssi: int | None


def parse_hcitool_scan(output: str) -> list[BluetoothDevice]:
    """Parse output from hcitool scan command."""
    devices = []
    lines = output.strip().split('\n')
    for line in lines[1:]:  # Skip header line "Scanning ..."
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            address = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else None
            if address and re.match(r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$', address):
                devices.append(BluetoothDevice(
                    name=name if name else None,
                    address=address,
                    rssi=None  # hcitool scan doesn't provide RSSI
                ))
    return devices


def parse_hcitool_lescan(output: str) -> list[BluetoothDevice]:
    """Parse output from hcitool lescan command."""
    devices = {}
    for line in output.strip().split('\n'):
        # Format: "AA:BB:CC:DD:EE:FF DeviceName" or "AA:BB:CC:DD:EE:FF (unknown)"
        match = re.match(r'^([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})\s+(.*)$', line.strip())
        if match:
            address = match.group(1)
            name = match.group(2).strip()
            if name == "(unknown)":
                name = None
            # Only keep the first occurrence (which usually has the name)
            if address not in devices or (name and not devices[address].name):
                devices[address] = BluetoothDevice(
                    name=name,
                    address=address,
                    rssi=None
                )
    return list(devices.values())


async def discover_devices_classic(timeout: float = 5.0) -> list[BluetoothDevice]:
    """Discover classic Bluetooth devices using hcitool scan."""
    try:
        proc = await asyncio.create_subprocess_exec(
            'hcitool', 'scan', '--flush',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
        if proc.returncode == 0:
            return parse_hcitool_scan(stdout.decode())
    except asyncio.TimeoutError:
        proc.kill()
    except Exception as e:
        print(f"Error during classic scan: {e}")
    return []


async def discover_devices_ble(timeout: float = 5.0) -> list[BluetoothDevice]:
    """Discover BLE devices using hcitool lescan."""
    devices: list[BluetoothDevice] = []
    try:
        # lescan runs continuously, so we need to kill it after timeout
        proc = await asyncio.create_subprocess_exec(
            'timeout', str(int(timeout)), 'hcitool', 'lescan',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 2)
        devices = parse_hcitool_lescan(stdout.decode())
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        print(f"Error during BLE scan: {e}")
    return devices


async def discover_devices(timeout: float = 5.0) -> list[BluetoothDevice]:
    """Discover both classic and BLE Bluetooth devices."""
    # Run both scans concurrently
    classic_task = asyncio.create_task(discover_devices_classic(timeout))
    ble_task = asyncio.create_task(discover_devices_ble(timeout))

    classic_devices = await classic_task
    ble_devices = await ble_task

    # Merge devices by address
    devices_map = {d.address: d for d in classic_devices}
    for d in ble_devices:
        if d.address not in devices_map:
            devices_map[d.address] = d

    return list(devices_map.values())


@app.get("/discovered")
async def get_discovered_devices():
    """Return a JSON array of all discovered Bluetooth devices within 5 seconds."""
    devices = await discover_devices(timeout=5.0)
    return [device.model_dump() for device in devices]


@app.get("/events")
async def device_events():
    """SSE endpoint that streams discovered devices in real-time."""
    async def event_generator():
        seen_devices: set[str] = set()
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= 30:  # Stop after 30 seconds
                break

            try:
                # Do a quick scan
                devices = await discover_devices(timeout=3.0)

                for device in devices:
                    if device.address not in seen_devices:
                        seen_devices.add(device.address)
                        yield {
                            "event": "device",
                            "data": json.dumps(device.model_dump())
                        }

                # Send keepalive between scans
                yield {
                    "event": "keepalive",
                    "data": ""
                }

            except Exception as e:
                print(f"Scan error: {e}")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
                await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


@app.get("/")
async def root():
    """Serve the index.html file."""
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path, media_type="text/html")
