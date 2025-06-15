import asyncio
from bleak import BleakScanner, BleakClient
from config import TARGET_NAME, CHAR_UUID
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

class BluetoothController:
    def __init__(self):
        self.client = None

    async def connect(self):
        log("Scanning for HM-10 Bluetooth device...")
        devices = await BleakScanner.discover()
        target = next((d for d in devices if d.name and TARGET_NAME in d.name), None)

        if not target:
            log("Target Bluetooth device not found")
            return

        self.client = BleakClient(target.address)
        await self.client.connect()

        if not self.client.is_connected:
            log("Failed to connect to Bluetooth device")
            self.client = None
        else:
            log("Connected to Bluetooth device")

    async def send_command(self, cmd):
        if not self.client or not self.client.is_connected:
            log("⚠️ Bluetooth is not connected")
            return
        await self.client.write_gatt_char(CHAR_UUID, (cmd + "\n").encode())
        log(f"Command sent: {cmd}")
        await asyncio.sleep(0.2)  # Prevent commands from piling up too quickly

    async def disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            log("Bluetooth disconnected")
