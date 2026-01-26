import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime

# =========================
# DEVICE INFO
# =========================
ADDRESS = "18:7A:93:12:26:AE"   # Rossmax X3 BT
BP_MEASUREMENT_UUID = "00002a35-0000-1000-8000-00805f9b34fb"

readings = []   # (time, sys, dia, map, pulse)


# =========================
# BP DATA DECODER
# =========================
def decode_bp(data: bytearray):
    flags = data[0]

    systolic  = int.from_bytes(data[1:3], "little")
    diastolic = int.from_bytes(data[3:5], "little")
    mean_art  = int.from_bytes(data[5:7], "little")

    index = 7
    if flags & 0x02:  # timestamp present
        index += 7

    pulse = None
    if flags & 0x04:
        pulse = int.from_bytes(data[index:index+2], "little")

    return systolic, diastolic, mean_art, pulse


# =========================
# NOTIFICATION HANDLER
# =========================
def handle_bp(sender, data):
    sys, dia, map_, pulse = decode_bp(data)

    if map_ == 0:
        map_ = round(dia + (sys - dia) / 3, 1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    readings.append((timestamp, sys, dia, map_, pulse))
    print(f"🩺 [{timestamp}] SYS:{sys} DIA:{dia} MAP:{map_} Pulse:{pulse}")


# =========================
# FAST CONNECT (NO LONG SCAN)
# =========================
async def fast_connect():
    try:
        client = BleakClient(ADDRESS, timeout=10)
        await client.connect()
        if client.is_connected:
            print("⚡ Fast connected (no scan)")
            return client
    except:
        pass
    return None


# =========================
# SHORT SCAN (BACKUP)
# =========================
async def scan_and_connect():
    print("🔍 Short scan...")
    devices = await BleakScanner.discover(timeout=3)
    for d in devices:
        if d.address == ADDRESS:
            client = BleakClient(ADDRESS, timeout=10)
            await client.connect()
            if client.is_connected:
                print("✅ Connected after scan")
                return client
    return None


# =========================
# CONNECT & READ ONCE
# =========================
async def connect_and_read():
    client = await fast_connect()
    if not client:
        client = await scan_and_connect()

    if not client:
        print("❌ Device not found")
        return False

    try:
        print("🔗 Press BP button on device")

        await client.start_notify(BP_MEASUREMENT_UUID, handle_bp)

        start_len = len(readings)
        timeout = 50

        for _ in range(timeout):
            if len(readings) > start_len:
                break
            await asyncio.sleep(1)

        await client.stop_notify(BP_MEASUREMENT_UUID)
        await client.disconnect()

        print("🔌 Reading done & disconnected\n")
        return True

    except Exception as e:
        print("❌ Error:", e)
        try:
            await client.disconnect()
        except:
            pass
        return False


# =========================
# MAIN LOOP
# =========================
async def main():
    total = int(input("📌 How many readings? "))

    for i in range(total):
        input(f"\n👉 Turn ON device & press ENTER for reading {i+1}...")
        await connect_and_read()

        if i < total - 1:
            input("⚠️ Turn device OFF & ON again, press ENTER...")

    print("\n📊 FINAL RESULTS")
    for i, r in enumerate(readings, 1):
        print(f"{i}. [{r[0]}] SYS={r[1]} DIA={r[2]} MAP={r[3]} Pulse={r[4]}")


asyncio.run(main())
