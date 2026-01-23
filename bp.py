
# # get max address of device
# from bleak import BleakScanner
# import asyncio
#
# async def scan():
#     devices = await BleakScanner.discover(10)
#     for d in devices:
#         print(d)
#
# asyncio.run(scan())
#

# # connect with max address
# import asyncio
# from bleak import BleakClient
#
# ADDRESS = "18:7A:93:12:26:AE"  #ROSSMAX X3 BT MAX ADDRESS
#
# async def test_connect():
#     async with BleakClient(ADDRESS) as client:
#         print("✅ Connected:", client.is_connected)
#
# asyncio.run(test_connect())
#


#GET  Characteristics OF UUID
#
# import asyncio
# from bleak import BleakClient
#
# ADDRESS = "18:7A:93:12:26:AE"
#
# async def list_services():
#     async with BleakClient(ADDRESS) as client:
#         for service in client.services:
#             print(service.uuid, service.description)
#             for char in service.characteristics:
#                 print("   ", char.uuid, char.properties)
#
# asyncio.run(list_services())


import asyncio
from bleak import BleakClient, BleakScanner

ADDRESS = "18:7A:93:12:26:AE"
BP_MEASUREMENT = "00002a35-0000-1000-8000-00805f9b34fb"

readings = []


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


def handle_bp(sender, data):
    sys, dia, map_, pulse = decode_bp(data)

    if map_ == 0:
        map_ = round(dia + (sys - dia) / 3, 1)

    readings.append((sys, dia, map_, pulse))
    print(f"🩺 READING → SYS:{sys} DIA:{dia} MAP:{map_} Pulse:{pulse}")


async def wait_for_device():
    print("🔍 Waiting for device to wake up...")
    while True:
        devices = await BleakScanner.discover(timeout=5)
        for d in devices:
            if d.address == ADDRESS:
                print("✅ Device detected")
                return
        await asyncio.sleep(2)

async def connect_and_read():
    await wait_for_device()

    async with BleakClient(ADDRESS) as client:
        print("🔗 Connected to Rossmax — press BP button on device")

        # Retry notify until it works
        for attempt in range(5):
            try:
                await client.start_notify(BP_MEASUREMENT, handle_bp)
                break
            except OSError:
                print(f"⚠️ Notify start failed, retrying... ({attempt+1}/5)")
                await asyncio.sleep(2)
        else:
            print("❌ Failed to start notify. Skipping this reading.")
            return

        # Wait until first reading is actually received
        reading_received = False
        while not reading_received:
            if len(readings) > 0 and readings[-1][0] is not None:
                reading_received = True
            else:
                print("⏳ Waiting for user to press BP button on device...")
                await asyncio.sleep(1)

        await client.stop_notify(BP_MEASUREMENT)
        print("🔌 Reading captured and connection closed\n")


async def main():
    total_readings = int(input("How many readings do you want to take? "))

    for i in range(total_readings):
        input(f"\n👉 Please TURN ON the device and press ENTER to connect for reading {i+1}...")

        await connect_and_read()

        if i < total_readings - 1:
            input("⚠️ Device may sleep. Turn it OFF and then ON again. Press ENTER when ready for next reading...")

    print("\n📊 FINAL RESULTS")
    for i, r in enumerate(readings, 1):
        print(f"Reading {i}: SYS={r[0]} DIA={r[1]} MAP={r[2]} Pulse={r[3]}")


asyncio.run(main())
