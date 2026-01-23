import subprocess
import sys
import time
from pylsl import StreamInlet, resolve_byprop
import csv

# =========================
# START MUSE STREAM WITH PPG
# =========================
python_exe = sys.executable
# Add --ppg flag to ensure PPG stream
proc = subprocess.Popen([python_exe, "-m", "muselsl", "stream", "--ppg"])
print("🎧 Waiting for Muse PPG stream...")

# Wait for PPG stream
timeout = 15  # seconds
start_time = time.time()
streams = []

while time.time() - start_time < timeout:
    streams = resolve_byprop("type", "PPG", timeout=1)
    if streams:
        print("🎉 Muse connected! PPG stream found.")
        break
    else:
        print("Searching for Muse PPG...", end="\r")
        time.sleep(1)

if not streams:
    print("\n❌ No PPG stream found. Exiting.")
    proc.terminate()
    exit()

# =========================
# CREATE INLET
# =========================
inlet = StreamInlet(streams[0])
channel_count = inlet.info().channel_count()
channel_names = [f"PPG{i+1}" for i in range(channel_count)]

# =========================
# SAVE PPG DATA
# =========================
filename = "ppg_test_data.csv"
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp"] + channel_names)
    print("Saving PPG data for 20 seconds... 🎧")
    start = time.time()
    try:
        while time.time() - start < 20:
            sample, ts = inlet.pull_sample(timeout=1.0)
            if sample:
                writer.writerow([ts] + sample)
    except KeyboardInterrupt:
        print("\n🛑 Data collection interrupted by user!")

print("✅ PPG data saved to", filename)
time.sleep(1)
proc.terminate()
print("🛑 PPG stream stopped successfully.")
