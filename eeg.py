import subprocess
import sys
import time
from pylsl import StreamInlet, resolve_byprop
import csv

# Start muselsl stream as subprocess
python_exe = sys.executable
proc = subprocess.Popen([python_exe, "-m", "muselsl", "stream"])

print("Waiting for Muse to connect...")

# Wait for EEG stream to appear, with a maximum timeout
timeout = 15  # seconds
start_time = time.time()
streams = []

while time.time() - start_time < timeout:
    streams = resolve_byprop("type", "EEG", timeout=1)
    if streams:
        print("🎉 Muse connected! EEG stream found.")
        break
    else:
        print("Searching for Muse...", end="\r")
        time.sleep(1)

if not streams:
    print("\n❌ No EEG stream found. Exiting.")
    proc.terminate()
    exit()

# Now we have a valid inlet
inlet = StreamInlet(streams[0])

# Save data
filename = "eeg_test_data.csv"
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "TP9", "AF7", "AF8", "TP10"])
    print("Saving EEG data for 20 seconds... 🎧")
    start = time.time()
    try:
        while time.time() - start < 20:
            sample, ts = inlet.pull_sample()
            writer.writerow([ts] + sample)
    except KeyboardInterrupt:
        print("Data collection interrupted by user!")

print("Data saved! 🎉")
time.sleep(2)
proc.terminate()
print("Stream stopped successfully.")
