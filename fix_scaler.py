import psutil, subprocess, sys, time

proc = subprocess.Popen([sys.executable, "app.py"])
time.sleep(3)  # let Flask/TF boot

# Find ALL python processes and pick the one using the most RAM
candidates = []
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        name = (p.info['name'] or '').lower()
        if 'python' in name:
            candidates.append(p)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

if not candidates:
    print("No python processes found")
    sys.exit(1)

# Sort by RSS descending, pick the biggest
candidates.sort(key=lambda p: p.memory_info().rss, reverse=True)
target = candidates[0]

print(f"Monitoring PID {target.pid} ({target.name()})")
print(f"All python PIDs found: {[p.pid for p in candidates]}")

peak_ram = peak_cpu = 0
while True:
    try:
        cpu = target.cpu_percent(interval=1)
        ram = target.memory_info().rss
        peak_ram = max(peak_ram, ram)
        peak_cpu = max(peak_cpu, cpu)
        print(f"CPU: {cpu:5.1f}%   RAM: {ram/1024**2:7.1f} MB   "
              f"| Peak CPU: {peak_cpu:5.1f}%  Peak RAM: {peak_ram/1024**2:7.1f} MB")
    except psutil.NoSuchProcess:
        print("Process ended")
        break