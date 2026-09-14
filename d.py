import psutil, subprocess, sys, time

proc = subprocess.Popen([sys.executable, "app.py"])
p = psutil.Process(proc.pid)

peak_ram = 0
peak_cpu = 0

print(f"Monitoring app.py (PID {proc.pid})... Ctrl+C to stop\n")

try:
    while proc.poll() is None:
        try:
            cpu = p.cpu_percent(interval=1)
            ram = p.memory_info().rss
            peak_ram = max(peak_ram, ram)
            peak_cpu = max(peak_cpu, cpu)
            print(f"CPU: {cpu:5.1f}%   RAM: {ram/1024**2:7.1f} MB   "
                  f"| Peak CPU: {peak_cpu:5.1f}%  Peak RAM: {peak_ram/1024**2:7.1f} MB")
        except psutil.NoSuchProcess:
            break
except KeyboardInterrupt:
    pass

print(f"\n=== FINAL ===")
print(f"Peak CPU : {peak_cpu:.1f}%")
print(f"Peak RAM : {peak_ram/1024**2:.1f} MB ({peak_ram/1024**3:.2f} GB)")