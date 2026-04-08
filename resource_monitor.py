"""
Resource Monitor — theo dõi RAM, CPU, Disk I/O của desktop app ASR theo thời gian thực.

Tự động tìm process app.py hoặc chỉ định PID.
Hiển thị console + ghi CSV khi kết thúc.
Space = pause/resume console. Ctrl+C = dừng.

Chạy từ bản portable:
    python\python.exe resource_monitor.py
    python\python.exe resource_monitor.py --interval 5
    python\python.exe resource_monitor.py --pid 12345
"""
import sys
import time
import argparse
import csv
import os
import msvcrt
from datetime import datetime

# Fix Windows console encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

import psutil

NUM_CORES = psutil.cpu_count(logical=True)


def find_asr_process():
    """Tìm process desktop app ASR (app.py hoặc sherpa-vietnamese-asr)."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'app.py' in cmdline and 'python' in cmdline.lower():
                return proc
            if 'sherpa-vietnamese-asr' in (proc.info['name'] or '').lower():
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def safe_proc_io(proc):
    try:
        io = proc.io_counters()
        return io.read_bytes, io.write_bytes
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return None


def read_phase(proc):
    """Đọc phase hiện tại từ file .asr_phase (do asr_engine.py ghi)."""
    try:
        exe = proc.exe()
        base = os.path.dirname(exe) if 'python' in exe.lower() else os.path.dirname(exe)
        cwd = proc.cwd()
        for d in [os.path.dirname(base), base, cwd, os.path.dirname(cwd)]:
            pf = os.path.join(d, '.asr_phase')
            if os.path.exists(pf):
                with open(pf, 'r', encoding='utf-8') as f:
                    line = f.read().strip()
                if line.startswith('PHASE:'):
                    parts = line[6:].split('|')
                    return parts[0] if parts else ''
                elif line:
                    return line
    except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return ''


def collect_sample(proc, prev_proc_io, prev_per_disk, dt, start_time, disk_names):
    elapsed = time.time() - start_time

    mem = proc.memory_info()
    rss = mem.rss / 1024 / 1024
    vms = mem.vms / 1024 / 1024
    cpu = proc.cpu_percent() / NUM_CORES
    threads = proc.num_threads()

    cur_io = safe_proc_io(proc)
    pr = pw = 0.0
    if cur_io and prev_proc_io:
        pr = (cur_io[0] - prev_proc_io[0]) / dt / 1024 / 1024
        pw = (cur_io[1] - prev_proc_io[1]) / dt / 1024 / 1024

    sm = psutil.virtual_memory()
    pf = psutil.swap_memory().used / 1024 / 1024

    cur_pd = psutil.disk_io_counters(perdisk=True)
    disks = {}
    for d in disk_names:
        c, p = cur_pd.get(d), prev_per_disk.get(d)
        if c and p:
            dr = (c.read_bytes - p.read_bytes) / dt / 1024 / 1024
            dw = (c.write_bytes - p.write_bytes) / dt / 1024 / 1024
            db = min(100, ((c.read_time - p.read_time) + (c.write_time - p.write_time)) / (dt * 1000) * 100)
            disks[d] = (dr, dw, db)
        else:
            disks[d] = (0, 0, 0)
    for nd in set(cur_pd.keys()) - set(disk_names):
        disk_names.append(nd)
        disks[nd] = (0, 0, 0)

    phase = read_phase(proc)

    row = {
        'elapsed': elapsed, 'rss': rss, 'vms': vms, 'cpu': cpu, 'threads': threads,
        'pr': pr, 'pw': pw,
        'ram_used': sm.used / 1024 / 1024, 'ram_total': sm.total / 1024 / 1024,
        'ram_pct': sm.percent, 'pagefile': pf, 'disks': disks, 'phase': phase,
    }
    return row, cur_io, cur_pd


def format_row(n, row, disk_names):
    ram_s = f"{row['ram_used']:.0f}/{row['ram_total']:.0f}"
    dl = ""
    for d in disk_names[:4]:
        r, w, b = row['disks'].get(d, (0, 0, 0))
        dl += f" |{d}: {r:>5.1f} {w:>4.1f} {b:>2.0f}%"
    phase = row.get('phase', '')
    phase_str = f" [{phase}]" if phase else ""
    return (f"{n:>4} {row['elapsed']:>5.0f}s {row['rss']:>5.0f}M {row['vms']:>6.0f}M {row['cpu']:>3.0f}% {row['threads']:>3} "
            f"{row['pr']:>5.1f}M {row['pw']:>5.1f}M {ram_s:>10} {row['pagefile']:>5.0f}M{dl}{phase_str}")


def main():
    parser = argparse.ArgumentParser(description="Resource Monitor cho Sherpa Vietnamese ASR Desktop App")
    parser.add_argument('--pid', type=int, help="PID cua process can monitor (mac dinh: tu dong tim app.py)")
    parser.add_argument('--interval', type=float, default=5.0, help="Khoang cach giua cac lan do (giay, mac dinh: 5)")
    args = parser.parse_args()

    # CSV file: cung thu muc voi monitor, ten theo thoi gian
    csv_name = f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)

    print(f"CPU cores: {NUM_CORES}")

    proc = None
    if args.pid:
        try:
            proc = psutil.Process(args.pid)
            print(f"Monitor PID {args.pid}: {proc.name()}")
        except psutil.NoSuchProcess:
            print(f"PID {args.pid} khong ton tai")
            return
    else:
        print("Dang tim process ASR (app.py)... Mo app roi cho.")
        while True:
            proc = find_asr_process()
            if proc:
                try:
                    print(f"\nPID {proc.pid}: {proc.exe()}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"\nPID {proc.pid}")
                break
            time.sleep(0.5)

    proc.cpu_percent()
    time.sleep(0.1)
    prev_proc_io = safe_proc_io(proc)
    prev_per_disk = psutil.disk_io_counters(perdisk=True)
    disk_names = sorted(prev_per_disk.keys())
    prev_time = time.time()
    start_time = prev_time
    peak_rss = 0
    peak_vms = 0
    peak_cpu = 0

    print(f"Disks: {disk_names}")
    print(f"Interval: {args.interval}s")
    print(f"CSV: {csv_path}")
    print(f"[Space] = Pause/Resume console | [Ctrl+C] = Dung\n")

    # Header
    hdr = (f"{'#':>4} {'Time':>5} {'RSS':>6} {'VMS':>7} {'CPU':>4} {'Thr':>3} "
           f"{'pR':>6} {'pW':>6} {'sRAM':>10} {'PgF':>6}")
    for d in disk_names[:4]:
        hdr += f" |{d}: {'R':>5} {'W':>4} {'B':>3}"
    hdr += "  Phase"
    print(hdr)
    print("-" * len(hdr))

    all_rows = []  # Ghi CSV 1 lan khi ket thuc (tranh I/O lien tuc)
    paused = False
    n = 0

    try:
        while True:
            time.sleep(args.interval)

            # Space = pause/resume CONSOLE (CSV van thu thap)
            while msvcrt.kbhit():
                if msvcrt.getch() == b' ':
                    paused = not paused
                    if paused:
                        print(f"\n  --- PAUSE (Space de tiep tuc) ---")
                    else:
                        print(f"  --- RESUME ---\n")

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt <= 0:
                continue

            try:
                if not proc.is_running():
                    print("\nProcess ket thuc.")
                    break

                row, prev_proc_io, prev_per_disk = collect_sample(
                    proc, prev_proc_io, prev_per_disk, dt, start_time, disk_names)
                n += 1

                if row['rss'] > peak_rss: peak_rss = row['rss']
                if row['vms'] > peak_vms: peak_vms = row['vms']
                if row['cpu'] > peak_cpu: peak_cpu = row['cpu']

                # Luon luu vao memory (ghi CSV 1 lan khi ket thuc)
                all_rows.append(row)

                # Console: chi hien khi khong pause
                if not paused:
                    print(format_row(n, row, disk_names))

            except psutil.NoSuchProcess:
                print("\nProcess ket thuc.")
                break

    except KeyboardInterrupt:
        print("\nDung.")

    finally:
        # Ghi CSV 1 lan duy nhat
        if all_rows:
            all_disk_names = sorted(set(d for row in all_rows for d in row['disks']))
            csv_header_final = ['#', 'Time(s)', 'RSS(MB)', 'VMS(MB)', 'CPU(%)', 'Threads',
                                'ProcRead(MB/s)', 'ProcWrite(MB/s)',
                                'SysRAM_used(MB)', 'SysRAM_total(MB)', 'SysRAM(%)',
                                'Pagefile(MB)', 'Phase']
            for d in all_disk_names:
                csv_header_final += [f'{d}_R(MB/s)', f'{d}_W(MB/s)', f'{d}_B(%)']

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(csv_header_final)
                for i, row in enumerate(all_rows):
                    vals = [i + 1, f"{row['elapsed']:.1f}", f"{row['rss']:.0f}",
                            f"{row['vms']:.0f}", f"{row['cpu']:.0f}", row['threads'],
                            f"{row['pr']:.1f}", f"{row['pw']:.1f}",
                            f"{row['ram_used']:.0f}", f"{row['ram_total']:.0f}",
                            f"{row['ram_pct']:.0f}", f"{row['pagefile']:.0f}",
                            row.get('phase', '')]
                    for d in all_disk_names:
                        r, ww, b = row['disks'].get(d, (0, 0, 0))
                        vals += [f"{r:.1f}", f"{ww:.1f}", f"{b:.0f}"]
                    w.writerow(vals)

        elapsed = time.time() - start_time
        tr = tw = 0
        fi = safe_proc_io(proc) if proc.is_running() else prev_proc_io
        if fi: tr, tw = fi

        print(f"\n{'='*50}")
        print(f"  TONG KET ({n} samples / {elapsed:.0f}s)")
        print(f"  Peak RSS:  {peak_rss:.0f} MB (RAM vat ly)")
        print(f"  Peak VMS:  {peak_vms:.0f} MB (virtual)")
        print(f"  Peak CPU:  {peak_cpu:.0f}%")
        print(f"  Disk read: {tr/1024/1024:.0f} MB")
        print(f"  Disk write:{tw/1024/1024:.0f} MB")
        if all_rows:
            print(f"  CSV:       {csv_path}")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()
