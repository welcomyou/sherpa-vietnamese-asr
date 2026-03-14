"""
Windows Service installer cho Sherpa Vietnamese ASR.
Dung pywin32 (win32serviceutil).

Usage (chay voi quyen Admin):
    python service_installer.py install
    python service_installer.py remove
    python service_installer.py start
    python service_installer.py stop
"""

import os
import sys

# Them BASE_DIR vao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("[Service] pywin32 chua duoc cai dat.")
    print("[Service] Cai dat: pip install pywin32")
    print("[Service] Hoac dung nssm thay the:")
    print(f'  nssm install SherpaVietnameseASR "{sys.executable}" "{os.path.join(BASE_DIR, "server_launcher.py")}" --no-gui')

import subprocess


if HAS_WIN32:
    class SherpaASRService(win32serviceutil.ServiceFramework):
        _svc_name_ = "SherpaVietnameseASR"
        _svc_display_name_ = "Sherpa Vietnamese ASR Service"
        _svc_description_ = "Web service ASR tieng Viet - Sherpa Vietnamese ASR"

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.server_process = None

        def SvcDoRun(self):
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )

            # Khoi dong server
            launcher = os.path.join(BASE_DIR, "server_launcher.py")
            self.server_process = subprocess.Popen(
                [sys.executable, launcher, "--no-gui"],
                cwd=BASE_DIR,
            )

            # Doi cho den khi nhan stop signal
            win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)
            if self.server_process:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()

    if __name__ == "__main__":
        if len(sys.argv) == 1:
            servicemanager.Initialize()
            servicemanager.PrepareToHostSingle(SherpaASRService)
            servicemanager.StartServiceCtrlDispatcher()
        else:
            win32serviceutil.HandleCommandLine(SherpaASRService)

else:
    if __name__ == "__main__":
        print("\n=== Huong dan cai dat bang nssm ===")
        print(f"1. Tai nssm tu https://nssm.cc/download")
        print(f"2. Chay voi quyen Admin:")
        print(f'   nssm install SherpaASRVNOnline "{sys.executable}" "{os.path.join(BASE_DIR, "server_launcher.py")}" --no-gui')
        print(f"3. Lenh quan ly:")
        print(f"   nssm start SherpaASRVNOnline")
        print(f"   nssm stop SherpaASRVNOnline")
        print(f"   nssm remove SherpaASRVNOnline confirm")
