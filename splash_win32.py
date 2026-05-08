"""Win32 splash screen bằng ctypes — zero deps, <10ms, luôn trên cùng.

Dùng khi app đang load thư viện nặng (onnxruntime, sherpa_onnx, PyQt6).
Không xung đột DLL với ORT vì không load Qt/tkinter.
"""
import ctypes
from ctypes import wintypes, WINFUNCTYPE

_state = {"hwnd": None, "fonts": [], "brush": None, "wndproc": None}


def _setup_signatures(user32, gdi32, kernel32):
    """Khai báo argtypes/restype cho 64-bit safety (HWND, HBRUSH là pointer)."""
    user32.DefWindowProcW.restype = ctypes.c_ssize_t
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                      wintypes.WPARAM, wintypes.LPARAM]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.UpdateWindow.argtypes = [wintypes.HWND]
    user32.UpdateWindow.restype = wintypes.BOOL
    user32.SendMessageW.restype = ctypes.c_ssize_t
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                    wintypes.WPARAM, wintypes.LPARAM]
    user32.LoadCursorW.restype = wintypes.HANDLE
    user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.PeekMessageW.argtypes = [ctypes.c_void_p, wintypes.HWND,
                                    wintypes.UINT, wintypes.UINT, wintypes.UINT]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.c_void_p]
    user32.DispatchMessageW.argtypes = [ctypes.c_void_p]
    user32.DispatchMessageW.restype = ctypes.c_ssize_t

    gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
    gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
    gdi32.CreateFontW.restype = wintypes.HFONT
    gdi32.CreateFontW.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPCWSTR,
    ]
    gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
    gdi32.SetTextColor.restype = wintypes.COLORREF
    gdi32.SetBkColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
    gdi32.SetBkColor.restype = wintypes.COLORREF
    gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
    gdi32.SetBkMode.restype = ctypes.c_int

    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]


def _enable_dpi_awareness(user32):
    """Bật DPI awareness khớp với Qt6 (per-monitor v2) để splash không bị
    bitmap-scale rồi thu nhỏ khi Qt load. Phải gọi TRƯỚC CreateWindowExW."""
    # Win10 1703+ API: SetProcessDpiAwarenessContext
    # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4 (giống Qt6)
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    # Fallback Win8.1+: shcore!SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE=2)
    try:
        shcore = ctypes.WinDLL('shcore')
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness(2)
        return
    except (OSError, AttributeError):
        pass
    # Fallback Vista+: SetProcessDPIAware (system DPI aware)
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


def _get_primary_dpi(user32):
    """Lấy DPI của màn hình chính. Trả về 96 nếu không query được."""
    try:
        user32.GetDpiForSystem.restype = wintypes.UINT
        return user32.GetDpiForSystem()
    except AttributeError:
        # Pre-Win10 fallback: query HDC của desktop
        try:
            gdi32 = ctypes.WinDLL('gdi32')
            user32.GetDC.restype = wintypes.HDC
            user32.GetDC.argtypes = [wintypes.HWND]
            gdi32.GetDeviceCaps.argtypes = [wintypes.HDC, ctypes.c_int]
            gdi32.GetDeviceCaps.restype = ctypes.c_int
            hdc = user32.GetDC(None)
            LOGPIXELSY = 90
            dpi = gdi32.GetDeviceCaps(hdc, LOGPIXELSY)
            user32.ReleaseDC(None, hdc)
            return dpi or 96
        except Exception:
            return 96


def show(title="sherpa-vietnamese-asr", status="Đang khởi động..."):
    """Hiện splash. Trả về True nếu OK, False nếu lỗi."""
    try:
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        _setup_signatures(user32, gdi32, kernel32)

        # Bật DPI awareness TRƯỚC khi tạo window để tránh Windows bitmap-scale
        # splash lên rồi đột ngột co lại khi Qt6 enable DPI awareness.
        _enable_dpi_awareness(user32)
        _dpi = _get_primary_dpi(user32)
        _scale = _dpi / 96.0

        WNDPROC = WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                              wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASSEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
                ("hIconSm", wintypes.HICON),
            ]

        user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
        user32.RegisterClassExW.restype = wintypes.ATOM

        # Constants
        WS_POPUP = 0x80000000
        WS_VISIBLE = 0x10000000
        WS_CHILD = 0x40000000
        WS_BORDER = 0x00800000
        WS_EX_TOPMOST = 0x00000008
        WS_EX_TOOLWINDOW = 0x00000080
        SS_CENTER = 0x00000001
        WM_SETFONT = 0x0030
        SM_CXSCREEN = 0
        SM_CYSCREEN = 1
        IDC_ARROW = 32512

        # Màu theo theme tối của app:
        # COLORREF = 0x00BBGGRR — 0x2E1E1E là BGR của #1E1E2E (xanh đen)
        BG_COLOR = 0x002E1E1E
        TEXT_COLOR = 0x00F0F0F0  # trắng ngà
        bg_brush = gdi32.CreateSolidBrush(BG_COLOR)

        # Custom WndProc: xử lý WM_CTLCOLORSTATIC để set text trắng + nền tối
        # cho các STATIC child (không thì Windows vẽ text đen trên nền xám mặc định).
        WM_CTLCOLORSTATIC = 0x0138
        TRANSPARENT = 1
        def _wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_CTLCOLORSTATIC:
                hdc = wparam  # HDC của STATIC child
                gdi32.SetTextColor(hdc, TEXT_COLOR)
                gdi32.SetBkColor(hdc, BG_COLOR)
                gdi32.SetBkMode(hdc, TRANSPARENT)
                return bg_brush  # trả HBRUSH nền cho Windows tô vùng control
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
        wndproc_fn = WNDPROC(_wndproc)

        hinstance = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = 0
        wc.lpfnWndProc = wndproc_fn
        wc.hInstance = hinstance
        # LoadCursorW với predefined ID cần cast int → LPCWSTR (MAKEINTRESOURCE)
        wc.hCursor = user32.LoadCursorW(None, ctypes.cast(IDC_ARROW, wintypes.LPCWSTR))
        wc.hbrBackground = bg_brush
        wc.lpszClassName = "AsrVnSplashCls"

        user32.RegisterClassExW(ctypes.byref(wc))
        # Nếu class đã registered từ lần trước, RegisterClassExW sẽ fail
        # nhưng vẫn OK vì class cũ còn dùng được.

        # Với DPI-aware, GetSystemMetrics trả physical pixels của màn hình chính
        sw = user32.GetSystemMetrics(SM_CXSCREEN)
        sh = user32.GetSystemMetrics(SM_CYSCREEN)
        # Scale logical size (420×160 at 96 DPI) → physical pixels theo DPI hiện tại
        ww = int(420 * _scale)
        wh = int(160 * _scale)
        x, y = (sw - ww) // 2, (sh - wh) // 2

        hwnd = user32.CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            "AsrVnSplashCls", "",
            WS_POPUP | WS_VISIBLE | WS_BORDER,
            x, y, ww, wh,
            None, None, hinstance, None,
        )
        if not hwnd:
            return False

        FW_BOLD = 700
        FW_NORMAL = 400
        # Font height âm = character height (không tính leading) — Windows
        # tự tính cell. Scale theo DPI.
        font_big = gdi32.CreateFontW(int(-22 * _scale), 0, 0, 0, FW_BOLD,
                                     0, 0, 0, 0, 0, 0, 0, 0, "Segoe UI")
        font_mid = gdi32.CreateFontW(int(-14 * _scale), 0, 0, 0, FW_NORMAL,
                                     0, 0, 0, 0, 0, 0, 0, 0, "Segoe UI")
        font_sml = gdi32.CreateFontW(int(-11 * _scale), 0, 0, 0, FW_NORMAL,
                                     0, 0, 0, 0, 0, 0, 0, 0, "Segoe UI")

        # Layout children: scale y/height theo DPI
        lbl1 = user32.CreateWindowExW(0, "STATIC", title,
            WS_CHILD | WS_VISIBLE | SS_CENTER,
            0, int(30 * _scale), ww, int(32 * _scale), hwnd, None, None, None)
        lbl2 = user32.CreateWindowExW(0, "STATIC", status,
            WS_CHILD | WS_VISIBLE | SS_CENTER,
            0, int(75 * _scale), ww, int(22 * _scale), hwnd, None, None, None)
        lbl3 = user32.CreateWindowExW(0, "STATIC",
            "Nhận dạng giọng nói tiếng Việt — Offline, CPU",
            WS_CHILD | WS_VISIBLE | SS_CENTER,
            0, int(125 * _scale), ww, int(18 * _scale), hwnd, None, None, None)

        user32.SendMessageW(lbl1, WM_SETFONT, font_big, 1)
        user32.SendMessageW(lbl2, WM_SETFONT, font_mid, 1)
        user32.SendMessageW(lbl3, WM_SETFONT, font_sml, 1)
        user32.UpdateWindow(hwnd)

        _state["hwnd"] = hwnd
        _state["fonts"] = [font_big, font_mid, font_sml]
        _state["brush"] = bg_brush
        _state["wndproc"] = wndproc_fn  # giữ reference tránh GC
        return True
    except Exception as e:
        print(f"[Splash] Win32 splash failed: {e}")
        return False


def pump():
    """Pump message queue để splash được vẽ lại (gọi giữa các import nặng)."""
    if not _state.get("hwnd"):
        return
    try:
        user32 = ctypes.WinDLL('user32')
        user32.PeekMessageW.argtypes = [ctypes.c_void_p, wintypes.HWND,
                                        wintypes.UINT, wintypes.UINT, wintypes.UINT]
        user32.PeekMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = [ctypes.c_void_p]
        user32.DispatchMessageW.argtypes = [ctypes.c_void_p]
        msg = wintypes.MSG()
        PM_REMOVE = 0x0001
        for _ in range(16):
            if not user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except Exception:
        pass


def destroy():
    """Đóng splash và giải phóng GDI objects."""
    try:
        user32 = ctypes.WinDLL('user32')
        gdi32 = ctypes.WinDLL('gdi32')
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        hwnd = _state.get("hwnd")
        if hwnd:
            user32.DestroyWindow(hwnd)
        for f in _state.get("fonts") or []:
            if f:
                gdi32.DeleteObject(f)
        brush = _state.get("brush")
        if brush:
            gdi32.DeleteObject(brush)
    except Exception:
        pass
    finally:
        _state["hwnd"] = None
        _state["fonts"] = []
        _state["brush"] = None
        _state["wndproc"] = None
