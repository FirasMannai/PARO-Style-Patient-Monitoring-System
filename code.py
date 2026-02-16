#!/usr/bin/env python3
# -- coding: utf-8 --
"""
ESP32C6 BLE Recorder — Audio + RGB LED/Button + (optional) Camera Pose (MediaPipe)

- RGB LED on BCM23/24/25 (R/G/B) with clear status colors.
- ONE start/confirm/stop button on BCM12 (to GND; pull-up disabled to match demo v2 behavior).
- Idle = solid Blue, Searching = Yellow blink, Connected-wait = solid Green,
  Recording = solid Red (on first audio), Errors = solid distinct colors.
- Battery low (2.0 V) = solid Violet override for non-error states.
- Optional camera + MediaPipe Pose tracking (logs JSONL, optional preview window).
- Graceful if GPIO or Camera libs are missing.

NOTE: Button semantics updated to match the simple toggle demo:
- Count a "press" on ANY state change (HIGH<->LOW), with a simple debounce sleep.
- No re-check after debounce and no explicit release gating.
"""

import asyncio # async event loop for BLE + tasks
import time # timestamps, sleeps
import os
import json # manifest + JSONL logging
import traceback
import threading # camera worker in background thread

import numpy as np
from bleak import BleakScanner, BleakClient
from scipy.io import wavfile

# ---------------- Optional Camera / Pose ----------------
CAMERA_ENABLED = True  # set False to disable camera explicitly
SHOW_PREVIEW = False    # set False to disable on-screen preview window

try:
    import cv2
    from picamera2 import Picamera2
    from libcamera import controls
    import mediapipe as mp
    CAMERA_AVAILABLE = True
except Exception as e:
    CAMERA_AVAILABLE = False
    if CAMERA_ENABLED:
        print(f"[CAMERA] Deaktiviert (Libs fehlen?): {e}")

# ---------------- GPIO (optional) ----------------
USE_GPIO = True
BTN_PIN = 12   # BCM pin — Taster nach GND

h = None  # lgpio handle
try:
    import lgpio  # sudo apt install python3-lgpio
except Exception:
    USE_GPIO = False
    lgpio = None

# ---------------- Einstellungen ----------------
ESP32_NAME = "ESP32C6-PARO" # erwarteter Gerätename beim Scan
CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
BLOCK_SIZE = 1600           # 100 ms @ 16 kHz
BYTES_PER_SAMPLE = 2        # int16
BASE_DIR = "/home/firas-mannai/Downloads/Muslim Messungen"
ADAPTER = None
SCAN_TIMEOUT = 10.0  # Sekunden pro Scan-Durchlauf

APP_VERSION = "2.0.6"

# ---------------- Session folder & files  ----------------
def neuer_messordner(base_dir=BASE_DIR):
    os.makedirs(base_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(base_dir, stamp)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return folder
    i = 1
    while True:
        candidate = os.path.join(base_dir, f"{stamp}_{i:02d}")
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        i += 1

# Globals set AFTER confirmation (2nd press)
messordner = None
Gesamt_raw_filename = None
Gesamt_dB_filename = None
Strom_Spannung_Leistung_filename = None
Paro_raw_filename = None
Paro_dB_filename = None
Pose_jsonl_path = None
Manifest_path = None

Gesamt_raw_file = None
db_file = None
strom_file = None
paro_raw_file = None
paro_db_file = None

def open_session_files():
    """Create session folder and open all files. Call ONLY once per program run, after user confirmation."""
    global messordner
    global Gesamt_raw_filename, Gesamt_dB_filename, Strom_Spannung_Leistung_filename
    global Paro_raw_filename, Paro_dB_filename, Pose_jsonl_path, Manifest_path
    global Gesamt_raw_file, db_file, strom_file, paro_raw_file, paro_db_file

    # ---> IMPORTANT: only create/open if no session exists yet
    if messordner is not None:
        return

    messordner = neuer_messordner()

    Gesamt_raw_filename = os.path.join(messordner, "Gesamt_raw_daten.raw")
    Gesamt_dB_filename = os.path.join(messordner, "Gesamt_dB_daten.csv")
    Strom_Spannung_Leistung_filename = os.path.join(messordner, "Strom_Spannung_Leistung.csv")
    Paro_raw_filename = os.path.join(messordner, "Paro_raw_daten.raw")
    Paro_dB_filename = os.path.join(messordner, "Paro_dB_daten.csv")
    Pose_jsonl_path = os.path.join(messordner, f"pose_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
    Manifest_path = os.path.join(messordner, "manifest.json")

    Gesamt_raw_file = open(Gesamt_raw_filename, "ab")

    db_file = open(Gesamt_dB_filename, "a", buffering=1)
    db_file.write("Datum; Uhrzeit; dB\n"); db_file.flush(); os.fsync(db_file.fileno())

    strom_file = open(Strom_Spannung_Leistung_filename, "a", buffering=1)
    strom_file.write("Datum; Uhrzeit; Strom (mA); Spannung (V); Leistung (mW)\n"); strom_file.flush(); os.fsync(strom_file.fileno())

    paro_raw_file = open(Paro_raw_filename, "ab", buffering=0)
    paro_db_file = open(Paro_dB_filename, "a", buffering=1)
    paro_db_file.write("Datum; Uhrzeit; dB\n"); paro_db_file.flush(); os.fsync(paro_db_file.fileno())

def close_session_files():
    for f in (Gesamt_raw_file, db_file, strom_file, paro_raw_file, paro_db_file):
        try:
            if f: f.close()
        except Exception:
            pass

# ---------------- Manifest (lazy write) ----------------
auto_resume_count = 0  # ONLY this counter

_manifest = {
    "created_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "start_time": None,
    "end_time": None,
    "app_version": APP_VERSION,
    "esp32": {"name": ESP32_NAME, "mac": None},
    "audio": {"block_size": BLOCK_SIZE, "bytes_per_sample": BYTES_PER_SAMPLE, "sr": 16000},
    "camera": {
        "enabled": bool(CAMERA_ENABLED and CAMERA_AVAILABLE),
        "show_preview": bool(SHOW_PREVIEW),
        "resolution": [640, 480],
        "fps": 15,
        "exposure_us": 10000
    },
    "stats": {
        "auto_resumes": 0
    }
}

def save_manifest(update=None):
    """Writes manifest ONLY if Manifest_path exists (i.e., session started)."""
    if update:
        for k, v in update.items():
            _manifest[k] = v

    _manifest["stats"]["auto_resumes"] = auto_resume_count

    if not Manifest_path:
        return  # no session yet -> do nothing
    try:
        with open(Manifest_path, "w") as f:
            json.dump(_manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass

# ======================================================================
#                           RGB LED HELPERS (fixed cancel)
# ======================================================================
LED_R, LED_G, LED_B = 23, 24, 25
LED_ACTIVE_HIGH = True  # True for common-cathode; set False for common-anode.

def _gpio_setup():
    global h, USE_GPIO
    if not USE_GPIO:
        print("[GPIO] lgpio nicht verfügbar → LED/Taster deaktiviert.")
        return
    try:
        h = lgpio.gpiochip_open(0)
        for pin in (LED_R, LED_G, LED_B):
            lgpio.gpio_claim_output(h, pin)
            lgpio.gpio_write(h, pin, (0 if LED_ACTIVE_HIGH else 1))  # OFF
        lgpio.gpio_claim_input(h, BTN_PIN)
    except Exception as e:
        print(f"[GPIO] Initialisierung fehlgeschlagen: {e} → GPIO deaktiviert.")
        USE_GPIO = False

def _pin_write(pin, on: bool):
    if not USE_GPIO or h is None:
        return
    val = 1 if on else 0
    if not LED_ACTIVE_HIGH:
        val ^= 1
    try:
        lgpio.gpio_write(h, pin, val)
    except Exception:
        pass
#alle led Kanele aus
def rgb_off():
    _pin_write(LED_R, False); _pin_write(LED_G, False); _pin_write(LED_B, False)

def rgb_set_tuple(rgb):
    # components may be floats 0..1; threshold at >0.5
    r, g, b = (bool(round(min(1, max(0, v)))) for v in rgb)
    _pin_write(LED_R, r); _pin_write(LED_G, g); _pin_write(LED_B, b)

# Blink manager with clean cancel (no final 'off' race)
_rgb_tasks = {}  # name -> asyncio.Task

async def _blink_task(name, rgb, period):
    try:
        while True:
            rgb_set_tuple(rgb); await asyncio.sleep(period)
            rgb_off();          await asyncio.sleep(period)
    except asyncio.CancelledError:
        return

def rgb_blink_start(name, rgb, period):
    rgb_blink_stop(name)
    task = asyncio.create_task(_blink_task(name, rgb, period))
    _rgb_tasks[name] = task

def rgb_blink_stop(name):
    task = _rgb_tasks.pop(name, None)
    if task and not task.done():
        task.cancel()

def rgb_stop_all():
    for k in list(_rgb_tasks.keys()):
        rgb_blink_stop(k)

STATUS_COLORS = {
    "idle":             {"rgb": (0,0,1),   "blink": None},   # Blue solid
    "searching":        {"rgb": (1,1,0),   "blink": 0.25},   # Yellow blink
    "connected_wait":   {"rgb": (0,1,0),   "blink": None},   # Green solid (await 2nd press)
    "recording":        {"rgb": (1,0,0),   "blink": None},   # Red solid
    "err_camera":       {"rgb": (1,0,1),   "blink": None},   # Magenta solid
    "err_notfound":     {"rgb": (1,1,1),   "blink": None},   # White solid
    "err_lost":         {"rgb": (1,1,0),   "blink": None},   # Orange/Yellow solid
    "err_general":      {"rgb": (0,1,1),   "blink": None},   # Cyan solid
    "battery_low":      {"rgb": (0.5,0,1), "blink": None},   # Violet solid
}

_current_status = None
_battery_low = False

def _gpio_read_btn():
    #Liest aktuellen Button-Zustand (1/0). Bei Fehler → 1 (keine Taste)
    if USE_GPIO and h is not None:
        try:
            return lgpio.gpio_read(h, BTN_PIN)
        except Exception:
            return 1
    return 1

def set_battery_low(flag: bool):
    #Setzt Battery-Low-Flag und aktualisiert Statusanzeige (Violet-Override)
    global _battery_low
    _battery_low = bool(flag)
    if _current_status:
        set_status(_current_status)  # refresh with override
#led status
def set_status(name: str):
    """Stop blinks first, then set LED. Battery-low overrides non-errors."""
    global _current_status
    _current_status = name
    if not USE_GPIO:
        return

    rgb_stop_all()  # stop previous blinkers

    # battery-low override for non-error states
    is_error = name.startswith("err_")
    if _battery_low and not is_error:
        rgb_set_tuple(STATUS_COLORS["battery_low"]["rgb"])
        return

    cfg = STATUS_COLORS.get(name)
    if not cfg:
        rgb_off()
        return
    if cfg["blink"]:
        rgb_blink_start(f"blink_{name}", cfg["rgb"], cfg["blink"])
    else:
        rgb_set_tuple(cfg["rgb"])

# ======================================================================
#                           AUDIO / DB
# ======================================================================
audio_samples = []
current_block = 0
first_audio_seen = False

def berechne_dezibel(samples):
    samples = np.array(samples, dtype=np.int16)
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    return 20 * np.log10(rms / 32767.0) if rms != 0 else -100.0

# ======================================================================
#                       Camera Pose Worker
# ======================================================================
class CameraPoseWorker(threading.Thread):
    def __init__(self, start_time, session_folder, show_window=True):
        super().__init__(daemon=True)
        self.t0 = start_time
        self.session_folder = session_folder
        self.show_window = bool(show_window and CAMERA_ENABLED and CAMERA_AVAILABLE)
        self.stop_evt = threading.Event()
        self.picam = None
        self.pose = None
        self.json_file = None

    def run(self):
        if not (CAMERA_ENABLED and CAMERA_AVAILABLE):
            return
        try:
            # jsonl path is created only after session starts
            self.json_file = open(Pose_jsonl_path, "w", buffering=1)

            self.picam = Picamera2()
            config = self.picam.create_video_configuration(
                main={"size": (640, 480)},
                controls={"FrameRate": 15, "ExposureTime": 10000, "AwbMode": controls.AwbModeEnum.Auto}
            )
            self.picam.configure(config)
            self.picam.start()
            #MediaPpe Pose intialisieren
            mp_pose = mp.solutions.pose
            mp_draw = mp.solutions.drawing_utils
            mp_style = mp.solutions.drawing_styles
            self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            while not self.stop_evt.is_set():
                frame_start = time.time()

                frame = self.picam.capture_array()
                if frame is None:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                frame = frame.astype(np.uint8)
                frame = cv2.flip(frame, 1)

                frame.flags.writeable = False
                res = self.pose.process(frame)
                frame.flags.writeable = True

                if res.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        res.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
                    )
                    landmarks = [{
                        "id": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    } for i, lm in enumerate(res.pose_landmarks.landmark)]
                    now_abs = time.time()
                    log = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_abs)),
                        "t_rel": round(now_abs - self.t0, 3),
                        "landmarks": landmarks
                    }
                    self.json_file.write(json.dumps(log) + "\n")

                if self.show_window:
                    fps = 1.0 / max(1e-6, (time.time() - frame_start))
                    cv2.putText(
                        frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    cv2.imshow("PARO Pose", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        self.stop_evt.set()
        except Exception as e:
            print("[CAMERA] Fehler:", e)
            set_status("err_camera")
            traceback.print_exc()
        finally:
            try:
                if self.pose:
                    self.pose.close()
            except Exception:
                pass
            try:
                if self.picam:
                    self.picam.stop()
            except Exception:
                pass
            try:
                if self.json_file:
                    # ensure final flush
                    self.json_file.flush()
                    os.fsync(self.json_file.fileno())
                    self.json_file.close()
            except Exception:
                pass
            try:
                if CAMERA_AVAILABLE:
                    cv2.destroyAllWindows()
            except Exception:
                pass
#Stop-Signal für den Thread setzen (graceful exit)
    def stop(self):
        self.stop_evt.set()

# ======================================================================
#                 BLE Notification Handler + Battery Low
# ======================================================================
BATTERY_LOW_VOLTAGE = 2.0  # per your request
#Verarbeitet BLE-Notifications (Telemetry-Text ODER Audio-Bytes)
def notification_handler(sender, data):
    global audio_samples, current_block, first_audio_seen
    datum = time.strftime("%d.%m.%Y"); uhrzeit = time.strftime("%H:%M:%S")

    # Power telemetry (STROM: <mA>;<V>;<mW>)
    try:
        text = data.decode()
        if text.startswith("STROM:"):
            parts = text[6:].strip().split(";")
            if len(parts) == 3:
                strom, spannung, leistung = parts
                if strom_file:
                    strom_file.write(f"{datum}; {uhrzeit}; {strom}; {spannung}; {leistung}\n")
                    strom_file.flush(); os.fsync(strom_file.fileno())
                try:
                    v = float(spannung.strip().replace(",", "."))
                    set_battery_low(v < BATTERY_LOW_VOLTAGE)
                except Exception:
                    pass
            return
    except UnicodeDecodeError:
        pass

    # First audio packet → switch to RECORDING status
    if not first_audio_seen:
        first_audio_seen = True
        set_status("recording")
        print("[STATE] RECORDING started (first audio packet).")

    # Audio accumulation
    samples = np.frombuffer(data, dtype="<i2")
    audio_samples.extend(samples)
    if Gesamt_raw_file:
        Gesamt_raw_file.write(data)
        # ensure raw hits disk regularly
        Gesamt_raw_file.flush(); os.fsync(Gesamt_raw_file.fileno())

    while len(audio_samples) >= BLOCK_SIZE:
        block = np.array(audio_samples[:BLOCK_SIZE], dtype=np.int16)
        db_wert = berechne_dezibel(block)

        if db_file:
            db_file.write(f"{datum}; {uhrzeit}; {db_wert:.2f}\n"); db_file.flush(); os.fsync(db_file.fileno())

        if db_wert > -10:
            if paro_raw_file:
                paro_raw_file.write(block.tobytes())
            if paro_db_file:
                paro_db_file.write(f"{datum}; {uhrzeit}; {db_wert:.2f}\n")
        else:
            silent_block = np.zeros_like(block, dtype=np.int16)
            if paro_raw_file:
                paro_raw_file.write(silent_block.tobytes())
            if paro_db_file:
                paro_db_file.write(f"{datum}; {uhrzeit}; 0.00\n")

        if paro_raw_file:
            paro_raw_file.flush(); os.fsync(paro_raw_file.fileno())
        if paro_db_file:
            paro_db_file.flush(); os.fsync(paro_db_file.fileno())

        audio_samples = audio_samples[BLOCK_SIZE:]
        current_block += 1
        if current_block % 10 == 0:
            print(f"[AUDIO] Block {current_block} dB: {db_wert:.2f}")

# ======================================================================
#                     Button: ANY-EDGE TOGGLE (like demo #2)
# ======================================================================
async def wait_for_button_toggle():
    """
    Wait for ANY state change on BTN_PIN (HIGH<->LOW), then return once.
    Simple debounce like the demo: sleep and accept; no re-check, no release gating.
    """
    if not USE_GPIO:
        return
    debounce_sec = 0.20
    poll = 0.02
    last = _gpio_read_btn()
    while True:
        cur = _gpio_read_btn()
        if cur != last:
            time.sleep(debounce_sec)  # simple debounce (no re-check)
            return
        last = cur
        await asyncio.sleep(poll)

async def scanning_indicator(stop_event: asyncio.Event):
    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0; t0 = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - t0)
        msg = f"\nSuche nach {ESP32_NAME} … {spinner[i % len(spinner)]}  ({elapsed}s)"
        print(msg, end="", flush=True)
        i += 1; await asyncio.sleep(0.15)
    print("\n" + " " * 60 + "\n", end="", flush=True)

# ======================================================================
#                               Main
# ======================================================================
# Auto-resume: after first confirmed start, reconnects auto-start without confirmation.
AUTO_RESUME = False

async def main():
    global AUTO_RESUME, first_audio_seen, audio_samples, current_block
    global messordner, auto_resume_count
    _gpio_setup()

    # Idle (solid blue)
    set_status("idle")
    print("[STATE] IDLE. Drücke die Taste (BCM12→GND), um die Suche zu starten …")
    await wait_for_button_toggle()  # first user action

    # Searching (yellow blink)
    set_status("searching")
    print("[STATE] SEARCHING…  (Press button again to exit)")

    cam_worker = None
    try:
        # allow exit while searching
        user_abort_search = False
        while True:
            stop_scan = asyncio.Event()
            indicator_task = asyncio.create_task(scanning_indicator(stop_scan))

            # parallel: scan vs cancel-by-button
            cancel_task = asyncio.create_task(wait_for_button_toggle())
            scan_task = asyncio.create_task(BleakScanner.discover(timeout=SCAN_TIMEOUT, adapter=ADAPTER))

            done, pending = await asyncio.wait(
                {scan_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED
            )

            if cancel_task in done:
                stop_scan.set(); await indicator_task
                for t in pending: t.cancel()
                print("[INPUT] Suche abgebrochen per Taste. Beende …")
                user_abort_search = True
                break

            cancel_task.cancel()
            stop_scan.set(); await indicator_task

            target = None
            try:
                devices = scan_task.result()
                target = next((d for d in devices if d.name and ESP32_NAME in d.name), None)
            except Exception as e:
                print(f"[SCAN] Fehler: {e} → retry in 3s")
                set_status("err_general"); await asyncio.sleep(3)
                set_status("searching")
                continue

            if user_abort_search:
                break

            if not target:
                print(f"[SCAN] {ESP32_NAME} nicht gefunden!")
                set_status("err_notfound")
                await asyncio.sleep(2.0)
                set_status("searching")
                await asyncio.sleep(3.0)
                continue

            print(f"[SCAN] Gefunden: {target.name} ({target.address})")
            _manifest["esp32"]["mac"] = getattr(target, 'address', None)

            # Green while waiting/auto-resume start
            set_status("connected_wait")  # LED Grün
            print("[STATE] CONNECTED " + ("(auto-resume) starting…" if AUTO_RESUME else "(waiting for confirmation). Press button again to start."))

            # --- Connect BLE (do not start notifications yet) ---
            client = None
            user_stopped = False
            try:
                client = BleakClient(
                    target.address,
                    timeout=60.0,
                    adapter=ADAPTER,
                    disconnected_callback=lambda _: (print("\n[BLE] Getrennt."), set_status("err_lost"))
                )
                await client.__aenter__()
                if not AUTO_RESUME:
                    print("[BLE] Verbunden. Warte auf 2. Tastendruck zur Bestätigung …")
                    set_status("connected_wait")  # LED Grün

                # SECOND CONFIRMATION (or skip if AUTO_RESUME)
                if not AUTO_RESUME:
                    await wait_for_button_toggle()
                    print("[INPUT] Bestätigung erhalten. Starte Session-Dateien & Notifications …")
                    AUTO_RESUME = True
                else:
                    print("[INPUT] Auto-Resume aktiv: starte Session-Dateien & Notifications …")
                    if messordner is not None:
                        auto_resume_count += 1
                        save_manifest()

                # ---- Create folder + open files ONLY ON FIRST START ----
                if messordner is None:
                    open_session_files()
                    print(f"Alle Daten werden gespeichert in: {messordner}/")
                    # Start manifest timing (only once)
                    if _manifest.get("start_time") is None:
                        t0 = time.time()
                        _manifest["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))
                        save_manifest()
                else:
                    print(f"[SESSION] Wiederaufnahme in bestehendem Ordner: {messordner}")

                # Reset per-connection state so LED turns red only on first audio of this connection
                first_audio_seen = False
                audio_samples = []
                current_block = 0

                # Start notifications
                await client.start_notify(CHAR_UUID, notification_handler)
                print("[BLE] Notifications gestartet.")

                # Start camera (optional) — pose file path exists now (created at first start)
                if CAMERA_ENABLED and CAMERA_AVAILABLE:
                    if cam_worker is None or not cam_worker.is_alive():
                        cam_worker = CameraPoseWorker(start_time=time.time(), session_folder=messordner, show_window=SHOW_PREVIEW)
                        cam_worker.start()
                        print("[CAMERA] Pose worker gestartet.")

                print(
                    "Empfange Daten in:\n"
                    f"  {Gesamt_raw_filename}\n"
                    f"  {Paro_raw_filename}\n"
                    f"  {Gesamt_dB_filename}\n"
                    f"  {Paro_dB_filename}\n"
                    f"  {Strom_Spannung_Leistung_filename}\n"
                    f"  {Pose_jsonl_path if CAMERA_ENABLED and CAMERA_AVAILABLE else '(Kamera aus)'}\n"
                    "(Dritte Taste = Stop)"
                )

                # THIRD TOGGLE to stop
                async def stop_on_button_toggle_after_confirm():
                    debounce_sec = 0.20; poll = 0.02
                    last = _gpio_read_btn()
                    while True:
                        cur = _gpio_read_btn()
                        if cur != last:
                            time.sleep(debounce_sec)
                            return True
                        last = cur
                        await asyncio.sleep(poll)

                stop_task = asyncio.create_task(stop_on_button_toggle_after_confirm())

                # Loop until stop or disconnect
                while client.is_connected:
                    if stop_task.done():
                        print("[INPUT] Stop-Taste erkannt (toggle).")
                        user_stopped = True
                        AUTO_RESUME = False  # explicit stop disables auto-resume next time
                        break
                    await asyncio.sleep(0.2)

            except Exception as e:
                print(f"[BLE] Verbindungs-/Laufzeitfehler: {e}")
                set_status("err_general")
                await asyncio.sleep(1.5)
                set_status("searching")
                continue
            finally:
                try:
                    if client is not None:
                        try:
                            await client.stop_notify(CHAR_UUID)
                        except Exception:
                            pass
                        await client.__aexit__(None, None, None)
                except Exception:
                    pass

                if user_stopped:
                    print("[STATE] User stop requested. Exiting main loop.")
                    break
                else:
                    print("[STATE] Disconnected or error; returning to SEARCH.")
                    set_status("searching")
                    continue

        # If user aborted search, fall through to cleanup (no files were created)
        if user_abort_search:
            return

    except KeyboardInterrupt:
        print("\n[EXIT] Beende auf Nutzerwunsch …")
    finally:
        # Stop camera
        try:
            if 'cam_worker' in locals() and cam_worker is not None:
                cam_worker.stop(); cam_worker.join(timeout=2.0)
        except Exception:
            pass

        # End time in manifest (only if session started)
        if messordner:
            _manifest["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_manifest()

        # Close files (only if session started)
        close_session_files()
        print("[CLOSE] Alle Dateien wurden sauber geschlossen." if messordner else "[CLOSE] Keine Session-Dateien zu schließen.")

        # GPIO tidy
        try:
            rgb_off()
            if USE_GPIO and h is not None:
                lgpio.gpiochip_close(h)
        except Exception:
            pass

        # ---------- Nachbearbeitung: RAW → WAV (only if session started) ----------
        try:
            if messordner and Paro_raw_filename and os.path.exists(Paro_raw_filename):
                raw_paro = np.fromfile(Paro_raw_filename, dtype=np.int16)
                sr = 16000
                wav_paro = os.path.join(messordner, "Paro_raw_daten.wav")
                wavfile.write(wav_paro, sr, raw_paro)
                print(f"[WAV] erstellt:\n - {wav_paro}")
            if messordner and Gesamt_raw_filename and os.path.exists(Gesamt_raw_filename):
                raw_gesamt = np.fromfile(Gesamt_raw_filename, dtype=np.int16)
                sr = 16000
                wav_gesamt = os.path.join(messordner, "Gesamt_raw_daten.wav")
                wavfile.write(wav_gesamt, sr, raw_gesamt)
                print(f"[WAV] erstellt:\n - {wav_gesamt}")
        except Exception as e:
            print(f"[WAV] Nachbearbeitung fehlgeschlagen: {e}")

if __name__ == "__main__":
    asyncio.run(main())
