"""Signal sources for the nap rig. Each keeps a ring buffer of TP9, AF7, AF8, TP10 in uV plus
an accelerometer vector, and exposes window(seconds) -> (eeg, eog, fs, contact_ok)."""
import asyncio
import collections
import socket
import struct
import threading
import time

import numpy as np

FS = 256.0
MUSE_CONTROL = "273e0001-4c4d-454d-96be-f03bac821358"
MUSE_EEG = ("273e0003-4c4d-454d-96be-f03bac821358", "273e0004-4c4d-454d-96be-f03bac821358",
            "273e0005-4c4d-454d-96be-f03bac821358", "273e0006-4c4d-454d-96be-f03bac821358")
MUSE_ACCEL = "273e000a-4c4d-454d-96be-f03bac821358"
MUSE_PRESETS = ("p21", "p20", "p50", "p51", "p1034", "p1035", "p1031")
MUSE_SCALE = 1650.0 / 4095.0
MUSE_ACCEL_SCALE = 0.0000610352


def derive(rows):
    """Muse montage to the two validated signals. rows: (n, 4) TP9, AF7, AF8, TP10."""
    tp9, af7, af8, tp10 = rows.T
    eeg = (af7 + af8) / 2 - (tp9 + tp10) / 2     # frontal EEG against the mastoids
    eog = af7 - af8                               # horizontal EOG across the forehead
    return eeg, eog


def contact_ok(rows, fs):
    """MuseCar's contact rule per 1-s block: an electrode is off if flat, railed or over 300 uV sd.
    The window counts only if AF7, AF8 and one TP electrode are on in at least 80% of blocks."""
    n = int(fs)
    blocks = [rows[i:i + n] for i in range(0, len(rows) - n + 1, n)]
    if not blocks:
        return False
    good = 0
    for b in blocks:
        sd, mu = b.std(0), b.mean(0)
        on = ~((mu < 30) | (mu > 1620) | (sd < 0.5) | (sd > 300))
        good += bool(on[1] and on[2] and (on[0] or on[3]))
    return good / len(blocks) >= 0.8


class Buffer:
    def __init__(self, fs=FS, seconds=120):
        self.fs = fs
        self.lock = threading.Lock()
        self.rows = collections.deque(maxlen=int(fs * seconds))
        self.extra = collections.deque(maxlen=int(fs * seconds))   # (aux, ax, ay, az) per sample
        self.accel = None
        self.aux = None             # fifth electrode, when the stream carries one
        self.last = None
        self.sink = None            # callable(t, row, accel) for raw logging
        self.count = 0              # samples received, for the rate check

    def resize(self, fs, seconds=120):
        with self.lock:
            self.fs = fs
            self.rows = collections.deque(maxlen=int(fs * seconds))
            self.extra = collections.deque(maxlen=int(fs * seconds))

    def add(self, t, row):
        acc = self.accel
        with self.lock:
            self.rows.append(row)
            self.extra.append((np.nan if self.aux is None else self.aux,
                               *(acc if acc is not None else (np.nan, np.nan, np.nan))))
            self.last = t
            self.count += 1
        if self.sink:
            self.sink(t, row, self.accel)

    def window(self, seconds):
        with self.lock:
            n = int(self.fs * seconds)
            if len(self.rows) < n:
                return None
            rows = np.array(list(self.rows)[-n:], dtype=float)
        eeg, eog = derive(rows)
        return eeg, eog, self.fs, contact_ok(rows, self.fs)

    def raw(self, seconds):
        """The last `seconds` of samples underived: (rows (n, 4) TP9 AF7 AF8 TP10, aux (n,), acc (n, 3)),
        nan where the stream had no fifth electrode or accelerometer; None until the buffer fills."""
        with self.lock:
            n = int(self.fs * seconds)
            if len(self.rows) < n:
                return None
            rows = np.array(list(self.rows)[-n:], dtype=float)
            extra = np.array(list(self.extra)[-n:], dtype=float)
        return rows, extra[:, 0], extra[:, 1:4]


class MuseBLE(threading.Thread):
    """Muse over the PC's Bluetooth, no phone app. Protocol and the Windows unpair fix come from
    MuseCar's muse_drive.py."""

    def __init__(self, buf, name="", log=print):
        super().__init__(daemon=True)
        self.buf, self.name, self.log = buf, name.lower(), log
        self.connected = False

    def run(self):
        asyncio.run(self.loop())

    async def loop(self):
        from bleak import BleakClient, BleakScanner
        unpaired = False
        while True:
            found = await BleakScanner.discover(timeout=5.0, return_adv=True)
            muses = sorted(((a.rssi, d) for d, a in found.values()
                            if "muse" in (d.name or "").lower() and self.name in (d.name or "").lower()),
                           key=lambda p: -p[0])
            if not muses:
                await asyncio.sleep(1.0)
                continue
            dev = muses[0][1]
            if not unpaired:
                unpaired = True
                try:
                    await BleakClient(dev).unpair()
                except Exception:
                    pass
            gone = asyncio.Event()
            try:
                async with BleakClient(dev, disconnected_callback=lambda _: gone.set()) as link:
                    await self.stream(link, gone)
            except Exception as e:  # bleak raises many kinds; retry
                self.log(f"Muse: {type(e).__name__} {e}; retrying")
            self.connected = False
            await asyncio.sleep(1.0)

    async def stream(self, link, gone):
        pending = {}

        def handler(el):
            def on(_s, data):
                c = (data[0] << 8) | data[1]
                smp = []
                for k in range(2, 20, 3):
                    a, b, d = data[k], data[k + 1], data[k + 2]
                    smp += [(a << 4) | (b >> 4), ((b & 0x0F) << 8) | d]
                p = pending.setdefault(c, {})
                p[el] = smp
                if len(p) < 4:
                    return
                del pending[c]
                for old in [x for x in pending if (c - x) % 65536 > 50]:
                    del pending[old]
                now = time.time()
                for i in range(12):
                    self.buf.add(now, [p[e][i] * MUSE_SCALE for e in range(4)])
            return on

        for el, uuid in enumerate(MUSE_EEG):
            await link.start_notify(uuid, handler(el))

        def on_acc(_s, data):
            if len(data) < 20:
                return
            v = struct.unpack(">H9h", bytes(data[:20]))[1:]
            self.buf.accel = [sum(v[k::3]) / 3.0 * MUSE_ACCEL_SCALE for k in range(3)]
        try:
            await link.start_notify(MUSE_ACCEL, on_acc)
        except Exception:
            self.log("Muse: no accelerometer, so the nod trigger is off")

        async def cmd(text):
            body = text.encode() + b"\n"
            await link.write_gatt_char(MUSE_CONTROL, bytes([len(body)]) + body, response=False)

        preset = MUSE_PRESETS[0]
        for preset in MUSE_PRESETS:
            last = self.buf.last
            await cmd("h")
            await cmd(preset)
            await cmd("d")
            await asyncio.sleep(3.0)
            if self.buf.last != last and self.buf.accel is not None:
                break
        self.connected = True
        self.log(f"Muse connected (preset {preset})")
        while not gone.is_set():
            await cmd("k")
            try:
                await asyncio.wait_for(gone.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


class MuseOSC(threading.Thread):
    """MuseLog on the phone streaming OSC to this PC: /muse/eeg (turn on Full-rate raw EEG)
    and /muse/acc."""

    def __init__(self, buf, port=5000, log=print):
        super().__init__(daemon=True)
        from pythonosc import osc_bundle, osc_message
        self.ob, self.om = osc_bundle, osc_message
        self.buf, self.log = buf, log
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind(("0.0.0.0", port))
        self.connected = False

    def run(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except OSError:
                continue
            try:
                if self.ob.OscBundle.dgram_is_bundle(data):
                    msgs = [m for m in self.ob.OscBundle(data) if isinstance(m, self.om.OscMessage)]
                else:
                    msgs = [self.om.OscMessage(data)]
            except Exception:
                continue
            now = time.time()
            for m in msgs:
                v = list(m.params)
                if m.address in ("/muse/eeg", "/eeg") and len(v) >= 4:
                    # a fifth value is the AUX electrode (a cup at Oz on this rig): kept for the
                    # raw log, not used by the detector, which was validated on four channels
                    self.buf.aux = v[4] if len(v) >= 5 else None
                    self.buf.add(now, v[:4])
                    self.connected = True
                elif m.address == "/muse/acc" and len(v) >= 3:
                    self.buf.accel = v[:3]


class Replay:
    """A Sleep-EDF night played through the rig in simulated time. AF7 and AF8 are built so that
    derive() returns exactly the EDF's Fpz-Cz and horizontal EOG; TP9 and TP10 carry a constant
    800 uV Muse-like offset. Tests the whole rig end to end without a sleeper."""

    def __init__(self, buf, psg, start_s=0.0):
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz", "EOG horizontal"], preload=True)
        self.fs = raw.info["sfreq"]
        self.eeg = raw.get_data(picks=0)[0] * 1e6
        self.eog = raw.get_data(picks=1)[0] * 1e6
        self.start = start_s
        self.i = int(start_s * self.fs)
        buf.resize(self.fs)
        self.buf = buf
        self.connected = True

    def advance_to(self, sim_t):
        """sim_t counts from the start of the session; returns False at the end of the night."""
        j = min(int((self.start + sim_t) * self.fs), len(self.eeg))
        off = 800.0
        noise = np.random.default_rng(self.i).normal(0, 3.0, max(j - self.i, 0))
        for n_, k in enumerate(range(self.i, j)):
            e, o, z = self.eeg[k], self.eog[k], noise[n_]
            # TP9 and TP10 get equal and opposite noise so their mean (the reference) stays flat
            self.buf.add(k / self.fs - self.start,
                         [off + z, off + e + o / 2, off + e - o / 2, off - z])
        self.i = j
        return j < len(self.eeg)
