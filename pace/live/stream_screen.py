"""
stream_screen.py — Windows-side screen streamer for the LIVE coach.

Grabs your Minecraft window (or a screen region) a few times per second, encodes
each frame as a small base64 JPEG, and POSTs it to the coach's /frame endpoint.
The coach (rl_causal/live/coach.py) uses the freshest frame as the companion's
vision input.

Runs on the machine where you PLAY Minecraft (Windows). Standalone — no repo
imports, only: pip install mss pillow requests   (pygetwindow optional, for
--window-title).

Deployment note
---------------
Simplest is to run coach.py on the SAME Windows machine (it reaches the local
bot at :8765 and the Tokyo companion over your existing SSH tunnel to :8001).
Then frames stay local: --coach-url http://127.0.0.1:8770/frame. If the coach
runs on the server instead, point --coach-url at the tunnel that reaches its
frame port.

Run:
    python stream_screen.py --coach-url http://127.0.0.1:8770/frame --fps 1 \
        --window-title Minecraft --max-width 640
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time

import requests

try:
    import mss
    from PIL import Image
except ImportError:
    print("Need: pip install mss pillow requests", file=sys.stderr)
    raise


def _resolve_region(args):
    """Return an mss-style {left, top, width, height} dict, or None for full monitor."""
    if args.window_title:
        try:
            import pygetwindow as gw
            wins = [w for w in gw.getAllWindows()
                    if args.window_title.lower() in (w.title or "").lower()]
            if wins:
                w = wins[0]
                return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
            print(f"[stream] window '{args.window_title}' not found; using monitor",
                  file=sys.stderr)
        except ImportError:
            print("[stream] pygetwindow not installed; --window-title ignored "
                  "(pip install pygetwindow)", file=sys.stderr)
    if args.region:
        x, y, w, h = (int(v) for v in args.region.split(","))
        return {"left": x, "top": y, "width": w, "height": h}
    return None  # full monitor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coach-url", default="http://127.0.0.1:8770/frame",
                    help="the coach's POST /frame endpoint")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-width", type=int, default=640,
                    help="downscale frames to this width (keeps payload small)")
    ap.add_argument("--quality", type=int, default=60, help="JPEG quality")
    ap.add_argument("--monitor", type=int, default=1, help="mss monitor index")
    ap.add_argument("--region", default=None, help="x,y,w,h to capture a fixed area")
    ap.add_argument("--window-title", default=None,
                    help="capture the window whose title contains this (needs pygetwindow)")
    args = ap.parse_args()

    interval = 1.0 / max(0.1, args.fps)
    sess = requests.Session()
    n_ok = n_err = 0

    with mss.mss() as sct:
        print(f"[stream] -> {args.coach_url}  fps={args.fps}  max_width={args.max_width}",
              flush=True)
        while True:
            t0 = time.time()
            try:
                region = _resolve_region(args) or sct.monitors[args.monitor]
                shot = sct.grab(region)
                img = Image.frombytes("RGB", shot.size, shot.rgb)
                if img.width > args.max_width:
                    h = int(img.height * args.max_width / img.width)
                    img = img.resize((args.max_width, h))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=args.quality)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                r = sess.post(args.coach_url, data=b64.encode("ascii"), timeout=4)
                n_ok += int(r.ok)
                n_err += int(not r.ok)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                if n_err % 10 == 1:
                    print(f"[stream] send failed ({e}); is the coach up?", flush=True)
            if (n_ok + n_err) % 20 == 0:
                print(f"[stream] sent ok={n_ok} err={n_err}", flush=True)
            dt = time.time() - t0
            time.sleep(max(0.0, interval - dt))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[stream] stopped.")
