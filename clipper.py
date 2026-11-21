#!/usr/bin/env python3
"""Stream clipper: chat spikes + audio loudness + gameplay motion -> captioned clips.

Works on VODs / finished streams (chat replay must be available).
Usage:  python clipper.py "<url>" --hype 5 --key 3 --captions
"""
import argparse, json, re, subprocess, wave
from pathlib import Path
import numpy as np

# Chat messages containing these count 3x (hype words, emotes, emoji)
HYPE = re.compile(r"\b(pog\w*|lol|lmao|omg|wow|insane|gg|wtf|clutch|nooo+|let'?s go+|w)\b|[😂🤣🔥😱💀]", re.I)


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def download(url, out):
    sh(["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]", "--merge-output-format", "mp4",
        "-o", str(out / "source.%(ext)s"), url])
    return next(p for p in out.glob("source.*") if p.suffix in (".mp4", ".mkv", ".webm"))


def duration(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def fit(a, n):
    out = np.zeros(n)
    out[:min(n, len(a))] = np.asarray(a)[:n]
    return out


# ---------- SIGNAL 1: chat spikes ----------
def chat_signal(url, n):
    """Messages per second (hype messages weigh 3x). None if no chat replay."""
    try:
        from chat_downloader import ChatDownloader
        counts = np.zeros(n)
        for m in ChatDownloader().get_chat(url):
            t = m.get("time_in_seconds")
            if t is None or not (0 <= t < n):
                continue
            counts[int(t)] += 3 if HYPE.search(m.get("message") or "") else 1
        return counts if counts.sum() > 0 else None
    except Exception as e:
        print("Chat not available, using audio + video only:", e)
        return None


# ---------- SIGNAL 2: audio loudness ----------
def audio_signal(video, n, tmp):
    wav = tmp / "audio.wav"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "8000", str(wav)])
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    sr, k = 8000, len(x) // 8000
    rms = np.sqrt((x[:k * sr].reshape(k, sr) ** 2).mean(axis=1))
    wav.unlink()
    return fit(rms, n)


# ---------- SIGNAL 3: gameplay action (frame change, optionally inside a region) ----------
def visual_signal(video, n, roi):
    """roi = (x, y, w, h) as fractions of the frame, e.g. kill feed or scoreboard."""
    W, H = 192, 108
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf",
                          f"fps=1,scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE)
    prev, out = None, []
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        f = np.frombuffer(b, dtype=np.uint8).astype(np.float32).reshape(H, W)
        if roi:
            x, y, w, h = roi
            f = f[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)]
        out.append(0.0 if prev is None else float(np.abs(f - prev).mean()))
        prev = f
    p.wait()
    return fit(out, n)


# ---------- SPIKE DETECTION ----------
def z(a, win=5):
    """Smooth, then robust z-score: how many 'normal deviations' above the typical level."""
    a = np.convolve(a, np.ones(win) / win, mode="same")
    med = np.median(a)
    mad = np.median(np.abs(a - med)) * 1.4826 or a.std() or 1.0
    return np.clip((a - med) / mad, 0, 10)


def pick(score, k, gap, min_score, taken=()):
    """Highest peaks first, at least `gap` seconds apart, skipping already-used windows."""
    s = score.copy()
    for a, b in taken:
        s[max(0, a - gap):b + gap] = -1e9
    peaks = []
    while len(peaks) < k:
        i = int(np.argmax(s))
        if s[i] < min_score:
            break
        peaks.append(i)
        s[max(0, i - gap):i + gap] = -1e9
    return peaks


# ---------- CLIPS + CAPTIONS ----------
def cut(video, start, end, dst, vertical):
    vf = ["-vf", "crop=ih*9/16:ih,scale=720:1280"] if vertical else []
    sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(video), "-t", str(end - start),
        *vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(dst)])


def srt_time(t):
    return f"{int(t // 3600):02}:{int(t % 3600 // 60):02}:{t % 60:06.3f}".replace(".", ",")


def caption(clip, model, lang):
    segs, _ = model.transcribe(str(clip), language=lang, vad_filter=True)
    blocks = [f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{s.text.strip()}\n"
              for i, s in enumerate(segs, 1)]
    if not blocks:
        return clip
    srt = clip.with_suffix(".srt")
    srt.write_text("\n".join(blocks), encoding="utf-8")
    out = clip.with_name(clip.stem + "_cc.mp4")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", clip.name, "-vf",
        f"subtitles={srt.name}:force_style='FontSize=20,Outline=2,Alignment=2'",
        "-c:a", "copy", out.name], cwd=clip.parent)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--hype", type=int, default=5, help="number of hype clips")
    ap.add_argument("--key", type=int, default=3, help="number of calm 'key gameplay' clips")
    ap.add_argument("--pre", type=int, default=20, help="seconds before the peak")
    ap.add_argument("--post", type=int, default=15, help="seconds after the peak")
    ap.add_argument("--roi", default="", help="x,y,w,h fractions, e.g. 0.7,0.05,0.28,0.3 (kill feed)")
    ap.add_argument("--vertical", action="store_true", help="9:16 centre crop for Shorts")
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--whisper", default="base", help="tiny/base/small (small is better for Hindi/Telugu)")
    ap.add_argument("--lang", default=None, help="en, hi, te... (default: auto-detect)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roi = tuple(float(v) for v in a.roi.split(",")) if a.roi else None

    video = download(a.url, out)
    n = int(duration(video))
    chat = chat_signal(a.url, n)
    audio = audio_signal(video, n, out)
    vis = visual_signal(video, n, roi)

    zv, za = z(vis), z(audio)
    hype = (0.5 * z(chat) + 0.2 * za + 0.3 * zv) if chat is not None else (0.5 * za + 0.5 * zv)

    gap = a.pre + a.post
    plan = [("hype", p) for p in pick(hype, a.hype, gap, min_score=2.0)]
    windows = [(max(0, p - a.pre), min(n, p + a.post)) for _, p in plan]
    # Calm-but-important moments: strong gameplay change, but not already covered by a hype clip
    plan += [("key", p) for p in pick(zv, a.key, gap, min_score=1.5, taken=windows)]

    model = None
    if a.captions:
        from faster_whisper import WhisperModel
        model = WhisperModel(a.whisper, device="cpu", compute_type="int8")

    manifest = []
    for i, (kind, p) in enumerate(plan, 1):
        s, e = max(0, p - a.pre), min(n, p + a.post)
        dst = out / f"{i:02}_{kind}_{p}s.mp4"
        cut(video, s, e, dst, a.vertical)
        final = caption(dst, model, a.lang) if model else dst
        manifest.append({"file": final.name, "kind": kind, "peak_sec": p, "start": s, "end": e,
                         "hype_score": round(float(hype[p]), 2), "chat_score": round(float(z(chat)[p]), 2) if chat is not None else None})
        print("made", final.name)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    video.unlink()


if __name__ == "__main__":
    main()
