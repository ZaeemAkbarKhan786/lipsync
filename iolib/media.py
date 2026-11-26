"""FFmpeg-backed media probing and manipulation.

Pure subprocess calls — no torch, no GPU. FFmpeg is invoked as an external binary
(never linked), which keeps its GPL build out of this project's license surface.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

EXTEND_MODES = ("loop", "pingpong", "freeze")

#: Declare BT.709 SDR on anything this module re-encodes.
#:
#: Uploads are tone-mapped to BT.709 at ingest (core.ingest.normalise_video), so
#: this is a statement of fact, not a conversion. It is set explicitly because
#: LatentSync's own writer emits untagged video: a delivered file that merely
#: *happens* to be interpreted correctly by most players is one player away from
#: looking wrong, and an untagged file carrying HDR pixels is exactly the
#: washed-out-colour bug this guards against recurring.
SDR_TAG_ARGS = [
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-color_range", "tv",
]


class MediaError(RuntimeError):
    pass


def _exe(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"{name} not found on PATH; install ffmpeg")
    return path


def run(cmd: list[str], *, quiet: bool = True) -> str:
    """Run a command, raising :class:`MediaError` with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:6])}...\n"
            f"{proc.stderr[-2000:]}"
        )
    return proc.stdout


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    n_frames: int
    has_audio: bool


def ffprobe_json(path: Path) -> dict:
    out = run(
        [
            _exe("ffprobe"), "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    return json.loads(out)


def probe_video(path: Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"video not found: {path}")
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")

    num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if fps <= 0:
        num, _, den = (video.get("r_frame_rate") or "25/1").partition("/")
        fps = float(num) / float(den) if float(den) else 25.0

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    n_frames = int(video.get("nb_frames") or 0) or int(round(duration * fps))

    return VideoInfo(
        path=path,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        duration=duration,
        n_frames=n_frames,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def probe_duration(path: Path) -> float:
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"media not found: {path}")
    data = ffprobe_json(path)
    dur = data.get("format", {}).get("duration")
    if dur is None:
        for stream in data.get("streams", []):
            if stream.get("duration"):
                dur = stream["duration"]
                break
    if dur is None:
        raise MediaError(f"could not determine duration of {path}")
    return float(dur)


def extend_video(
    src: Path,
    dst: Path,
    target_seconds: float,
    *,
    mode: str = "pingpong",
    fps: float | None = None,
) -> Path:
    """Extend ``src`` so it lasts at least ``target_seconds``.

    LatentSync only repaints the mouth inside footage it is given, so the output
    can never be longer than the base clip. This makes the base clip long enough.

    Modes:
      ``loop``     concatenate the clip head-to-tail. Cheapest, but produces a
                   visible jump wherever the last frame meets the first.
      ``pingpong`` concatenate forward+reversed passes so the seam is continuous.
                   Default: for a talking-head clip this reads as natural motion.
      ``freeze``   hold the final frame. Cheapest of all, but the presenter goes
                   visibly still — only sensible for very short overruns.
    """
    src, dst = Path(src), Path(dst)
    if mode not in EXTEND_MODES:
        raise MediaError(f"extend_video mode must be one of {EXTEND_MODES}, got {mode!r}")

    info = probe_video(src)
    fps = fps or info.fps or 25.0
    dst.parent.mkdir(parents=True, exist_ok=True)

    # A little headroom so rounding never leaves us a frame short.
    target = target_seconds + (2.0 / fps)

    if info.duration >= target:
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        return dst

    ff = _exe("ffmpeg")
    common_out = [
        "-an",                      # audio comes from the cloned voice, not the clip
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-r", f"{fps:.6f}",
        "-t", f"{target:.3f}",
    ]

    if mode == "freeze":
        pad = target - info.duration
        run([
            ff, "-y", "-v", "error",
            "-i", str(src),
            "-vf", f"tpad=stop_mode=clone:stop_duration={pad + 1:.3f}",
            *common_out, str(dst),
        ])
        return dst

    if mode == "loop":
        loops = math.ceil(target / max(info.duration, 1e-6))
        run([
            ff, "-y", "-v", "error",
            "-stream_loop", str(loops),
            "-i", str(src),
            *common_out, str(dst),
        ])
        return dst

    # pingpong: forward + reverse is one cycle of 2x duration.
    cycles = math.ceil(target / max(info.duration * 2.0, 1e-6))
    run([
        ff, "-y", "-v", "error",
        "-i", str(src),
        "-filter_complex",
        (
            "[0:v]split=2[fwd][tmp];"
            "[tmp]reverse[rev];"
            "[fwd][rev]concat=n=2:v=1:a=0[pp];"
            f"[pp]loop=loop={cycles}:size=32767:start=0,setpts=N/FRAME_RATE/TB[out]"
        ),
        "-map", "[out]",
        *common_out, str(dst),
    ])
    return dst


def mux(video: Path, audio: Path, dst: Path, *, copy_video: bool = True) -> Path:
    """Combine a video stream with an audio track.

    Deliberately does not pass ``-shortest``. With ``-c:v copy`` ffmpeg can only
    cut the video at a packet boundary, so ``-shortest`` overshoots: a 25.000s
    render muxed against 24.957s of speech came out as 24.800s of video, leaving
    157 ms of audio playing past the last frame at the end of every delivery.

    The two streams are inherently within a frame of each other here - the video
    is generated from the audio - so letting both run to their natural length is
    correct, and :func:`check_av_sync` verifies that rather than assuming it.
    """
    video, audio, dst = Path(video), Path(audio), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Tagged in both branches. The copy branch is the one that matters in
    # practice: assembly skips the HD re-encode for portrait footage that is
    # already tall enough, so a stream copy is what produces the delivered file
    # - and LatentSync's writer emits untagged video. ffmpeg writes the colr
    # atom from these flags even under `-c:v copy`, so this costs nothing and
    # is the difference between a correctly-described delivery and one that
    # relies on every player guessing BT.709.
    vcodec = ["-c:v", "copy", *SDR_TAG_ARGS] if copy_video else [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        *SDR_TAG_ARGS,
    ]
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        *vcodec,
        "-c:a", "aac", "-b:a", "192k",
        str(dst),
    ])
    return dst


def check_av_sync(path: Path, *, tolerance_s: float = 0.040) -> dict:
    """Report A/V timing facts about a finished file, from ffprobe not filenames.

    ``tolerance_s`` bounds the acceptable video/audio duration difference. 40 ms
    is one frame at 25fps; it is a sanity bound on muxing, not a claim about
    perceptual sync - a file can pass this and still look mistimed if the
    generated mouth shapes are wrong.
    """
    path = Path(path)
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    v = [s for s in streams if s.get("codec_type") == "video"]
    a = [s for s in streams if s.get("codec_type") == "audio"]

    def _f(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    vdur = _f(v[0].get("duration")) if v else 0.0
    adur = _f(a[0].get("duration")) if a else 0.0
    report = {
        "path": str(path),
        "video_streams": len(v),
        "audio_streams": len(a),
        "video_duration": round(vdur, 3),
        "audio_duration": round(adur, 3),
        "duration_diff": round(abs(vdur - adur), 3),
        "video_start": round(_f(v[0].get("start_time")) if v else 0.0, 3),
        "audio_start": round(_f(a[0].get("start_time")) if a else 0.0, 3),
        "fps": (v[0].get("avg_frame_rate") if v else None),
        "frames": int(v[0].get("nb_frames") or 0) if v else 0,
    }

    problems = []
    if len(v) != 1:
        problems.append(f"expected exactly 1 video stream, found {len(v)}")
    if len(a) != 1:
        problems.append(f"expected exactly 1 audio stream, found {len(a)}")

    # Direction matters, and a bare difference threshold gets this wrong.
    # Video is generated in whole frames, so its length quantises up to the next
    # frame boundary: 24.957s of speech necessarily becomes 25.000s of video.
    # That overhang is harmless - the final frame simply holds. Audio outlasting
    # video is the actual defect, because speech then plays with no picture.
    frame_s = 1.0 / 25.0
    num, _, den = (report["fps"] or "25/1").partition("/")
    try:
        if float(den):
            frame_s = 1.0 / (float(num) / float(den))
    except (ValueError, ZeroDivisionError):
        pass

    audio_overhang = adur - vdur
    if audio_overhang > tolerance_s:
        problems.append(
            f"audio outlasts video by {audio_overhang:.3f}s - that much speech "
            f"plays past the final frame (tolerance {tolerance_s:.3f}s)"
        )
    elif (vdur - adur) > (frame_s + tolerance_s):
        problems.append(
            f"video outlasts audio by {vdur - adur:.3f}s, more than one frame "
            f"({frame_s:.3f}s) of quantisation explains"
        )
    for label in ("video_start", "audio_start"):
        if abs(report[label]) > tolerance_s:
            problems.append(f"{label} is {report[label]:.3f}s, expected ~0")
    if v and report["fps"] not in (None, "25/1"):
        problems.append(f"output is {report['fps']} fps; the pipeline targets 25/1")

    report["problems"] = problems
    report["ok"] = not problems
    return report


#: ASS colours are &HAABBGGRR - alpha, then *reversed* RGB. Getting this
#: backwards silently yields blue where you wanted yellow, so both are named.
_ASS_WHITE = "&H00FFFFFF"
_ASS_YELLOW = "&H0000FFFF"

#: How much bigger the word currently being spoken renders, in percent.
#: Enough to draw the eye, small enough not to reflow the line around it.
_ACTIVE_SCALE = 118

#: Caption height as a fraction of video height. Subtitles that look right on a
#: 16:9 laptop are enormous burned into a 1080x1920 phone video, because the
#: font is sized in the subtitle renderer's own coordinate space - so size from
#: the actual frame rather than a fixed point size.
_FONT_FRACTION = 0.030
_MARGIN_FRACTION = 0.075

#: Side margin as a fraction of width, per side. Doubles as a title-safe inset.
_SIDE_MARGIN_FRACTION = 0.06

#: Approximate advance width of one character in DejaVu Sans **Bold**, as a
#: fraction of the font size. Bold is wider than regular, and this is what makes
#: a character budget correspond to real pixels instead of a guess - an earlier
#: fixed 38-character limit overflowed a 1080-wide frame by ~260px because it
#: was never checked against the rendered width.
_AVG_CHAR_EM = 0.58

#: Safety factor on the computed budget. The active word renders enlarged
#: (_ACTIVE_SCALE), so a line sized exactly to the plain width grows past the
#: margin the moment the highlight lands on a long word.
_LINE_FIT_SAFETY = 0.88


def _max_line_chars(width: int, font_size: int) -> int:
    """How many characters fit between the margins at this font size."""
    usable = width - 2 * round(width * _SIDE_MARGIN_FRACTION)
    per_char = max(1.0, font_size * _AVG_CHAR_EM)
    return max(12, int((usable / per_char) * _LINE_FIT_SAFETY))


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_escape(text: str) -> str:
    """Neutralise ASS override syntax in transcribed text."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def group_words_into_lines(
    words: list[dict], *, max_chars: int = 30, max_gap_s: float = 0.7
) -> list[list[dict]]:
    """Break a word stream into single-line caption groups.

    Splits on length, and also on any noticeable pause - a gap in speech is a
    natural caption boundary, and keeping words from either side of one on the
    same line makes the highlight appear to stall.
    """
    lines: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        text = str(w.get("text", "")).strip()
        if not text:
            continue
        gap = current and (float(w["start"]) - float(current[-1]["end"])) > max_gap_s
        too_long = current and sum(len(x["text"]) + 1 for x in current) + len(text) > max_chars
        if gap or too_long:
            lines.append(current)
            current = []
        current.append({**w, "text": text})
    if current:
        lines.append(current)
    return lines


def words_to_ass(words: list[dict], *, width: int, height: int) -> str:
    """Render word timings as karaoke-style ASS: one line, active word highlighted.

    One Dialogue event per word rather than per line: each shows the whole line
    but recolours and enlarges just the word being spoken at that moment. ASS
    has a native ``\\k`` karaoke tag, but it only sweeps a colour across
    pre-timed syllables and cannot also change size, so explicit per-word events
    are what actually delivers "bigger and yellow".
    """
    font_size = max(12, round(height * _FONT_FRACTION))
    margin_v = max(10, round(height * _MARGIN_FRACTION))
    margin_h = round(width * _SIDE_MARGIN_FRACTION)
    outline = max(1, round(font_size * 0.09))
    max_chars = _max_line_chars(width, font_size)
    anchor_x, anchor_y = round(width / 2), height - margin_v

    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        # 0 = wrap at the margins if a line somehow still overruns. Lines are
        # pre-split to fit, so this should never fire - but wrapping onto a
        # second line degrades far better than running off both edges of the
        # frame, which is what WrapStyle 2 (never wrap) did.
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        # Bold=-1 (ASS's true). BorderStyle=1 is outline+shadow - *not* 3, which
        # paints the opaque box behind the text. A thin dark outline is what
        # keeps white text readable over light footage without a box.
        f"Style: Default,DejaVu Sans,{font_size},{_ASS_WHITE},{_ASS_YELLOW},"
        f"&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,"
        f"{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []
    for line in group_words_into_lines(words, max_chars=max_chars):
        for i, active in enumerate(line):
            parts = []
            for j, w in enumerate(line):
                token = _ass_escape(w["text"])
                if j == i:
                    # Override tags need the trailing '&' that style-definition
                    # colours omit; libass tolerates its absence inconsistently.
                    parts.append(
                        f"{{\\c{_ASS_YELLOW}&\\fscx{_ACTIVE_SCALE}\\fscy{_ACTIVE_SCALE}}}"
                        f"{token}{{\\r}}"
                    )
                else:
                    parts.append(token)
            start, end = float(active["start"]), float(active["end"])
            # Hold each word until the next begins, so the line never blinks out
            # between words; the final word keeps a short tail.
            end = float(line[i + 1]["start"]) if i + 1 < len(line) else max(end, start + 0.25)
            if end <= start:
                end = start + 0.08
            # Pin every caption to one absolute point instead of letting it sit
            # on a bottom margin. Margin-positioned text is laid out relative to
            # the line box, and that box changes height whenever the highlighted
            # word scales up (\fscy) or a line wraps - so the captions visibly
            # bob up and down as the highlight moves. \an2 anchors bottom-centre
            # at \pos, so the baseline stays put and an enlarged word grows
            # upward from it rather than shifting the whole line.
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
                f"{{\\an2\\pos({anchor_x},{anchor_y})}}" + " ".join(parts)
            )

    return "\n".join(head + events) + "\n"


def burn_word_captions(video: Path, words_json: Path, dst: Path) -> Path:
    """Burn karaoke-style captions using per-word timings from the captions stage."""
    video, words_json, dst = Path(video), Path(words_json), Path(dst)
    words = json.loads(words_json.read_text(encoding="utf-8"))
    if not words:
        raise MediaError(f"no word timings in {words_json}")

    info = probe_video(video)
    ass_path = dst.parent / f".{dst.stem}.ass"
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(
        words_to_ass(words, width=info.width, height=info.height), encoding="utf-8"
    )

    escaped = str(ass_path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    try:
        run([
            _exe("ffmpeg"), "-y", "-v", "error",
            "-i", str(video),
            # ass filter, not subtitles= - the latter re-wraps using its own
            # defaults and would discard the styling built above.
            "-vf", f"ass='{escaped}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            *SDR_TAG_ARGS,
            "-c:a", "copy",
            str(dst),
        ])
    finally:
        ass_path.unlink(missing_ok=True)
    return dst


def burn_subtitles(video: Path, srt: Path, dst: Path) -> Path:
    """Hard-burn an SRT into the video."""
    video, srt, dst = Path(video), Path(srt), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg's subtitles filter needs escaping for : and \ inside the path.
    escaped = str(srt).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    # BorderStyle=1 (outline), not 3 (opaque box) - a box behind burned-in
    # captions looks like a hardcoded overlay rather than part of the video.
    # Bold, with a dark outline instead, which stays readable over light footage.
    style = (
        "FontName=DejaVu Sans,FontSize=16,Bold=-1,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=40"
    )
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(video),
        "-vf", f"subtitles='{escaped}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        *SDR_TAG_ARGS,
        "-c:a", "copy",
        str(dst),
    ])
    return dst


def to_hd(src: Path, dst: Path, *, height: int = 1080, fps: float | None = None) -> Path:
    """Upscale/pad to an HD frame with even dimensions, preserving aspect ratio."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale=-2:{height}:flags=lanczos,"
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    cmd = [
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        *SDR_TAG_ARGS,
    ]
    if fps:
        cmd += ["-r", f"{fps:.6f}"]
    cmd += ["-c:a", "copy", str(dst)]
    run(cmd)
    return dst


def ensure_wav(src: Path, dst: Path, *, sample_rate: int = 16000, mono: bool = True) -> Path:
    """Transcode any audio into plain PCM WAV at a known rate."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(src),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ])
    return dst


def trim_video(src: Path, dst: Path, seconds: float) -> Path:
    """Cut ``src`` down to roughly ``seconds`` from its start.

    Stream-copied, so this costs almost nothing and never re-encodes. The cut
    lands on the next keyframe, which means the result can run slightly long -
    fine for every caller here, which only need "at least this much".
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(src),
        "-t", f"{seconds:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ])
    return dst


def time_stretch(src: Path, dst: Path, factor: float) -> Path:
    """Change speech tempo without shifting pitch.

    ``factor`` is a tempo multiplier: 0.8 makes the delivery 25% longer (slower),
    1.2 makes it faster. A cloned voice inherits its cadence from the reference
    sample, which is often faster than a scripted presenter delivery wants to be,
    and no generation parameter reliably controls that — ``cfg_weight`` barely
    moves it. Stretching afterwards is deterministic and keeps the timbre intact.

    ffmpeg's ``atempo`` only accepts 0.5-2.0 per instance, so larger changes are
    chained. Below about 0.7 the artefacts start to be audible, so callers should
    prefer rewriting the script over stretching very hard.
    """
    src, dst = Path(src), Path(dst)
    if factor <= 0:
        raise MediaError(f"time_stretch factor must be positive, got {factor}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if abs(factor - 1.0) < 1e-3:
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        return dst

    # Decompose into steps ffmpeg will accept.
    steps: list[float] = []
    remaining = factor
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        steps.append(2.0)
        remaining /= 2.0
    steps.append(remaining)

    chain = ",".join(f"atempo={s:.6f}" for s in steps)
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(src),
        "-filter:a", chain,
        "-c:a", "pcm_s16le",
        str(dst),
    ])
    return dst


def is_playable(path: Path) -> bool:
    """Cheap validity check: does it decode and report a positive duration?"""
    try:
        info = probe_video(Path(path))
    except MediaError:
        return False
    return info.n_frames > 0 and info.duration > 0 and info.width > 0
