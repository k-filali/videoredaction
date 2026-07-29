import subprocess
from pathlib import Path

from clearframe.media import MediaProcessor


def generate_test_video(
    destination: Path,
    media: MediaProcessor,
    *,
    duration_seconds: float = 1.2,
    audio: bool = True,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(media.ffmpeg_path),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x17233a:s=640x360:r=15:d={duration_seconds}",
    ]
    if audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=880:sample_rate=44100:duration={duration_seconds}",
            ]
        )
    command.extend(
        [
            "-vf",
            "drawbox=x=160:y=220:w=180:h=54:color=white:t=fill,"
            "drawbox=x=172:y=232:w=156:h=30:color=black:t=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if audio:
        command.extend(["-c:a", "aac", "-shortest"])
    command.append(str(destination))
    subprocess.run(
        command,
        capture_output=True,
        check=True,
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return destination

