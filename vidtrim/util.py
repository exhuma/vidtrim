import re


def parse_ffmpeg_progress(line):
    match_a = re.match(r'frame=(.*?) '
                       r'fps=(.*?) '
                       r'q=(.*?) '
                       r'size=(.*?) '
                       r'time=(.*?) '
                       r'bitrate=(.*?)$',
                       line)
    match_b = re.match(r'frame=(.*?) '
                       r'fps=(.*?) '
                       r'q=(.*?) '
                       r'Lsize=(.*?) '
                       r'time=(.*?) '
                       r'bitrate=(.*?)$',
                       line)
    if match_a:
        frame, fps, q, size, time, bitrate = match_a.groups()
        data = dict(
            frame=int(frame),
            fps=float(fps),
            q=float(q),
            size=size.strip(),
            time=time.strip(),
            bitrate=bitrate.strip(),
        )
    elif match_b:
        frame, fps, q, size, time, bitrate = match_b.groups()
        data = dict(
            frame=int(frame),
            fps=float(fps),
            q=float(q),
            Lsize=size.strip(),
            time=time.strip(),
            bitrate=bitrate.strip(),
        )
    else:
        raise ValueError('Unsupported status line: %r', line)
    return data
