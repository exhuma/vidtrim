from vidtrim import util


def test_ffmpeg_progress():
    data = ('frame=  210 fps=0.0 q=29.0 size=     100kB time=00:00:07.22 '
            'bitrate= 114.0kbits/s    \n')
    expected = {
        'frame': 210,
        'fps': 0.0,
        'q': 29.0,
        'size': '100kB',
        'time': '00:00:07.22',
        'bitrate': '114.0kbits/s'
    }
    result = util.parse_ffmpeg_progress(data)
    assert result == expected


def test_ffmpeg_progress_2():
    data = ('frame=11274 fps=362 q=-1.0 Lsize=    7350kB time=00:06:15.73 '
            'bitrate= 160.2kbits/s    \n')
    expected = {
        'frame': 11274,
        'fps': 362,
        'q': -1.0,
        'Lsize': '7350kB',
        'time': '00:06:15.73',
        'bitrate': '160.2kbits/s'
    }
    result = util.parse_ffmpeg_progress(data)
    assert result == expected
