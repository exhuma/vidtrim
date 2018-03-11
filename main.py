from raspicam.main import Application
from shutil import move
from os import unlink
import logging
from subprocess import check_call
from raspicam.pipeline import (
    DetectionPipeline,
    resizer,
    togray,
    blur,
    MutatorOutput,
    MotionDetector,
)
from raspicam.localtypes import Dimension
from raspicam.source import FileReader

import cv2


LOG = logging.getLogger(__name__)


class MotionSegment:

    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end

    def __repr__(self):
        return 'MotionSegment(start=%r, end=%r)' % (self.start, self.end)

    @property
    def duration(self):
        return self.end - self.start


class SwitchDetector:

    def __init__(self, source):
        self.position = 0
        self.current_state = None
        self.segments = [MotionSegment(start=0)]
        self.total_frames = source.total_frames

    def __call__(self, frames, regions):
        frame_has_motion = bool(regions)
        # Give the motion detector some time to settle
        LOG.debug('Frame #%-6d/%-10d has motion: %5s (%3.2f%%)',
                  self.position,
                  self.total_frames,
                  bool(regions),
                  (self.position/self.total_frames*100))
        if self.position < 20:
            self.current_state = 'motion' if frame_has_motion else 'still'
            self.position += 1
            return MutatorOutput([], regions)

        if frame_has_motion:
            if self.current_state == 'still':
                self.current_state = 'motion'
                self.segments.append(MotionSegment(start=self.position))
        else:
            if self.current_state == 'motion':
                self.current_state = 'still'
                self.segments[-1].end = self.position
        self.position += 1
        return MutatorOutput([], regions)


class MyPipeline(DetectionPipeline):

    def __init__(self, switch_detector):
        super().__init__([
            resizer(Dimension(320, 240)),
            togray,
            blur(5),
            MotionDetector(),
            switch_detector
        ])


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('filename')
    return parser.parse_args()


def main():

    args = parse_args()
    frame_source = FileReader(args.filename)
    frame_source.init()
    switch_detector = SwitchDetector(frame_source)
    app = Application()
    app.init_scripted(frame_source.frame_generator(),
                      debug=True, verbosity=5,
                      custom_pipeline=MyPipeline(switch_detector))
    logging.getLogger('raspicam.operations').setLevel(logging.ERROR)
    app.run()
    fps = frame_source.source.get(cv2.CAP_PROP_FPS)
    filename = args.filename

    LOG.info('%d motion segments found', len(switch_detector.segments))

    merged_segments = [switch_detector.segments[0]]
    for segment in switch_detector.segments[1:]:
        if merged_segments[-1].end is None:
            # we reached the end
            merged_segments[-1].end = switch_detector.position
        if segment.start - merged_segments[-1].end <= 100:
            merged_segments[-1].end = segment.end
            continue
        else:
            merged_segments.append(segment)
    LOG.info('%d motion segments remained after merging.', len(merged_segments))
    return filename, merged_segments, fps


def create_keyframes(filename, merged_segments, fps):
    keyframes = []
    basename, _, ext = filename.rpartition('.')
    for segment in merged_segments:
        keyframes.append(str(segment.start // fps))
        keyframes.append(str(segment.end // fps))
    args = ','.join(keyframes)
    keyed_filename = '{basename}-keyframes.{ext}'.format(
        basename=basename,
        ext=ext
    )
    cmd = [
        'ffmpeg',
        '-loglevel', 'warning',
        '-i', '%s.%s' % (basename, ext),
        '-force_key_frames', args,
        '-y',
        keyed_filename
    ]
    LOG.debug('Running %r', ' '.join(cmd))
    check_call(cmd)
    return keyed_filename


def extract_segments(input_file):
    for segment in merged_segments:
        LOG.info('Extracting %s', segment)
        basename, _, ext = input_file.rpartition('.')
        start = segment.start // fps
        outfile = '{basename}-strip-{start}.{ext}'.format(
            start=start,
            basename=basename,
            ext=ext
        )
        cmd = [
            'ffmpeg',
            '-loglevel', 'warning',
            '-ss',
            str(start),
            '-i',
            '%s.%s' % (basename, ext),
            '-t', str(segment.duration // fps),
            '-vcodec',
            'copy',
            '-acodec',
            'copy',
            '-y',
            outfile
        ]
        LOG.debug('Running %r', ' '.join(cmd))
        check_call(cmd)
        yield outfile


def join(origin_file, segments):
    filenames = list(segments)
    segments_file = 'segments.list'
    with open(segments_file, 'w') as fptr:
        fptr.writelines("file '%s'\n" % line for line in filenames)

    basename, _, ext = origin_file.rpartition('.')
    joined_filename = '%s-onlymotion.%s' % (basename, ext)

    cmd = [
        'ffmpeg',
        '-loglevel', 'warning',
        '-f', 'concat',
        '-i', segments_file,
        '-c', 'copy',
        '-y',
        joined_filename
    ]
    LOG.debug('Running %r', ' '.join(cmd))
    check_call(cmd)

    for filename in filenames:
        unlink(filename)
    unlink(segments_file)
    return joined_filename


if __name__ == '__main__':
    logging.basicConfig(level=0)
    filename, merged_segments, fps = main()
    keyed_filename = create_keyframes(filename, merged_segments, fps)
    segments = extract_segments(keyed_filename)
    joined_filename = join(filename, segments)
    unlink(keyed_filename)
    move(filename, filename + '.bak')
    move(joined_filename, filename)
