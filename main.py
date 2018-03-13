import logging
from os import rmdir, unlink
from os.path import basename, exists
from os.path import join as pjoin
from shutil import move
from subprocess import check_call
from tempfile import mkstemp

import cv2
from raspicam.localtypes import Dimension
from raspicam.pipeline import (DetectionPipeline, MotionDetector,
                               MutatorOutput, blur, resizer, togray)

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

    def __init__(self, filename, total_frames):
        self.position = 0
        self.current_state = 'motion'
        self.segments = [MotionSegment(start=0)]
        self.total_frames = total_frames
        self.basename = basename(filename)

    def __call__(self, frames, regions):
        frame_has_motion = bool(regions)
        # Give the motion detector some time to settle
        progress = (self.position/self.total_frames*100)

        if frame_has_motion:
            if self.current_state == 'still':
                self.current_state = 'motion'
                self.segments.append(MotionSegment(start=self.position))
        else:
            if self.current_state == 'motion':
                self.current_state = 'still'
                self.segments[-1].end = self.position
                LOG.info('%s | Added segment: %r (%.2f%%)',
                         self.basename, self.segments[-1], progress)
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
    parser.add_argument('-n', '--no-backup', dest='backup', default=True,
                        action='store_false',
                        help='Do not keep a .bak file of the modified video.')
    parser.add_argument('-d', '--destination', default='',
                        help='Destination folder of the modified video')
    parser.add_argument('-C', '--no-cleanup', dest='cleanup', default=True,
                        action='store_false',
                        help='Do not remove temporary files.')
    parser.add_argument('-w', '--work-dir', dest='workdir', default=None,
                        help='Folder to store temporary files.')
    return parser.parse_args()


def merge_segments(segments, end_position, threshold):
    merged_segments = [segments[0]]
    for segment in segments[1:]:
        if segment.start - merged_segments[-1].end <= threshold:
            LOG.debug('Merging %s with %s', merged_segments[-1], segment)
            merged_segments[-1].end = segment.end
            continue
        else:
            merged_segments.append(segment)
    if merged_segments[-1].end is None:
        # we reached the end
        LOG.info('Open-ended segment. Setting to max frame number (%s)',
                 end_position)
        merged_segments[-1].end = end_position
    return merged_segments


def create_keyframes(filename, segments, fps, workdir=None):
    keyframes = []
    basename, _, ext = filename.rpartition('.')
    for segment in segments:
        keyframes.append(str(segment.start // fps))
        keyframes.append(str(segment.end // fps))
    args = ','.join(keyframes)
    _, keyed_filename = mkstemp(prefix=basename, suffix='-keyframes.%s' % ext,
                                dir=workdir)
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


def extract_segments(input_file, segments, fps, workdir=None):
    for segment in segments:
        LOG.info('Extracting %s', segment)
        basename, _, ext = input_file.rpartition('.')
        start = segment.start // fps
        _, outfile = mkstemp(prefix=basename,
                             suffix='-strip-%s.%s' % (start, ext),
                             dir=workdir)
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


def join(origin_file, segments, cleanup=True, workdir=None):
    filenames = list(segments)
    basename, _, ext = origin_file.rpartition('.')
    _, segments_file = mkstemp(prefix=basename, suffix='-segments.list',
                               dir=workdir)
    with open(segments_file, 'w') as fptr:
        fptr.writelines("file '%s'\n" % line for line in filenames)

    _, joined_filename = mkstemp(
        prefix=basename,
        suffix='-onlymotion.%s' % ext,
        dir=workdir
    )
    cmd = [
        'ffmpeg',
        '-loglevel', 'warning',
        '-f', 'concat',
        '-safe', '0',
        '-i', segments_file,
        '-c', 'copy',
        '-y',
        joined_filename
    ]
    LOG.debug('Running %r', ' '.join(cmd))
    check_call(cmd)

    if cleanup:
        for filename in filenames:
            do_cleanup(filename)
        do_cleanup(segments_file)
    return joined_filename


def frame_generator(filename):

    source = cv2.VideoCapture(filename)
    total_frames = source.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = source.get(cv2.CAP_PROP_FPS)
    if not source.isOpened():
        raise Exception('Unable to open %s' % filename)

    def frames():
        while True:
            next_frame = source.get(cv2.CAP_PROP_POS_FRAMES)
            success, image = source.read()
            if not success:
                LOG.info('Unable to read frame from %s. Might have reached '
                         'end of file!', filename)
                return
            yield next_frame, image

    return frames, total_frames, fps


def do_cleanup(filename):
    LOG.debug('Cleaning up %r', filename)
    unlink(filename)


def main():
    logging.basicConfig(level=0)
    args = parse_args()

    if args.workdir and not exists(args.workdir):
        LOG.error('Workdir %s is missing!', args.workdir)
        return 1

    switch_detector = SwitchDetector(args.filename, None)
    pipeline = MyPipeline(switch_detector)
    LOG.info('Processing %s', args.filename)

    generator, total, fps = frame_generator(args.filename)
    for pos, frame in generator():
        LOG.debug('Processing frame %d/%d %3.2f%%' % (
            pos, total, (pos/total)*100))
        switch_detector.total_frames = total
        pipeline.feed(frame)

    inter_frame_threshold = 100
    LOG.info('Merging segments which are closer than %d frames',
             inter_frame_threshold)
    merged_segments = merge_segments(
        switch_detector.segments,
        switch_detector.position,
        inter_frame_threshold)

    if len(merged_segments) == 1:
        LOG.info('Only 1 segment remains: Nothing to extract')
    else:
        LOG.info('Extracting %d segments', len(merged_segments))
        keyed_filename = create_keyframes(
            args.filename, merged_segments, fps, workdir=args.workdir)
        segments = extract_segments(
            keyed_filename,
            merged_segments,
            fps,
            workdir=args.workdir
        )
        joined_filename = join(
            args.filename,
            segments,
            cleanup=args.cleanup,
            workdir=args.workdir)
        if args.cleanup:
            do_cleanup(keyed_filename)
        if args.backup:
            backup_filename = args.filename + '.bak'
            LOG.info('Backing up original file as %s', backup_filename)
            move(args.filename, backup_filename)
        move(joined_filename, args.filename)
        if args.destination:
            final_destination = pjoin(args.destination, args.filename)
            LOG.info('Moving to %s', final_destination)
            move(args.filename, final_destination)


if __name__ == '__main__':
    main()
