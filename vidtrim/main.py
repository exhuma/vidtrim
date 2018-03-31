import concurrent.futures
import logging
from glob import glob
from logging import FileHandler, Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from os import unlink
from os.path import basename, exists
from os.path import join as pjoin
from shutil import copystat, move
from subprocess import check_call
from tempfile import mkstemp

import cv2
from gouge.colourcli import Simple
from raspicam.localtypes import Dimension
from raspicam.pipeline import (DetectionPipeline, MotionDetector,
                               MutatorOutput, blur, resizer, togray)

from vidtrim.ui import Monitor


LOG = logging.getLogger(__name__)


class Job:

    def __init__(self, filename, destination, workdir, cleanup, backup):
        self.filename = filename
        self.destination = destination
        self.workdir = workdir
        self.cleanup = cleanup
        self.backup = backup
        self.processed_frames = 0
        self.total_frames = 0

    @property
    def progress(self):
        if self.total_frames:
            return self.processed_frames / self.total_frames
        else:
            return 0.0


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
                LOG.debug('%s | Added segment: %r (%.2f%%)',
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
    parser.add_argument('filenames', nargs='+')
    parser.add_argument('-n', '--no-backup', dest='backup', default=True,
                        action='store_false',
                        help='Do not keep a .bak file of the modified video.')
    parser.add_argument('-d', '--destination', default='',
                        help='Destination folder of the modified video')
    parser.add_argument('-C', '--no-cleanup', dest='cleanup', default=True,
                        action='store_false',
                        help='Do not remove temporary files.')
    parser.add_argument('--trace-file', dest='trace_file', default='',
                        help='Trace log')
    parser.add_argument('--grow-trace-file', dest='rotate_trace_file',
                        default=True, action='store_false',
                        help='Do not automatically rotate trace file')
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
        LOG.debug('Open-ended segment. Setting to max frame number (%s)',
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
    fptr, keyed_filename = mkstemp(prefix=basename,
                                   suffix='-keyframes.%s' % ext,
                                   dir=workdir)
    fptr.close()
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
        LOG.debug('Extracting %s', segment)
        basename, _, ext = input_file.rpartition('.')
        start = segment.start // fps
        fptr, outfile = mkstemp(prefix=basename,
                                suffix='-strip-%s.%s' % (start, ext),
                                dir=workdir)
        fptr.close()
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


def join(origin_file, segments, do_cleanup=True, workdir=None):
    filenames = list(segments)
    basename, _, ext = origin_file.rpartition('.')
    fptr, segments_file = mkstemp(prefix=basename, suffix='-segments.list',
                                  dir=workdir)
    fptr.close()
    with open(segments_file, 'w') as fptr:
        fptr.writelines("file '%s'\n" % line for line in filenames)

    fptr, joined_filename = mkstemp(
        prefix=basename,
        suffix='-onlymotion.%s' % ext,
        dir=workdir
    )
    fptr.close()
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

    if do_cleanup:
        for filename in filenames:
            remove(filename)
        remove(segments_file)
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


def remove(filename):
    LOG.debug('Cleaning up %r', filename)
    unlink(filename)


def progress_bar(progress, n=60):
    num_filled = round(progress * n)
    num_empty = n - num_filled
    filled = '#' * num_filled
    empty = '-' * num_empty
    return '[%s%s]' % (filled, empty)


def process(job):

    switch_detector = SwitchDetector(job.filename, None)
    pipeline = MyPipeline(switch_detector)
    LOG.info('Processing %s', job.filename)
    file_basename = basename(job.filename)

    generator, total, fps = frame_generator(job.filename)
    job.total_frames = total
    for pos, frame in generator():
        job.processed_frames =  pos
        if pos % 300 == 0:
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
        if job.destination:
            final_destination = pjoin(job.destination, file_basename)
            LOG.info('Moving to %s', final_destination)
            move(job.filename, final_destination)
    else:
        LOG.info('Extracting %d segments', len(merged_segments))
        keyed_filename = create_keyframes(
            job.filename, merged_segments, fps, workdir=job.workdir)
        segments = extract_segments(
            keyed_filename,
            merged_segments,
            fps,
            workdir=job.workdir
        )
        joined_filename = join(
            job.filename,
            segments,
            do_cleanup=job.do_cleanup,
            workdir=job.workdir)
        if job.do_cleanup:
            remove(keyed_filename)
        original_file = job.filename
        if job.do_backup:
            backup_filename = job.filename + '.bak'
            if job.destination:
                backup_filename = pjoin(job.destination, backup_filename)
            LOG.info('Backing up original file as %s', backup_filename)
            move(job.filename, backup_filename)
            original_file = backup_filename
        move(joined_filename, job.filename)
        copystat(original_file, job.filename)
        if job.destination:
            final_destination = pjoin(job.destination, file_basename)
            LOG.info('Moving to %s', final_destination)
            move(job.filename, final_destination)


def setup_logging(trace_file='', rotate_trace_file=True):
    console_handler = StreamHandler()
    console_handler.setFormatter(Simple(show_threads=True))
    console_handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    if trace_file:
        if rotate_trace_file:
            handler = RotatingFileHandler(
                trace_file, maxBytes=1024*1024*5, backupCount=5)
        else:
            handler = FileHandler(trace_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(Formatter(
            '%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s'
        ))
        logging.getLogger().addHandler(handler)


def main():
    args = parse_args()
    setup_logging(args.trace_file, args.rotate_trace_file)

    if args.workdir and not exists(args.workdir):
        LOG.error('Workdir %s is missing!', args.workdir)
        return 1

    unglobbed = set()
    for filename in args.filenames:
        unglobbed |= set(glob(filename))

    jobs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_filename = {}
        for filename in sorted(unglobbed):
            job = Job(
                filename,
                args.destination,
                args.workdir,
                args.cleanup,
                args.backup)
            jobs.append(job)
            future_to_filename[executor.submit(process, job)] = filename

        monitor = Monitor(jobs)
        monitor.start()

if __name__ == '__main__':
    main()
