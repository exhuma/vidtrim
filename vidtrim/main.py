import concurrent.futures
import curses
import logging
import os
import sys
from datetime import datetime, timedelta
from enum import Enum
from glob import glob
from logging import FileHandler, Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from math import floor, inf
from os import close, unlink
from os.path import basename, exists, getsize
from os.path import join as pjoin
from queue import Queue
from shutil import copystat, move
from statistics import mean
from subprocess import PIPE, STDOUT, CalledProcessError, Popen
from tempfile import mkstemp

import cv2
from gouge.colourcli import Simple

from raspicam.localtypes import Dimension
from raspicam.pipeline import (DetectionPipeline, MotionDetector,
                               MutatorOutput, blur, resizer, togray)
from vidtrim.util import parse_ffmpeg_progress

LOG = logging.getLogger(__name__)
SUPPORTED_EXTS = {'mkv'}
TTY_CLEAR_SEQ = None
MAX_WORKERS = 5


class FFMpegState(Enum):
    INITIAL = 'initial'
    INPUT = 'input'
    OUTPUT = 'output'
    HAVE_DURATION = 'have_duration'
    RUNNING = 'running'


class FFMpegProgress:

    def __init__(self, progress_callback=None):
        self.state = FFMpegState.INITIAL
        self.progress_callback = progress_callback
        self.__lines = []

    @property
    def lines(self):
        return '\n'.join(self.__lines)

    def feed(self, line):
        self.__lines.append(line)
        func = getattr(self, '%s_feed' % self.state.value, None)
        if func:
            func(line)
        else:
            LOG.error('No handler available for state %s', self.state)

    def initial_feed(self, line):
        if line.startswith('Input'):
            self.state = FFMpegState.INPUT
        elif line.strip().startswith('frame='):
            data = parse_ffmpeg_progress(line)
            if self.progress_callback:
                self.progress_callback(data)

    def input_feed(self, line):
        line = line.strip()
        if not line.strip().startswith('Duration:'):
            return

        duration, start, bitrate = line.split(',')
        _, duration = duration.split(':', 1)
        _, start = start.split(':', 1)
        _, bitrate = bitrate.split(':', 1)
        duration = duration.strip()
        start = start.strip()
        bitrate = bitrate.strip()
        LOG.debug('ffmpeg input: duration=%s, start=%s, bitrate=%s',
                  duration, start, bitrate)
        self.state = FFMpegState.INITIAL


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


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


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


def run_ffmpeg(args, input_framecount=None, progress_callback=None):
    cmd = [
        'ffmpeg',
        # '-loglevel', 'warning',
        '-stats',
        '-y',
        *args
    ]
    LOG.debug('Running %r', ' '.join(cmd))

    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT,
                 env={'AV_LOG_FORCE_NOCOLOR': '1'},
                 bufsize=1,
                 universal_newlines=True)

    progress_monitor = None
    if input_framecount:
        def handle_progess(data):
            if progress_callback:
                progress_callback(data['frame'] / input_framecount)

        progress_monitor = FFMpegProgress(handle_progess)
        for line in iter(proc.stdout.readline, ''):
            progress_monitor.feed(line)

    remaining_output = proc.stdout.read()
    proc.stdout.close()
    return_code = proc.wait()
    if return_code != 0:
        LOG.error(progress_monitor.lines
                  if progress_monitor else remaining_output)
        raise CalledProcessError(return_code, cmd)


def create_keyframes(filename, segments, fps, total_framecount, workdir=None,
                     progress_callback=None):
    keyframes = []
    without_ext, _, ext = filename.rpartition('.')
    for segment in segments:
        keyframes.append(str(segment.start // fps))
        keyframes.append(str(segment.end // fps))
    args = ','.join(keyframes)
    fptr, keyed_filename = mkstemp(prefix=basename(filename),
                                   suffix='-keyframes.%s' % ext,
                                   dir=workdir)
    close(fptr)
    run_ffmpeg([
        '-i', '%s.%s' % (without_ext, ext),
        '-force_key_frames', args,
        keyed_filename
    ], total_framecount, progress_callback)
    return keyed_filename


def extract_segments(input_file, segments, fps, workdir=None):
    for segment in segments:
        LOG.debug('Extracting %s', segment)
        without_ext, _, ext = input_file.rpartition('.')
        start = segment.start // fps
        fptr, outfile = mkstemp(prefix=basename(input_file),
                                suffix='-strip-%s.%s' % (start, ext),
                                dir=workdir)
        close(fptr)
        run_ffmpeg([
            '-ss',
            str(start),
            '-i', '%s.%s' % (without_ext, ext),
            '-t', str(segment.duration // fps),
            '-vcodec', 'copy',
            '-acodec', 'copy',
            outfile
        ])
        yield outfile


def join(origin_file, segments, do_cleanup=True, workdir=None):
    filenames = list(segments)
    without_ext, _, ext = origin_file.rpartition('.')
    fptr, segments_file = mkstemp(
        prefix=basename(origin_file),
        suffix='-segments.list',
        dir=workdir)
    close(fptr)
    with open(segments_file, 'w') as fptr:
        fptr.writelines("file '%s'\n" % line for line in filenames)

    fptr, joined_filename = mkstemp(
        prefix=basename(origin_file),
        suffix='-onlymotion.%s' % ext,
        dir=workdir
    )
    close(fptr)
    run_ffmpeg([
        '-f', 'concat',
        '-safe', '0',
        '-i', segments_file,
        '-c', 'copy',
        joined_filename
    ])

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


def process(filename, queue, destination, workdir, do_cleanup, do_backup):
    switch_detector = SwitchDetector(filename, None)
    pipeline = MyPipeline(switch_detector)
    LOG.info('Processing %s', filename)
    file_basename = basename(filename)

    generator, total, fps = frame_generator(filename)
    for pos, frame in generator():
        LOG.debug('Processing frame %d/%d %3.2f%%' % (
            pos, total, (pos/total)*100))
        switch_detector.total_frames = total
        pipeline.feed(frame)
        # The value 0.70 has an impact to the code below. See the comments.
        queue.put(('detecting', filename, (pos/total) * 0.70, None))

    inter_frame_threshold = 100
    LOG.info('Merging segments which are closer than %d frames',
             inter_frame_threshold)
    queue.put(('merging', filename, 0.70, None))
    merged_segments = merge_segments(
        switch_detector.segments,
        switch_detector.position,
        inter_frame_threshold)

    if len(merged_segments) == 1:
        LOG.info('Only 1 segment remains: Nothing to extract')
        if destination:
            final_destination = pjoin(destination, file_basename)
            LOG.info('Moving to %s', final_destination)
            final_filesize = getsize(filename)
            queue.put(('done', filename, 1.0, final_filesize))
            move(filename, final_destination)
    else:
        LOG.info('Extracting %d segments', len(merged_segments))

        def progress_callback(value):
            # Given the code above, we're now at 0.70. Adding 0.25 will get us
            # to the 0.95 which is found below.
            relative_progress = 0.25 * value
            queue.put(('extracting', filename, 0.70 + relative_progress, None))

        keyed_filename = create_keyframes(
            filename, merged_segments, fps, total, workdir=workdir,
            progress_callback=progress_callback)
        segments = extract_segments(
            keyed_filename,
            merged_segments,
            fps,
            workdir=workdir
        )
        # See above (extracting) how we get to 0.95
        queue.put(('joining', filename, 0.95, None))
        joined_filename = join(
            filename,
            segments,
            do_cleanup=do_cleanup,
            workdir=workdir)
        if do_cleanup:
            remove(keyed_filename)
        original_file = filename
        if do_backup:
            backup_filename = filename + '.bak'
            if destination:
                backup_filename = pjoin(destination, basename(backup_filename))
            LOG.info('Backing up original file as %s', backup_filename)
            move(filename, backup_filename)
            original_file = backup_filename
        move(joined_filename, filename)
        LOG.info('Copying file metadata from %r to %r', original_file, filename)
        copystat(original_file, filename)
        if destination:
            final_destination = pjoin(destination, file_basename)
            LOG.info('Moving to %s', final_destination)
            move(filename, final_destination)
        final_filesize = getsize(final_destination)
        queue.put(('done', filename, 1.0, final_filesize))


def clear_screen():
    global TTY_CLEAR_SEQ
    if TTY_CLEAR_SEQ is None:
        if sys.stdout.isatty():
            curses.setupterm()
            TTY_CLEAR_SEQ = curses.tigetstr('clear')
        else:
            TTY_CLEAR_SEQ = b''
    if TTY_CLEAR_SEQ:
        os.write(sys.stdout.fileno(), TTY_CLEAR_SEQ)


def setup_logging(trace_file='', rotate_trace_file=True):
    console_handler = StreamHandler()
    console_handler.setFormatter(Simple(show_threads=True, show_exc=True))
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


def print_progress(start_time, map):
    clear_screen()
    all_progress = []
    pending_count = 0
    done_count = 0
    total_old_size = sum(item['old_size'] for item in map.values())
    total_new_size = total_old_size - sum(
        item.get('new_size', 0) for item in map.values()
        if item.get('new_size', 0) is not None
    )
    saved_size = total_old_size - total_new_size

    for filename, details in sorted(map.items()):
        state = details['state']
        progress = details['progress']
        error = details.get('error', '')
        all_progress.append(progress)
        if state == 'pending':
            pending_count += 1
            continue
        if state in 'done':
            done_count += 1
            continue
        print('%-30s [%10s]: %6.2f%% %s' % (
            basename(filename)[:30],
            state,
            (progress * 100),
            error or progress_bar(progress, length=20),
        ))
        if error:
            LOG.exception('Error processing %r', details['filename'],
                          exc_info=error)

    overall_progress = mean(all_progress)
    current_time = datetime.now()
    elapsed_time = current_time - start_time
    percent_per_second = overall_progress / elapsed_time.total_seconds()
    if percent_per_second > 0:
        estimated_total_time = timedelta(seconds=1/percent_per_second)
    else:
        estimated_total_time = inf
    remaining_time = estimated_total_time - elapsed_time

    print(80*'-')
    print('pending/done/total: %d/%d/%s' % (pending_count, done_count, len(map)))
    print(80*'-')
    print('Old size:   %9s' % sizeof_fmt(total_old_size))
    print('New size:   %9s' % sizeof_fmt(total_new_size))
    print('Saved size: %9s' % sizeof_fmt(saved_size))
    print(80*'-')
    print('Elapsed time  : %s ' % elapsed_time)
    print('Remaining time: %s ' % remaining_time)
    print(80*'-')
    print('Overall: %s %3.2f%%' % (
        progress_bar(overall_progress),
        overall_progress * 100,
    ))
    print(80*'-')


def progress_bar(value,  length=40):
    # Block progression is 1/8
    blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    lsep, rsep = "[", "]"

    v = value*length
    x = floor(v)  # integer part
    y = v - x  # fractional part
    i = int(round(y*8))
    bar = "█"*x + blocks[i]
    n = length-len(bar)
    bar = lsep + bar + " "*n + rsep
    return bar


def main():
    args = parse_args()
    setup_logging(args.trace_file, args.rotate_trace_file)
    start_time = datetime.now()

    if args.workdir and not exists(args.workdir):
        LOG.error('Workdir %s is missing!', args.workdir)
        return 1

    unglobbed = set()
    for filename in args.filenames:
        unglobbed |= set(glob(filename))
    supported_files = {fn for fn in unglobbed if fn[-3:] in SUPPORTED_EXTS}

    queue = Queue()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_filename = {}
        progress_map = {
            filename: {
                'filename': filename,
                'state': 'pending',
                'progress': 0.0,
                'old_size': getsize(filename)
            }
            for filename in supported_files
        }
        for filename in sorted(supported_files):
            future_to_filename[executor.submit(
                process, filename, queue, args.destination, args.workdir,
                args.cleanup, args.backup)] = filename

        keep_waiting = True
        while keep_waiting:
            done, pending = concurrent.futures.wait(
                future_to_filename, timeout=1)
            if not pending:
                keep_waiting = False

            while queue.qsize():
                state, filename, progress, new_size = queue.get()
                progress_map[filename]['state'] = state
                progress_map[filename]['progress'] = progress
                progress_map[filename]['new_size'] = new_size

            for future in done:
                exc = future.exception()
                if exc:
                    progress_map[future_to_filename[future]]['state'] = 'error'
                    progress_map[future_to_filename[future]]['error'] = exc
            print_progress(start_time, progress_map)


if __name__ == '__main__':
    main()
