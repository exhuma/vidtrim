import curses
import math
from curses import curs_set, wrapper
from threading import Thread


def progress(value,  length=40):
    # Block progression is 1/8
    blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    lsep, rsep = "[", "]"

    v = value*length
    x = math.floor(v)  # integer part
    y = v - x  # fractional part
    i = int(round(y*8))
    bar = "█"*x + blocks[i]
    n = length-len(bar)
    bar = lsep + bar + " "*n + rsep
    return bar


def make_window(data):

    title_length = max([len(j.filename) for j in data])
    line_template = '%%%ds %%10d/%%-10d %%s' % title_length

    def mainloop(stdscr):
        curs_set(0)
        stdscr.clear()
        stdscr.nodelay(True)
        scroll_pos = 0
        content = curses.newpad(len(data)+1, curses.COLS)
        while True:
            ch = stdscr.getch()
            lines = [
                line_template % (
                    line.filename,
                    line.processed_frames,
                    line.total_frames,
                    progress(line.progress, 5)
                )
                for line in data
            ]
            pad_height = curses.LINES

            for i, line in enumerate(lines):
                content.addstr(i, 0, line)
            if ch == ord('q'):
                break
            elif ch in (curses.KEY_DOWN, ord('j')) and (len(lines) > pad_height):
                scroll_pos = min(len(lines)-pad_height, scroll_pos+1)
            elif ch in (curses.KEY_UP, ord('k')):
                scroll_pos = max(0, scroll_pos-1)

            content.refresh(scroll_pos, 0, 0, 0, pad_height, curses.COLS)
            stdscr.refresh()
    return mainloop


def main(data):
    wrapper(make_window(data))


class Monitor(Thread):

    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deamon = True
        self.data = data

    def run(self):
        main(self.data)


if __name__ == '__main__':
    class Job:
        def __init__(self, progress, total, filename):
            self.progress = progress
            self.total = total
            self.filename = filename

        def __str__(self):
            return '%-20s: %10d/%-10d %s' % (
                self.filename,
                self.progress,
                self.total,
                progress(self.progress / self.total, 3)
            )

        @property
        def progress(self):
            if self.total:
                return self.progress / self.total
            else:
                return 0.0

    data = [
        Job(0, 10000, 'Title 1'),
        Job(0, 20000, 'This is Title 2'),
        Job(0, 22000, '3'),
        Job(0, 30000, 'Tit 5'),
        Job(0, 5000, 'Taital 4'),
        Job(0, 100000, 'blabla'),
        Job(0, 50000, ''),
    ]
    thread = Monitor(data)
    thread.start()
