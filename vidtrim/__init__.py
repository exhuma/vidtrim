from os.path import dirname, join

with open(join(dirname(__file__), 'version.txt')) as fp:
    __version__ = fp.read().strip()
