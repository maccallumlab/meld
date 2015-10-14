#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import os
import shutil
import tempfile
import contextlib
import time
from functools import wraps
import logging
import logging.handlers
import SocketServer
import struct
import pickle
import Queue


@contextlib.contextmanager
def in_temp_dir():
    """Context manager to run in temporary directory"""
    try:
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        yield

    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)


def log_timing(dest_logger):
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            t1 = time.time()
            res = func(*args, **kwds)
            t2 = time.time()
            dest_logger.debug('%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0))
            return res
        return wrapper
    return wrap


class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver.

    The configure_logging_and_launch_listener function should be launched
    in another process. The socket number can be retrieved
    from socket_queue and the logger can be told to abort
    through the abort_queue.
    """

    allow_reuse_address = 1

    def __init__(self, host, abort_queue, socket_queue, handler=LogRecordStreamHandler):
        # we request port zero, which should get an unused, non-privileged port
        SocketServer.ThreadingTCPServer.__init__(self, (host, 0), handler)
        # queue used to communicate from the main MELD process that the
        # LogRecordSocketReciever should abort
        self.abort_queue = abort_queue
        # queue used to communicate the listening port back to the
        # main MELD process
        self.socket_queue = socket_queue
        # send our port number back to the launching process
        self.socket_queue.put(self.server_address)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            # check the abort queue to see if we should terminate
            try:
                abort = self.abort_queue.get(block=False)
            except Queue.Empty:
                pass


class HostNameContextFilter(logging.Filter):
    """Filter class that adds hostid information to logging records."""
    def __init__(self, hostid):
        logging.Filter.__init__(self)
        self.hostid = hostid

    def filter(self, record):
        record.hostid = self.hostid
        return True


def configure_logging_and_launch_listener(host, abort_queue, socket_queue):
    fmt = '%(hostid)s %(asctime)s %(levelname)s %(name)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename='remd.log', format=fmt, datefmt=datefmt)
    receiver = LogRecordSocketReceiver(host, abort_queue, socket_queue)
    receiver.serve_until_stopped()
