"""Emit periodic stacks only for an explicitly enabled preflight.
仅在显式开启的预检进程中周期性打印调用栈，定位首步耗时。
"""
import faulthandler
import os

if os.environ.get("TM_PREFLIGHT_TRACE") == "1":
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
