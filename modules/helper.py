import functools
import logging
import sys
import time
import traceback
from pprint import pprint
from typing import List
from typing import Tuple

from termcolor import colored
from pprint import pformat
import torch

__all__ = [
    "pvars",
    "debug_log",
    "class_debug_log",
]
DECORATED_FUNCTIONS = set()

logger = logging.getLogger(__name__)

def params(args,kwargs):
        params = ""
        def brackets(text, color = 'light_red'):
            return f"{colored('(',color)}{text}{colored(')',color)}"
        for arg in args:
            if type(arg) is torch.Tensor:
                params += f"{colored(f'Tensor(shape={arg.shape})', 'yellow', attrs=['dark'])}, "
                continue
            params += f"{colored(pformat(arg,indent=2), 'yellow', attrs=['dark'])}, "
        if kwargs:
            for key, value in kwargs.items():
                if type(value) is torch.Tensor:
                    value = f'Tensor(shape={value.shape})'
                params += (
                    f"\n{colored(f'{key}=', 'magenta')}"
                    f"{colored(pformat(value), 'yellow', attrs=['dark'])}, "
                )
        return brackets(params.rstrip(", "))
        

class PrettyStack:
    def __init__(self, e):
        self.e = e

    def get(self) -> str:
        stack_lines = []
        last_filname = ""

        for frame, line_no in traceback.walk_tb(self.e.__traceback__):
            filename = frame.f_code.co_filename
            func_name = frame.f_code.co_name
            redacted_parts, unique_parts = self.redact_path(filename, last_filname)
            local_vars_str = "\nLocal Vars:\n"
            local_vars_str += params([],frame.f_locals)
            stack_line = local_vars_str
            stack_line += f"{colored(redacted_parts, 'grey')}/{colored(unique_parts, 'yellow', attrs=['bold'])}:"
            stack_line += f"{colored(func_name, 'green')}()"
            stack_line += f"({colored(line_no, 'red', attrs=['bold'])})"

            stack_lines.append(stack_line)
            last_filname = filename

        trace = "\n".join(stack_lines)

        msg = f"\n{'-'*5}\nStacktrace:\n{trace}\n\n"
        msg += f"{'-'*5}\nError:\n"
        msg += f"{colored(str(self.e), 'red')}\n"
        #        msg += f"\nFile: {traceback.extract_tb(self.e.__traceback__)[-1].filename}, Line: {traceback.extract_tb(self.e.__traceback__)[-1].lineno}"
        msg += f"{'-'*5}\n"
        return msg

    def html_color(self, text: str, color: str, attrs: List[str] = []) -> str:
        style = f"color:{color};"
        if "bold" in attrs:
            style += "font-weight:bold;"
        if "dark" in attrs:
            style += "opacity:0.7;"
        return f"<span style='{style}'>{text}</span>"

    def get_html(self) -> str:
        stack_lines: List[str] = []
        last_filename = ""

        for frame, line_no in traceback.walk_tb(self.e.__traceback__):
            filename, func_name = frame.f_code.co_filename, frame.f_code.co_name
            redacted_parts, unique_parts = self.redact_path(filename, last_filename)

            local_vars_str = "<p><strong>Local Vars:</strong></p><ul>"
            for k, v in frame.f_locals.items():
                local_vars_str += f"<li>{self.html_color(k, 'yellow')} {self.html_color('=', 'yellow', ['dark'])} {v}</li>"
            local_vars_str += "</ul>"

            stack_line = (
                f"{local_vars_str}"
                f"<p>{self.html_color(redacted_parts, 'grey')}/{self.html_color(unique_parts, 'yellow', ['bold'])}: "
                f"{self.html_color(func_name + '()', 'green')} {self.html_color(f'({line_no})', 'red', ['bold'])}</p>"
            )

            stack_lines.append(stack_line)
            last_filename = filename

        trace = "<br>".join(stack_lines)

        msg = (
            f"<hr><p><strong>Stacktrace:</strong></p>{trace}<br>"
            f"<hr><p><strong>Error:</strong></p>"
            f"{self.html_color(str(self.e), 'red')}"
            f"<hr>"
        )
        return msg

    def redact_path(self, filename: str, last_filename: str) -> Tuple[str, str]:
        common_prefix_len = len(set(filename).intersection(last_filename))
        return filename[:common_prefix_len], filename[common_prefix_len:]


class IndentLogger:
    def __init__(self):
        self.indent = 1
        self.start_times = {}
        self.start_time = time.perf_counter()

    def log(self, message):
        time_diff = (time.perf_counter() - self.start_time) * 1000
        # timediff from ms to "h m s ms"
        hours, rem = divmod(time_diff // 1000, 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds = int(round(time_diff % 1000))
        formatted_time = (
            f"{int(hours)}:{int(minutes)}:{int(seconds)} {milliseconds:03.2F}ms"
        )
        if message:
            logger.info(f"{formatted_time}" + " " * self.indent + message)
            print(f"{formatted_time}" + " " * self.indent + message)
        else:
            logger.info("")

    def enter(self, class_name, method_name, args, kwargs):
        self.start_times[self.indent] = time.perf_counter()

        msg = f"+  {colored(class_name, 'light_blue')}.{colored(method_name, 'cyan')}"
        params_str = params(args,kwargs)

        self.log(f"{msg}{params_str}")
        self.indent += 1
    
    
        
    def exit(self, class_name, method_name, result):
        self.indent -= 1
        start_time = self.start_times[self.indent]
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        class_name = colored(class_name, "light_blue")
        method_name = colored(method_name, "cyan")
        msg = f"-  {class_name}.{method_name}{colored('()', 'light_red')} ({execution_time:03.2f})"
        if result:
            msg += f" = {colored(result, 'yellow', attrs=['dark'])}"
        self.log(msg)
        self.log("")


indent_logger = IndentLogger()


def pvars(text="", frame=None):
    frame = sys._getframe(1) if frame is None else frame
    local_vars = frame.f_locals
    info = frame.f_code
    print(f"{info.co_filename}: {info.co_name} ({frame.f_lineno})")
    print(f"+++ local vars {text}+++")

    pprint(local_vars, depth=2)
    print("--- local vars ---\n")


def debug_log(method):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        try:
            class_name = args[0].__class__.__name__ if len(args) else ""
            method_name = method.__name__
            indent_logger.enter(class_name, method_name, args[1:], kwargs)
            result = method(*args, **kwargs)
            indent_logger.exit(class_name, method_name, result)
        except Exception as e:
            msg = PrettyStack(e).get()
            logging.error(msg)
            sys.exit(-1)
        return result

    DECORATED_FUNCTIONS.add(method.__name__)
    return wrapped


def class_debug_log(cls):
    if logger.level == logging.DEBUG:
        for key, value in vars(cls).items():
            if callable(value):
                setattr(cls, key, debug_log(value))
    return cls


def apply_decorators(module_globals):
    for name in DECORATED_FUNCTIONS:
        func = module_globals.get(name)
        if callable(func):
            module_globals[name] = debug_log(func)
