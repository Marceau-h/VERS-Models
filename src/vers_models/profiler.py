# SPDX-FileCopyrightText: 2025-present Marceau <git@marceau-h.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Profiler utilities for training and evaluation using torch.profiler.
"""
from datetime import datetime

from torch.cuda import is_available
from torch.profiler import profile, ProfilerActivity

def profiler_wrapper(func:callable, profile_:bool=True):
    """
    Wrapper to profile a function using torch.profiler.
    :param func: The function to profile
    :param profile_: Whether to profile the function or not (default: True)
    :return: the result of the original function call, the profiler results will be printed and saved to a file if profile is True
    """
    def wrapper(*args, **kwargs):
        if not profile_:
            return func(*args, **kwargs)
        activities = [ProfilerActivity.CPU]
        if is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=True) as prof:
            result = func(*args, **kwargs)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        prof.export_chrome_trace(f"profiler_trace_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        return result
    return wrapper

