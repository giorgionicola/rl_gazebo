import numpy as np
from numpy import pi, sin
from termcolor import cprint
import sys

from typing import Callable, NoReturn


def normalize_to_01(value: float, max_value: float, min_value: float) -> float:
    """
    Normalize a value between 0 and 1
    
    :param value: value ti be normalized
    :param max_value: upper limit
    :param min_value: lower limit
    :return:
    """
    assert max_value > min_value, 'The max_value given is lower than min_value'
    return (value - min_value) / (max_value - min_value)


def normalize_from_01_to_any_range(value: float, upper_range: float, lower_range: float) -> float:
    """
    Normalize a value between the given ranges, the value must have been already normalized between 0 and 1
    
    :param value:
    :param upper_range:
    :param lower_range:
    :return:
    """
    eps = sys.float_info.epsilon
    if not (0 - eps <= value <= 1 + eps):
        print('The value given is not normalized between 0 and +1')
        print(f'value {v}')
    assert upper_range > lower_range, 'The max_value given is lower than min_value'
    return value * (upper_range - lower_range) + lower_range


def wrap_values(value: float, low_limit: float, up_limit: float) -> float:
    """
    wrap values between low_limit
    
    :param value: value to be wrapped
    :param low_limit:
    :param up_limit:
    :return:
    """
    assert type(value) is float or type(value) is int, 'The value to be wrapped must be a number'
    assert type(low_limit) is float or type(low_limit) is int, 'The low_limit must be a number'
    assert type(up_limit) is float or type(up_limit) is int, 'The up_limit must be a number'
    delta = up_limit - low_limit
    while value > up_limit:
        value -= delta
    while value <= low_limit:
        value += delta
    return value


def wrap_to_2pi(value: float) -> float:
    """
    Wrap value between [0,2*pi)
    
    :param value:
    :return:
    """
    return wrap_values(value, low_limit=0, up_limit=2 * pi)


def wrap_to_pi(value: float) -> float:
    """
    Wrap value between [-pi,pi)
    
    :param value:
    :return:
    """
    return wrap_values(value, low_limit=-pi, up_limit=pi)


def clip_value(value: float, upper_range: float, lower_range: float) -> float:
    """
    Clip a value between upper and lower_range
    
    :param value:
    :param upper_range:
    :param lower_range:
    :param substitute_lower_with:
    :return:
    """
    assert upper_range > lower_range, 'The max_value given is lower than min_value'
    
    if value > upper_range:
        return upper_range
    elif value < lower_range:
        return lower_range
    else:
        return value


def pseudo_random_function(max_ampl, min_ampl, max_freq, min_freq=0) -> Callable:
    """
    Pseudo random function based on the sum of 3 sin() with decreasing amplitude and random frequency and phases
    
    :param max_ampl: max amplitude of the single sin(), the overall max amplitude can be
        (max_ampl + min_ampl) / 2 + 0.9*max_ampl
    :param min_ampl: min amplitude of the single sin()
    :param max_freq: max frequency of the single sin(
    :param min_freq: min frequency of the single sin(
    :return:
    """
    rand_amplitude = np.random.uniform(low=min_ampl, high=max_ampl, )
    rand_f = np.random.uniform(low=min_freq, high=max_freq, )
    rand_ph1 = np.random.uniform(low=0.0, high=2 * pi, )
    rand_ph2 = np.random.uniform(low=0.0, high=2 * pi, )
    rand_ph3 = np.random.uniform(low=0.0, high=2 * pi, )
    
    def foo(t): return (max_ampl + min_ampl) / 2 + 0.5 * rand_amplitude * sin(2 * pi * rand_f * t + rand_ph1) + \
                       0.3 * rand_amplitude * sin(2 * pi * 3 * rand_f * t + rand_ph2) + \
                       0.1 * rand_amplitude * sin(2 * pi * 4 * rand_f * t + rand_ph3)
    
    return foo


def print_red_warn(string: str) -> NoReturn:
    """
    Print a red Warning
    
    :param string:
    :return:
    """
    cprint('WARN: ' + string, color='red')
    return
