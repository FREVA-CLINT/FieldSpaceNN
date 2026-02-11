from typing import Dict, List


def _check_times(times: List[int], t_0: int, t_T: int) -> None:
    """
    Validate a generated repaint schedule.

    :param times: List of timesteps (descending, with a terminal -1).
    :param t_0: Minimum valid timestep.
    :param t_T: Maximum valid timestep.
    :return: None.
    """
    # Ensure schedule starts strictly decreasing.
    assert times[0] > times[1], (times[0], times[1])

    # Ensure schedule terminates at -1.
    assert times[-1] == -1, times[-1]

    # Ensure unit step transitions.
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Ensure values are within [t_0, t_T].
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)


def _plot_times(x: List[int], times: List[int]) -> None:
    """
    Plot a repaint schedule for debugging.

    :param x: X-axis indices.
    :param times: Timestep values to plot.
    :return: None.
    """
    import matplotlib.pyplot as plt
    plt.plot(x, times)
    plt.show()


def get_schedule_jump(
    t_T: int,
    n_sample: int,
    jump_length: int,
    jump_n_sample: int,
    jump2_length: int = 1,
    jump2_n_sample: int = 1,
    jump3_length: int = 1,
    jump3_n_sample: int = 1,
    start_resampling: int = 100000000,
    debug: bool = False,
) -> List[int]:
    """
    Build a repaint-style schedule with optional nested jumps.

    :param t_T: Maximum diffusion timestep.
    :param n_sample: Number of resampling steps within each base step.
    :param jump_length: Jump length for the primary schedule.
    :param jump_n_sample: Number of samples for the primary jump schedule.
    :param jump2_length: Jump length for the secondary schedule.
    :param jump2_n_sample: Number of samples for the secondary schedule.
    :param jump3_length: Jump length for the tertiary schedule.
    :param jump3_n_sample: Number of samples for the tertiary schedule.
    :param start_resampling: Latest timestep at which resampling is allowed.
    :param debug: Whether to plot the resulting schedule.
    :return: List of timesteps including terminal -1.
    """

    jumps: Dict[int, int] = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2: Dict[int, int] = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3: Dict[int, int] = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts: List[int] = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    _check_times(ts, -1, t_T)

    if debug:
        _plot_times(x=range(len(ts)), times=ts)

    return ts
