#!/usr/bin/env python3
import random

from gas_simulation.ball_collision_animator import MultiBallSimulationAnimatorMain

def runSimulation1(
    framerate: int,
    print_mechE: bool=False,
    check_overlap: bool=True,
) -> None:
    sim1 = MultiBallSimulationAnimatorMain(dist_unit=20,\
            arena_dims=(20, 20),\
            framerate=framerate,\
            n_sim_cycle_per_frame=1, g=3)
    sim1.addBall(m=2, radius=2, r0=(2, 3), v0=(5, 3), color=(255,0,0))
    sim1.addBall(m=1, radius=1, r0=(5, 9), v0=(-12, 1),\
            color=(0,255,0))
    sim1.addBall(m=1, radius=1, r0=(15, 16), v0=(0, 0),\
            color=(0,0,255))
    sim1.run(print_mechE=print_mechE, check_overlap=check_overlap)
    return

def runSimulation2(
    framerate: int,
    n_rows: int,
    print_mechE: bool=False,
    check_overlap: bool=True,
    print_n_balls: bool=False,
) -> None:
    d = 50
    sim2 = MultiBallSimulationAnimatorMain(dist_unit=12,\
            arena_dims=(d, d), framerate=framerate,\
            n_sim_cycle_per_frame=1, g=10)
    cnt = 0
    for i2 in range(1, n_rows * 2, 2):
        if i2 == 1:
            color = (255, 0, 0)
        elif i2 == 3:
            color = (0, 255, 0)
        elif i2 >= 5:
            color = (0, 0, 255)
        m = i2 ** 2
        for i1 in range(1, d - 1, 3):
            cnt += sim2.addBall(m=m, radius=1, r0=(i1, i2),\
                    v0=(random.uniform(-20, 20),\
                    random.uniform(-20, 20)), color=color,\
                    balls_t_updated=True, check_overlap=True)
    if print_n_balls:
        print(f"n balls = {cnt}")
    sim2.run(print_mechE=print_mechE, check_overlap=check_overlap)
    return
