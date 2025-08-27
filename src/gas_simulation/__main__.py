#!/usr/bin/env python3

from gas_simulation.example_simulation_animations import (
    runSimulation1,
    runSimulation2,
)

def main() -> None:
    runSimulation1(
        60, print_mechE=False,
        check_overlap=True,
    )
    runSimulation2(
        60,
        n_rows=3,
        print_mechE=False,
        check_overlap=True,
        print_n_balls=True
    )

if __name__ == "__main__":
    main()

