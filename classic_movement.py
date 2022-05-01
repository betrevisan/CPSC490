""" Predator-Prey Task for Movement (Classic Approach)
This implements the Predator-Prey task for movement within classic computing
using a restricted boltzmann machine.
"""

from data import movement_data

TRAIN_SIZE = 100
TEST_SIZE = 30
WIDTH = HEIGHT = 100
MAX_SPEED = 5

def main():
    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane and the maximum speed
    dataset = movement_data.MovementData(TRAIN_SIZE, TEST_SIZE, MAX_SPEED, WIDTH, HEIGHT)


if __name__ == "__main__":
    main()
    