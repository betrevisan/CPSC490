import math
from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def dist(p1, p2):
    """Computes the distance between two points.
    Parameters
    ----------
    p1 : List
        Coordinates of point #1 as [x, y]
    p2 : List
        Coordinates of point #2 as [x, y]
    Returns
    -------
    float
        The distance between p1 and p2
    """
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def main():
    # quantum = [[61.00, 71.00], [65.00, 71.00], [63.00, 75.0], [67.00, 75.00], [65.00, 79.00], [61.00, 75.00], [58.00, 79.00], [62.00, 79.00], [60.00, 83.00], [64.00, 85.00], [63.00, 89.00], [67.00, 89.00], [63.00, 92.00]]
    classical = [[50, 50], [46.0, 46.0], [42.0, 42.0], [38.0, 38.0], [42.0, 35.0], [40.0, 30.0], [37.0, 34.0], [36.0, 29.0], [33.0, 33.0], [35.0, 37.0], [34.0, 32.0], [37.0, 34.0], [38.0, 29.0]]
    # ideal = [[61, 71], [56, 73], [51, 70], [46, 67], [41, 64], [36, 61], [31, 58], [26, 55], [21, 52], [16, 49], [11, 46], [6, 43], [3, 38]]
    predator1 = [[70, 70], [69.0, 69.0], [68.0, 68.0], [67.0, 67.0], [66.0, 66.0], [65.0, 65.0], [64.0, 64.0], [63.0, 63.0], [62.0, 62.0], [61.0, 61.0], [60.0, 60.0], [59.0, 59.0], [58.0, 58.0]]
    predator2 = [[95, 95], [94.0, 94.0], [93.0, 93.0], [92.0, 92.0], [91.0, 91.0], [90.0, 90.0], [89.0, 89.0], [88.0, 88.0], [87.0, 87.0], [86.0, 86.0], [85.0, 85.0], [84.0, 84.0], [83.0, 83.0]]
    prey1 = [[35, 35], [34.0, 34.0], [33.0, 33.0], [32.0, 32.0], [31.0, 31.0], [30.0, 30.0], [29.0, 30.0], [28.0, 29.0], [27.0, 29.0], [26.0, 28.0], [25.0, 27.0], [24.0, 27.0], [23.0, 26.0]]
    prey2 = [[25, 25], [24.0, 24.0], [23.0, 23.0], [22.0, 22.0], [21.0, 21.0], [20.0, 20.0], [19.0, 19.0], [18.0, 18.0], [17.0, 17.0], [16.0, 16.0], [15.0, 15.0], [14.0, 14.0], [13.0, 13.0]]

    classical = np.array(classical)
    # ideal = np.array(ideal)
    # quantum = np.array(quantum)
    predator1 = np.array(predator1)
    predator2 = np.array(predator2)
    prey1 = np.array(prey1)
    prey2 = np.array(prey2)
  
    # plot lines
    # plt.plot(ideal[:, 0], ideal[:, 1], label = "ideal", marker="<", markersize=3)
    plt.plot(classical[:, 0], classical[:, 1], label = "classical", marker="<", markersize=3)
    # plt.plot(quantum[:, 0], quantum[:, 1], label = "quantum", marker=">", markersize=3)
    plt.plot(predator1[:, 0], predator1[:, 1], linestyle="dashed", label="predator1", marker="<", markersize=3)
    plt.plot(predator2[:, 0], predator2[:, 1], linestyle="dashed", label="predator2", marker="<", markersize=3)
    plt.plot(prey1[:, 0], prey1[:, 1], linestyle="dashed", label="prey1", marker="<", markersize=3)
    plt.plot(prey2[:, 0], prey2[:, 1], linestyle="dashed", label="prey2", marker="<", markersize=3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()