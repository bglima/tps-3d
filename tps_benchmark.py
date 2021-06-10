
#!/usr/bin/env python
from tps_3d import TPS3D
import numpy as np
import argparse
import os
import time

def get_time_of_train(user_path, robot_path):
    user_abs_path = os.path.abspath(user_path)
    robot_abs_path = os.path.abspath(robot_path)
    try:
        user_data = np.genfromtxt(user_abs_path, delimiter=',', names=True)
        robot_data = np.genfromtxt(robot_abs_path, delimiter=',', names=True)
    except Exception as e:
        print('[ERROR] {}'.format(e))
        quit()
    
    assert len(user_data) == len(robot_data), "[ERROR] Source and target are not the same lenght."
    n = len(user_data)
    
    xs, ys, zs = user_data['x'], user_data['y'], user_data['z']
    xt, yt, zt = robot_data['x'], robot_data['y'], robot_data['z']
    c_points = np.vstack([xs, ys, zs]).T
    t_points = np.vstack([xt, yt, zt]).T

    model = TPS3D()
    
    start_time = time.time()
    model.fit(c_points, t_points)
    end_time = time.time()
    return (model, end_time - start_time)

def benchmark_train_over_n_iterations(n_iterations):
    benchmark_start = time.time()
    print("\nRunning benchmark for model training")
    time_buffer = []
    for i in range(n_iterations):
        _, train_time = get_time_of_train(
            user_path = "bruno_definitive.csv",
            robot_path = "denso_mirrored.csv"
        )
        time_buffer.append( train_time )
    benchmark_end = time.time()
    print("Finished iterations. Took {:.2f} seconds.".format(benchmark_end - benchmark_start))
    print("Training time statistics over {} iterations.".format(n_iterations))
    print("Median is:", np.median(time_buffer))
    print("Mean is:", np.mean(time_buffer))
    print("Std is:", np.std(time_buffer))
    print("Min value is:", min(time_buffer))
    print("Max value is:", max(time_buffer))
    print("Mean iters per second:", 1/np.mean(time_buffer))

def benchmark_transform_over_n_iterations(n_iterations):
    benchmark_start = time.time()
    print("\nRunning benchmark for point transformation")
    time_buffer = []
    model, _ = get_time_of_train(
        user_path = "person.csv",
        robot_path = "robot.csv"
    )
    for i in range(n_iterations):
        px = np.random.randint(-150, 150)
        py = np.random.randint(-150, 150)
        pz = np.random.randint(-150, 150)
        p = np.array([px, py, pz])

        start_time = time.time()
        model.transform(p)
        end_time = time.time()
        time_buffer.append( end_time - start_time )

    benchmark_end = time.time()
    print("Finished iterations. Took {:.2f} seconds.".format(benchmark_end - benchmark_start))
    print("Training time statistics over {} iterations.".format(n_iterations))
    print("Median is:", np.median(time_buffer))
    print("Mean is:", np.mean(time_buffer))
    print("Std is:", np.std(time_buffer))
    print("Min value is:", min(time_buffer))
    print("Max value is:", max(time_buffer))
    print("Mean iters per second:", 1/np.mean(time_buffer))

if __name__ == '__main__':
    n_iterations = 1000
    benchmark_train_over_n_iterations(n_iterations)
    benchmark_transform_over_n_iterations(n_iterations)
