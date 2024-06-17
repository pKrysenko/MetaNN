import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from time import time
import scipy
import os
from tempfile import TemporaryFile
import tensorflow as tf
from numpy import load
from tqdm import tqdm


existed_figures = {"Figure1A_150x150_mkm": "Figure1A-2.npz",
                  "Figure1B_150x45_mkm": "Figure1B-2.npz",
                  "Figure2_600x600_nm": "Figure2-1.npz",
                  "figure3_750_280_t150_h120_nm": "Figure3-2.npz",
                  "figure3_600_224_t150_h100_nm": "Figure3-4.npz",
                  "figure3_600_224_t150_h140_nm": "Figure3-5.npz",
                  "figure3_600_224_t150_h180_nmply": "Figure3-6.npz",
                  "figure3_600_224_t150_h220_nm": "Figure3-7.npz",
                  "Figure4_450x450_nm": "Figure4-1.npz",
                   "Figure5": "Figure5-2.npz"
                  }


def get_data_graph(path):
    X = []
    y = []
    with open(path, "r") as data:
        for d in data:
            D = d.split(";")
            if len(D) != 2:
                break
            X.append(float(D[0]))
            Y = D[1].strip("\n")
            y.append(float(Y))
    return np.array(X).reshape(-1, 1), np.array(y).reshape(-1, 1)


def max_min_scaler(array, max_, min_):
    return (array - min_) / (max_ - min_)


def get_max_min(arrays):
    max_ = 0
    min_ = np.inf
    for array in arrays:
        max_ = array.max() if array.max() > max_ else max_
        min_ = array.min() if array.min() < min_ else min_
    return max_, min_


def create_struct_tensor(df):
    xs = sorted(set(df["x"].tolist()))
    ys = sorted(set(df["y"].tolist()))
    zs = sorted(set(df["z"].tolist()))
    data = np.zeros((len(xs), len(ys), len(zs), 3))
    for ox, x in tqdm(enumerate(xs), total=len(xs)):
        for oy, y in tqdm(enumerate(ys), total=len(ys)):
            for oz, z in enumerate(zs):
                material = df[(df.x == x) & (df.y == y) & (df.z == z)][["p1", "p2", "p3"]].to_numpy(dtype=np.float32)
                if material.size > 0:
                    data[ox][oy][oz] = material
    return data

def pipeline_structure(path):
    df = pd.read_csv(path, header=None, delim_whitespace=True, names=['x', 'y', 'z', 'p1', 'p2', 'p3']).drop_duplicates(
        ['x', 'y', 'z'])
    print(df)
    arr = np.array(df).T
    for k in range(3):
        scaler_str = MinMaxScaler()
        new = np.reshape(arr[k + 3], (-1, 1))
        scaler_str.fit([min(new), max(new)])
        new = scaler_str.transform(new)
        arr[k + 3] = new.flatten()
    arr = arr.T
    df = pd.DataFrame(arr, columns=['x', 'y', 'z', 'p1', 'p2', 'p3'])
    struct_tensor = create_struct_tensor(df)
    return struct_tensor


# polynomial
def pipeline_coefs_approximation(path):
    x, y = get_data_graph(path)
    x = x.flatten()
    y = y.flatten()
    scaler = MinMaxScaler()
    x = x.reshape(-1, 1)
    scaler.fit([[140], [371]])
    x = scaler.transform(x)
    x = x.flatten()
    z = np.polyfit(x, y, 10)
    p = np.poly1d(z)
    xp = np.linspace(min(x), max(x), 100)
    plt.plot(x, y, '.', label='Реальні значення')
    plt.plot(xp, p(xp), '-', label='Апроксимація')
    plt.ylim(0, 1)
    plt.xlabel("Масштабована частота")
    plt.ylabel("Коефіцієнт пропускання")
    plt.legend()
    plt.show()
    coef = p
    # coef = [round(val, 5) for val in np.array(coef)]
    return coef


# print(pipeline_coefs_approximation('graphs/graphs/graphs/Figure3/2/data.txt'))


# interpolation
def pipeline_coefs_interpolation(path):
    x, y = get_data_graph(path)
    x = x.flatten()
    y = y.flatten()
    x_new = np.linspace(min(x), max(x), 10)
    f = scipy.interpolate.interp1d(x, y, kind='linear')
    y_new = f(x_new)
    plt.plot(x_new, y_new, 'r', label='interp/extrap')
    plt.plot(x, y, 'b--', label='data')
    plt.legend()
    plt.show()
    coef = y_new
    return coef


# print(pipeline_coefs_interpolation('graphs/graphs/graphs/Figure3/2/data.txt'))


# points
def pipeline_points(path):
    x, y = get_data_graph(path)
    coef = [x, y]
    # plt.plot(x, y, 'b--', label='data')
    # plt.legend()
    # plt.show()
    return coef  # return [x, y]

def proc_graphs(graph_path, i):
    if i == 'appr':
        coef = pipeline_coefs_approximation(graph_path)  # preprocess of points/coefs
        max_min.append([max(coef), min(coef)])
        # print(coef)

        coef = np.reshape(coef, (-1, 1))
        scaler = MinMaxScaler()
        scaler.fit([min(coef), max(coef)])
        coef = scaler.transform(coef)
        coef = np.reshape(coef, (-1, 1))
        # print(coef)
    elif i == 'inter':
        coef = pipeline_coefs_interpolation(graph_path)
        max_min.append([max(coef), min(coef)])
        # print(coef)

        coef = np.reshape(coef, (-1, 1))
        scaler = MinMaxScaler()
        scaler.fit([min(coef), max(coef)])
        coef = scaler.transform(coef)
        coef = np.reshape(coef, (-1, 1))
        # print(coef)
    else:
        coef = pipeline_points(graph_path)
        max_min.append([[0.2], [371]])
        # print(coef)

        scaler = MinMaxScaler()
        scaler.fit([[0.2], [371]])
        x = scaler.transform(coef[0])
        y = coef[1]
        it = int(len(x) / 19)
        x, y = x[0::it], y[0::it]
        if len(x) > 20:
            x, y = x[1::], y[1::]
        coef = np.concatenate((x, y), axis=None)
        coef = coef.flatten()
        coef = np.reshape(coef, (-1, 1))
    return coef

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    
    cached_folder = "processed_data/zozyuk/points/"
    struct_folder = "xyzproperties/xyzproperties/"
    graph_folder = "graphs/graphs/graphs/"
    
    save_path = "processed_data/krysenko_condi/points"
    
    coefs = []
    structs = []
    max_min = []
    exp_conds = []
    for ind, row in data.iterrows():
        exp_conds.append(f"{row['polarization']}_{row['angle']}_{row['degree_ns']}_{row['num_plates']}")
        existed_struct = existed_figures.get(row["figure_name"])
        if existed_struct:
            path_to_existed_struct = os.path.join(cached_folder, existed_struct)
            #print(path_to_existed_struct)
            struct = np.load(path_to_existed_struct)["struct"]
        else:
            path_to_raw_struct = os.path.join(struct_folder, row["figure_name"] + ".txt")
            struct = pipeline_structure(path_to_raw_struct)

        graph_path = os.path.join(graph_folder, row["figure_fold"].capitalize(), "data.txt")
        structs.append(struct)
        coef = pipeline_points(graph_path)
        #print(coef)
        max_min.append([[0.2], [371]])
        # print(coef)

        scaler = MinMaxScaler()
        scaler.fit([[0.2], [371]])
        x = scaler.transform(coef[0])
        y = coef[1]
        it = int(len(x) / 19)
        x, y = x[0::it], y[0::it]
        if len(x) > 20:
            x, y = x[1::], y[1::]
        coef = np.concatenate((x, y), axis=None)
        coef = coef.flatten()
        coef = np.reshape(coef, (-1, 1))
        coefs.append(coef)
    
    i = 0
    for coef, struct, figure_fold, exp_cond in zip(coefs, structs, data.figure_fold.tolist(), exp_conds):
        i+=1
        print(i)
        name = f"{figure_fold.split('/')[0]}-{figure_fold.split('/')[1]}_{exp_cond}"
        save_name_data = os.path.join(save_path, name + ".npz")
        save_name_mm = os.path.join(save_path, "mm.npz")
        np.savez(save_name_data, struct=struct, coef=coef)
        if save_path == "processed_data/points/":
            np.savez(save_name_mm, mm_c_x=(0.2, 371), mm_c_y=(0, 1))