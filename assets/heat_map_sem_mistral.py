import numpy as np
import matplotlib.pyplot as plt
import re

with open("../mistral_results/sem_mis.txt") as input:
    scores = [l for l in input]
    sems = {"overall":[], "rel_size":[],"can_do_it": [], "is_a_dif":[],"risky":[],\
           "equip":[], "obj_facts":[], "quake":[], "instr": []}
    for p in scores:
        cat_nums = re.match("([a-zA-Z_]+) average sem score: (0\.\d+|1.0)", p)
        if cat_nums is None:
            continue
        cat = cat_nums.group(1)
        num = float(cat_nums.group(2))
        sems[cat].append(num)
    
    x_axis_labels = ["aFRIDA: relative size model","aFRIDA: object function model","aFRIDA: differences model","aFRIDA: objects causing harm model",\
                     "aFRIDA: specialized equipment model","aFRIDA: non-functional object facts model","aFRIDA: earthquake model","aFRIDA: instruction understanding model",\
                     "FRIDA", "Ministral 8B instruct"]
    y_axis_labels = ["relative size eval data", "object function eval data", "differences eval data", "objects causing harm eval data",\
                     "specialized equipment eval data", "non-functional object facts eval data", "earthquake eval data", "instruction understanding eval data", "all evaluation data"]
    x_axis_labels = ["MaFRIDA 8B: relative size","MaFRIDA 8B: object function","MaFRIDA 8B: differences","MaFRIDA 8B: objects causing harm",\
                     "MaFRIDA 8B: specialized equipment","MaFRIDA 8B: non-functional object facts","MaFRIDA 8B: earthquakes","MaFRIDA 8B: instruction understanding",\
                     "M-FRIDA 8B", "Ministral 8B instruct"]
    y_axis_labels = ["relative size templates", "object function templates", "differences templates", "objects causing harm templates",\
                     "specialized equipment templates", "non-functional object facts templates", "earthquake templates", "instruction understanding templates", "all evaluation data"]

    results = np.array([np.array(sems["rel_size"]),
                       np.array(sems["can_do_it"]),
                       np.array(sems["is_a_dif"]),
                       np.array(sems["risky"]),
                       np.array(sems["equip"]),
                       np.array(sems["obj_facts"]),
                       np.array(sems["quake"]),
                       np.array(sems["instr"]),
                       np.array(sems["overall"])])
    
    fig, ax = plt.subplots()
    im = ax.imshow(results)
    plt.rcParams.update({'font.size': 10})
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
    ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis_labels)):
        for j in range(len(x_axis_labels)):
            text = ax.text(j, i, f'{results[i, j]:.2f}',
                        ha="center", va="center", color="w")

    ax.set_title("SemScore Accuracy on evaluation data subsets")
    ax.set_title("SemScore by subsets, M-FRIDA 8B Suite")
    fig.tight_layout()
    plt.show()

