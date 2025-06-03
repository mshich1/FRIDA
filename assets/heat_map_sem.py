import numpy as np
import matplotlib.pyplot as plt
import re

with open("../llama_results/sem_md.txt") as l_in:
    scores = [l for l in l_in]
    sems = {"overall":[], "rel_size":[],"can_do_it": [], "is_a_dif":[],"risky":[],\
           "equip":[], "obj_facts":[], "quake":[], "instr": []}
    for p in scores:
        cat_nums = re.match("([a-zA-Z_]+) average sem score: (0\.\d+|1.0)", p)
        if cat_nums is None:
            continue
        cat = cat_nums.group(1)
        num = float(cat_nums.group(2))
        sems[cat].append(num)
    
    x_axis_labels = ["aFRIDA 8B: relative size","aFRIDA 8B: object function","aFRIDA 8B: differences","aFRIDA 8B: objects causing harm",\
                     "aFRIDA 8B: specialized equipment","aFRIDA 8B: non-functional object facts","aFRIDA 8B: earthquakes","aFRIDA 8B: instruction understanding",\
                     "FRIDA 8B", "LLaMa 3.1 8B instruct"]
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
    
    print(results)
    fig, ax = plt.subplots()
    im = ax.imshow(results)
    plt.rcParams.update({'font.size': 12})
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
    ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

    print(len(x_axis_labels))
    print(len(y_axis_labels))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis_labels)):
        for j in range(len(x_axis_labels)):
            text = ax.text(j, i, f'{results[i, j]:.2f}',
                        ha="center", va="center", color="w")

    ax.set_title("SemScore by subsets, FRIDA 8B")
    fig.tight_layout()
    plt.show()

