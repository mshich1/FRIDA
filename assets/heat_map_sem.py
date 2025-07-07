import numpy as np
import matplotlib.pyplot as plt
import re

files = ["llama_sem.txt","../mistral_results/sem_mis.txt",]
label_prefix = [("aFRIDA 8B","LLaMa 3.1 8B Intruct","LLaMa FRIDA 8B"), ("MaFRIDA", "Ministral 8B Instruct", "M-FRIDA 8B")]
for f, lab in zip(files, label_prefix):
    with open(f) as f_in:
        scores = [l for l in f_in]
        ems = {"overall":[], "rel_size":[],"can_do_it": [], "is_a_dif":[],"risky":[],\
            "equip":[], "obj_facts":[], "quake":[], "instr": []}
        for p in scores:
            cat_nums = re.match("([a-zA-Z_]+) average sem score: (0\.\d+|1.0)", p)
            if cat_nums is None:
                continue
            cat = cat_nums.group(1)
            num = float(cat_nums.group(2))
            ems[cat].append(num)
        
        x_axis_labels = [f"{lab[0]}: relative sizes and shapes",f"{lab[0]}: object function",f"{lab[0]}: differences",f"{lab[0]}: objects causing harm",\
                        f"{lab[0]}: specialized equipment",f"{lab[0]}: non-functional object facts",f"{lab[0]}: earthquake",f"{lab[0]}: instruction understanding",\
                        f"{lab[0]}", f"{lab[1]}"]
        y_axis_labels = ["relative sizes and shapes templates", "object function templates", "differences templates", "objects causing harm templates",\
                        "specialized equipment templates", "non-functional object facts templates", "earthquake templates", "instruction understanding templates", "all evaluation data"]

        results = np.array([np.array(ems["rel_size"]),
                        np.array(ems["can_do_it"]),
                        np.array(ems["is_a_dif"]),
                        np.array(ems["risky"]),
                        np.array(ems["equip"]),
                        np.array(ems["obj_facts"]),
                        np.array(ems["quake"]),
                        np.array(ems["instr"]),
                        np.array(ems["overall"])])
        
        fig, ax = plt.subplots()
        im = ax.imshow(results)
        plt.rcParams.update({'font.size': 12})
        # plt.rcParams['dpi'] = 500
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels, fontsize=12)
        ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels,fontsize=12)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(y_axis_labels)):
            for j in range(len(x_axis_labels)):
                text = ax.text(j, i, f'{results[i, j]:.2f}',
                            ha="center", va="center", color="w")

        ax.set_title(f"SemScore on data subsets, {lab[2]}")
        fig.tight_layout()
        plt.show()

