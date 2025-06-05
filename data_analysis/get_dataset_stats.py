import json
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
# This script gets the max rouge score for each instruction and the instruction length
# It then generates averages over template dataset, category dataset, and overall dataset
cat_map = {"rel_size":["biggest", "heaviest", "fits", "interact"],\
            "can_do_it":["can_do", "can_do_size", "can_do_shape", "can_do_char", "can_do_goal"], \
            "is_a_dif": ["difference", "diff_criteria", "use_as","is_a", "types_of"], \
            "risky":["injury","danger", "damage_to_obj"], \
            "equip":["explain_use", "equip_used", "equip_in_task"], \
            "obj_facts":["obj_loc","objs_in_loc", "secondary_use"], \
            "quake":["earthquake"], \
            "instr":["instruct", "followup"]}

cat_eval = {"rel_size":{'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]},"can_do_it":{'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]}, \
            "is_a_dif": {'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]}, "risky":{'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]}, \
            "equip":{'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]}, "obj_facts":{'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]},\
            "quake":{'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]}, "instr":{'instr_len':[],'ans_len':[],'total_len':[],'rouge_scores':[]}}

files= ["biggest", "heaviest", "fits", "interact",
            "can_do", "can_do_size", "can_do_shape", "can_do_char", "can_do_goal", \
            "difference", "diff_criteria", "use_as","is_a", "types_of", \
            "injury","danger", "damage_to_obj", \
            "explain_use", "equip_used", "equip_in_task", \
            "obj_loc","objs_in_loc", "secondary_use", \
            "earthquake", \
            "instruct", "followup"]

def add_to_dict(n,i,c, t, r):
    cat_eval[n]['instr_len'].append(i)
    cat_eval[n]['ans_len'].append(c)
    cat_eval[n]['total_len'].append(t)
    cat_eval[n]['rouge_scores'].append(r)

def put_in_cat(filename, i, c, t, r):
    if filename in cat_map["rel_size"]:
        add_to_dict("rel_size",i,c,t,r)
    elif filename in cat_map["can_do_it"]:
        add_to_dict("can_do_it",i,c,t,r)
    elif filename in cat_map["is_a_dif"]:
        add_to_dict("is_a_dif",i,c,t,r)
    elif filename in cat_map["risky"]:
        add_to_dict("risky",i,c,t,r)
    elif filename in cat_map["equip"]:
        add_to_dict("equip",i,c,t,r)
    elif filename in cat_map["obj_facts"]:
        add_to_dict("obj_facts",i,c,t,r)
    elif filename in cat_map["quake"]:
        add_to_dict("quake",i,c,t,r)
    elif filename in cat_map["instr"]:
        add_to_dict("instr",i,c,t,r)
    else:
        print("Freakout")

with open("dataset_stats.txt","w") as stat_out:
    overall_len = []
    overall_rouge = []
    for f in files:
        with open(f"../gemini_results/{f}.json") as data_in:
            all_data = json.load(data_in)
            instr_len = []
            ans_len = []
            total_len = []
            rouge_scores = []
            for q in all_data:
                i_len = len(q["instruction"].split(" "))
                instr_len.append(i_len)
                if q['input'] != None:
                    ch_len = len(q["input"].split(" "))
                else:
                    ch_len = 0
                ans_len.append(ch_len)
                total_len.append(i_len+ch_len)
                overall_len.append(i_len+ch_len)
                r_score = next(iter(q["most_similar_instructions"].values()))# thank you chatGPT for this line specifically
                rouge_scores.append(r_score)
                overall_rouge.append(r_score)
                put_in_cat(f,i_len, ch_len, i_len+ch_len, r_score) 
            np_inst = np.array(instr_len)
            np_ans = np.array(ans_len)
            np_rouge = np.array(rouge_scores)
            np_total = np.array(total_len)
            stat_out.write(f"**DATASET IS {f}.json**\n")
            stat_out.write(f'average instruction length: {np_inst.mean()}\n')
            stat_out.write(f'median instruction length: {np.median(np_inst)}\n')
            stat_out.write(f'average answer choice length: {np_ans.mean()}\n')
            stat_out.write(f'median answer choice length: {np.median(np_ans)}\n')
            stat_out.write(f'average total length: {np_total.mean()}\n')
            stat_out.write(f'median total length: {np.median(np_total)}\n')            
            stat_out.write(f'average ROUGE score: {np_rouge.mean()}\n')
            stat_out.write(f'median ROUGE score: {np.median(np_rouge)}\n\n')

    #stats for the categories
    for n,c in cat_eval.items():
        np_inst = np.array(c['instr_len'])
        np_ans = np.array(c['ans_len'])
        np_total = np.array(c['total_len'])
        np_rouge = np.array(c['rouge_scores'])
        stat_out.write(f'**CATEGORY IS {n}**\n')
        stat_out.write(f'average instruction length: {np_inst.mean()}\n')
        stat_out.write(f'median instruction length: {np.median(np_inst)}\n')
        stat_out.write(f'average answer choice length: {np_ans.mean()}\n')
        stat_out.write(f'median answer choice length: {np.median(np_ans)}\n')
        stat_out.write(f'average total length: {np_total.mean()}\n')
        stat_out.write(f'median total length: {np.median(np_total)}\n')
        stat_out.write(f'average ROUGE score: {np_rouge.mean()}\n')
        stat_out.write(f'median ROUGE score: {np.median(np_rouge)}\n\n')   

    #overall stats
    np_ol = np.array(overall_len)
    np_or = np.array(overall_rouge)
    stat_out.write(f'overall avg instruction length: {np_ol.mean()}\n')
    stat_out.write(f'overall median instruction length: {np.median(np_ol)}\n')
    stat_out.write(f'overall avg ROUGE score: {np_or.mean()}\n')
    stat_out.write(f'overall median ROUGE score: {np.median(np_or)}')
    stat_out.write(f'overall range in instruction lengths: {np_ol.min()} - {np_ol.max()}\n')
    stat_out.write(f'overall range in ROUGE score: {np_or.min()} - {np_or.max()}\n')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 7))

axes[0].hist(np_ol, bins=30, color='skyblue', edgecolor='black', range=(15,80))
axes[0].set_title('Instruction Length Distribution')
axes[0].set_xlabel('Instruction Length (words)')
axes[0].axvline(np_ol.mean(), color='k', linestyle='dashed', linewidth=2)
axes[0].xaxis.set_major_locator(MaxNLocator(nbins=20))

axes[1].hist(np_or, bins=30, color='lightgreen', edgecolor='black', range=(0.25,0.97))
axes[1].set_title('Instructions\' Max ROUGE Score Distribution')
axes[1].set_xlabel('ROUGE score')
axes[1].axvline(np_or.mean(), color='k', linestyle='dashed', linewidth=2)
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=20))
 
# Adding labels and title
# for ax in axes:
#     ax.set_ylabel('Frequency')

# Adjusting layout for better spacing
plt.tight_layout()

# Display the figure
plt.show()



