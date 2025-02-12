import json
import os
# this script combined the individual category responses into larger groups
rel_size = ["biggest.json", "heaviest.json", "fits.json", "interact.json"]
can_do_it = ["can_do.json", "can_do_size.json", "can_do_shape.json", "can_do_char.json", "can_do_goal.json"]
is_a_dif = ["difference.json", "diff_criteria.json", "use_as.json","is_a.json", "types_of.json"]
risky = ["injury.json","danger.json", "damage_to_obj.json"]
equip = ["explain_use.json", "equip_used.json", "equip_in_task.json"]
obj_facts = ["obj_loc.json","objs_in_loc.json", "secondary_use.json"]
quake = ["earthquake.json"]
instr = ["instruct.json", "followup.json"]

catos = {"rel_size": rel_size, "can_do_it": can_do_it, "is_a_dif": is_a_dif, "risky": risky, "equip": equip,\
          "obj_facts": obj_facts, "quake": quake, "instr": instr}

out_dir = "../gemini_results/"

ALL = True
if ALL:
    go_out = []
    for _, val in catos.items():
        for v in val:
            curr_file = json.load(open(os.path.join(out_dir, v)))
            go_out.extend(curr_file)
    print(f"Length of go_out: {len(go_out)}")
    out = open(os.path.join(out_dir, "all.json"),"w")
    json.dump(go_out, out)
else:
    for key, val in catos.items():
        if key != "quake":
            continue
        go_out = []
        for v in val:
            curr_file = json.load(open(os.path.join(out_dir, v)))
            go_out.extend(curr_file)
        out = open(os.path.join(out_dir, f"{key}.json"),"w")
        json.dump(go_out, out)


