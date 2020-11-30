import json
import codecs

# PARAMS TO CHANGE
# TODO: Use args
num_agents = 2
num_instances = 5

print("CHANGE num_agents and number of instances to generate for different number of agents & instance runs\n")

ego_start = "3"
codes = ["30", "31", "32"]

temp_intention = codes
final_intention = []

if num_agents < 4:
    for i in range(num_agents - 1):
        temp = []
        for j in range(3):
            start_pos = str(j)
            for elem in temp_intention:
                valid = True
                for i in range(0, len(elem), 2):
                    if elem[i] == start_pos:
                        valid = False
                        break
                if valid:
                    temp.append(elem + start_pos)
        temp2 = []
        for k in range(4):
            end_pos = str(k)
            temp2.extend([elem + end_pos for elem in temp if end_pos != elem[-1]])
        temp_intention = temp2

    final_intention = temp_intention
else:
    agents = ["0", "1", "2"]
    combinations = [[], [], []]
    for i, agent in enumerate(agents):
        for j in range(4):
            end_pos = str(j)
            if end_pos != agent:
                combinations[i].append(agent + end_pos)

    print(combinations)
    final_intention = []
    for code in codes:
        for agent_2 in combinations[0]:
            for agent_3 in combinations[1]:
                for agent_4 in combinations[2]:
                    intention = code + agent_2 + agent_3 + agent_4
                    final_intention.append(intention)


num_intentions = len(final_intention)
per_instance = num_intentions // num_instances
# print (per_instance)

# generate splits and save to json
for i in range(num_instances):
    scenarios = dict()
    if i == (num_instances - 1):
        scenarios['split'] = final_intention[(num_instances - 1) * per_instance:]
    else:
        scenarios['split'] = final_intention[i * per_instance: (i + 1) * per_instance]
    with open('split_{}.json'.format(i + 1), 'w') as f:
        json.dump(scenarios, f)

scenarios_dict = json.loads(codecs.open(
    'split_1.json', 'r', encoding='utf-8').read())
scenarios = scenarios_dict['split']
