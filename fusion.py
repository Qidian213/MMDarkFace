import os

root = '/data/wangs/gflv2/GFocalV2-master/fusion/'
bus = root + '2000_2800/'
ori = root + '800_1600/'
fusion = root + 'fusion/'

bus_files = os.listdir(bus)
for name in bus_files:
    bus_result = []
    with open(bus + name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'bus' in line:
                bus_result.append(line.strip())
    other_result = []
    with open(ori + name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'bus' not in line:
                other_result.append(line.strip())

    with open(fusion + name, 'w') as f:
        for line in other_result:
            f.writelines(line + '\n')
        for line in bus_result:
            f.writelines(line + '\n')
    print("1")