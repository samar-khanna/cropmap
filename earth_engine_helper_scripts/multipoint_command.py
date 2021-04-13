names = []
point_locs = []
with open('points.txt') as f:
    for line in f:
        name = line.split()[0].strip()
        names.append(name)
        point_loc_str = line.split('(')[1].strip()[1:-3]
        point_loc = [float(point_loc_str.split(',')[0]),
                     float(point_loc_str.split(',')[1])]
        point_locs.append(point_loc)
print("var names = {}".format(names))
print("var multipoint = ee.Geometry.MultiPoint({})".format(point_locs))
print("var indices = {}".format(list(range(len(names)))))

print({i:n for i,n in enumerate(names)})
