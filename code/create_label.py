from os import listdir
from os.path import isfile, join

IMAGE_FOLDER = '../data/training'
PAINTER_FOLDER = '../data/images'

# get image file names and put them in list
painting_fnames = [f for f in listdir(IMAGE_FOLDER) if isfile(join(IMAGE_FOLDER, f)) and f != '.DS_Store']
painting_fnames.sort()
# get painter names
painter_names = [n for n in listdir(PAINTER_FOLDER) if n != '.DS_Store']
painter_names.sort()
painter_label = {n: l + 1 for l, n in enumerate(painter_names)}

# create csv file
f = open('painting_label.csv', 'w')
f2 = open('painter_list.csv', 'w')
f.write('painting, label\n')
f2.write('painter\n')

# write to csv file
for painting_fname in painting_fnames:
    for painter_name in painter_names:
        if painter_name in painting_fname:
            label = painter_label[painter_name]
            break
    print(f'painting file name: {painting_fname}, label: {label}')
    line = painting_fname + ', ' + str(label) + '\n'
    f.write(line)

for painter_name in painter_names:
    f2.write(painter_name + '\n')

f.close()
f2.close()