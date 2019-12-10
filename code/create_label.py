from os import listdir
from os.path import isfile, join

TRAINING_IMAGE_FOLDER = '../data/training'
TEST_IMAGE_FOLDER = '../data/testing'
PAINTER_FOLDER = '../data/images'

# get image file names and put them in list
training_painting_fnames = [f for f in listdir(TRAINING_IMAGE_FOLDER) if isfile(join(TRAINING_IMAGE_FOLDER, f)) and f != '.DS_Store']
testing_painting_fnames = [f for f in listdir(TEST_IMAGE_FOLDER) if isfile(join(TEST_IMAGE_FOLDER, f)) and f != '.DS_Store']
training_painting_fnames.sort()
testing_painting_fnames.sort()

# get painter names
painter_names = [n for n in listdir(PAINTER_FOLDER) if n != '.DS_Store']
painter_names.sort()
painter_label = {n: l for l, n in enumerate(painter_names)}

# create csv file
f = open('training_painting_label.csv', 'w')
f2 = open('testing_painting_label.csv', 'w')
f3 = open('painter_list.csv', 'w')
f.write('painting, label\n')
f2.write('painting, label\n')
f3.write('painter\n')

# write to csv file
for painting_fname in training_painting_fnames:
    for painter_name in painter_names:
        if painter_name in painting_fname:
            label = painter_label[painter_name]
            break
    print(f'painting file name: {painting_fname}, label: {label}')
    line = painting_fname + ', ' + str(label) + '\n'
    f.write(line)

for painting_fname in testing_painting_fnames:
    for painter_name in painter_names:
        if painter_name in painting_fname:
            label = painter_label[painter_name]
            break

    print(f'painting file name: {painting_fname}, label: {label}')
    line = painting_fname + ', ' + str(label) + '\n'
    f2.write(line)

for painter_name in painter_names:
    f3.write(painter_name + '\n')

f.close()
f2.close()
f3.close()