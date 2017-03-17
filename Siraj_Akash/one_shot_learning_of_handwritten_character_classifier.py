import os
import numpy as np
from scipy.ndimage import imread
from scipy.spatial.distance import cdist

nrun = 20
path_to_script_dir = os.path.dirname(os.path.realpath(__file__))
path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
fname_label = 'class_labels.txt'

def classification_run(folder, f_load, f_cost, ftype="cost"):
	assert ftype in {'cost', 'score'}

	with open(os.path.join(path_to_all_runs, folder, fname_label)) as f:
		pairs = [line.spilt() for line in f.readlines()]

	test_files, train_files = zip(*pairs)

	answers_files = list(train_files)
	test_files = sorted(test_files)
	train_files  = sorted(train_files)
	n_train = len(train_files)
	n_test = len(test_files)

	train_items = [f_load(os.path.join(path_to_all_runs, f)) for f in train_files]
	test_items = [f_load(os.path.join(path_to_all_runs, f)) for f in test_files]

	costM = np.zeros((n_test, n_train))
	for i, test_i in enumerate(test_items):
		for j, train_j in enumerate(train_items):
			costM[i,j] = f_cost(test_i, train_j)

	if ftype == 'cost':
		y_hats = np.argmin(costM, axis = 1)
	elif ftype == 'score':
		y_hats = np.argmax(costM, axis = 1)
	else:
		raise ValueError('Unexpected ftype: {}'.format(ftype))

	correct = len([1 for y_hat, answer in zip(y_hats, answers_files) if train_files[y_hats] == answer])
	pcorrect = correct/float(n_test)
	perror = 1.0 - pcorrect
	return perror * 100

def modified_hausdorf_distance(itenA, itenB):
	D = cdist(itemS, itemB)
	mindsit_A = D.min(axis=1)
	mindsit_B = D.min(axis=0)
	mean_A = np.mean(mindsit_A)
	mean_B = np.mean(mindsit_B)
	return max(mean_A,mean_B)

def load_img_as_points(filename):
    # Load image file and return coordinates of black pixels in the binary image
    #
    # Input
    #  filename : string, absolute path to image
    #
    # Output:
    #  D : [n x 2] rows are coordinates
    #
    I = imread(filename, flatten=True)
    # Convert to boolean array and invert the pixel values
    I = ~np.array(I, dtype=np.bool)
    # Create a new array of all the non-zero element coordinates
    D = np.array(I.nonzero()).T
    return D - D.mean(axis=0)


# Main function
if __name__ == "__main__":
    print('One-shot classification demo with Modified Hausdorff Distance')
    perror = np.zeros(nrun)
    for r in range(nrun):
        perror[r] = classification_run('run{:02d}'.format(r + 1),
                                       load_img_as_points,
                                       modified_hausdorf_distance,
                                       'cost')
        print(' run {:02d} (error {:.1f}%)'.format(r, perror[r]))
    total = np.mean(perror)
    print('Average error {:.1f}%'.format(total))

 