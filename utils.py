def save_img(x, path, annotation=''):
    fig = plt.gcf()  # generate outputs
    plt.imshow(tensor_to_img(x[0]), aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image

def tensor_to_img(x, imtype=np.uint8):
	mean = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
	std = [1. / 255., 1. / 255., 1. / 255.]

	if not isinstance(x, np.ndarray):
		if isinstance(x, torch.Tensor):  # get the data from a variable
			image_tensor = x.data
		else:
			return x
		image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
		if image_numpy.shape[0] == 1:  # grayscale to RGB
			image_numpy = np.tile(image_numpy, (3, 1, 1))
		for i in range(len(mean)):
			image_numpy[i] = image_numpy[i] * std[i] + mean[i]
		image_numpy = image_numpy * 255
		image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
	else:  # if it is a numpy array, do nothing
		image_numpy = x
	return image_numpy.astype(imtype)
