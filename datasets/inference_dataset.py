from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import dlib
import torch
from scripts.align_all_parallel import align_face


class InferenceDataset(Dataset):

	def __init__(self, root, opts, predictor_path, transform=None):
		self.paths = self._get_paths(root)
		print(self.paths)
		self.transform = transform
		self.opts = opts
		self.predictor = dlib.shape_predictor(predictor_path)

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		try:
			from_path = self.paths[index]
			# from_im = Image.open(from_path)
			# from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
			from_im = align_face(filepath=from_path, predictor=self.predictor)
			if self.transform:
				from_im = self.transform(from_im)
			return from_im
		except Exception as e:
			print(e)

	def _get_paths(self, root):
		paths = sorted(data_utils.make_dataset(root))
		result_paths = []
		for path in paths:
			# try:
				align_face(filepath=path, predictor=self.predictor)
				result_paths.append(path)
			# except Exception as e:
			# 	print(e)
		return result_paths

