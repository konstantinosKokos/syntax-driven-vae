import numpy as np
import vocabulary_tools as vt
import pickle
import tqdm

def center_and_norm(comatrix):
	mu = np.mean(comatrix)
	comatrix = comatrix - mu
	std = np.std(comatrix)
	return comatrix/std	

def pca(comatrix):
	U, s, Vt = np.linalg.svd(comatrix)

def compress_pca(U, s, Vt, num_dim=300):
	S = np.diag(s[:num_dim])
	V = np.transpose(Vt)
	V = V[:num_dim, :num_dim])
	compressed = np.dot(U[:,:num_dim]), np.dot(S, V))
	return compressed

def reconstruct_pca(comatrix, U, s, Vt, num_dim=300):
	S = np.diag(s[:num_dim])
	V = np.transpose(Vt)
	V = V[:num_dim, :]
	USV = np.dot(U[:,:num_dim]), np.dot(S, V))
	return USV, np.mean((USV-comatrix)**2)

