from scipy.spatial.distance import cosine
from keras.layers import Input, Dense, LSTM, Reshape
from keras.models import Model
from keras import backend as K
from keras.optimizers import adam
from sklearn.model_selection import train_test_split

def cosine_proximity(y_true, y_pred):
	y_true = K.l2_normalize(y_true, axis=-1)
	y_pred = K.l2_normalize(y_pred, axis=-1)
	return 1 - K.sum(y_true * y_pred, axis=-1)

def construct_LSTM_AE(comatrix, lstm_size, dense_sizes):
	x = Input(shape=(comatrix.shape[0],))
	reshape = Reshape(1, comatrix.shape[0]),)(x)
	h = LSTM(lstm_size, activation='tanh')(reshape)
	e = Dense(300, activation='tanh')(h)
	y = Dense(comatrix.shape[0], activation='tanh')(e)
	simple = Model(inputs=[x], outputs=[y])
	encoder = Model(inputs[x], outputs=[y])
	decoder = Model(inputs=[e], outputs=[y])
	simple.compile(optimizer=adam(), loss=cosine_proximity)
	simple.summary()
	encoder.compile(optimizer=adam(), loss=cosine_proximity)
	decoder.compile(optimizer=adam(), loss=binary_crossentropy)
	return simple, encoder, decoder

def train_LSTM_AE(simple, comatrix):
	h = simple.fit(comatrix, comatrix, shuffle=True, epochs=200, batch_size=64, verbose=1)
	return h

