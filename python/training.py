import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import prepMidiData

def train(x, y, n_volcab):
	# creation of lstm neural net
	model = prepMidiData.getModel(x, y, n_volcab)
	
	filepath = "../weight-gen/{epoch:02d}-beethoven-S128-B128-L{loss:.4f}.hdf5" # destination and name of saved checkpoint files
	checkpoint = ModelCheckpoint(
		filepath, 
		monitor="loss", 
		verbose=0, 
		save_best_only=True, 
		mode="min"
	)
	callbacks_list = [checkpoint]
	
	# runs the training on the data set
	model.fit(
		x, 
		y, 
		epochs=240, 
		batch_size=128, 
		callbacks=callbacks_list
	)

x, y, dataX, n_volcab, intToNote = prepMidiData.data()
train(x, y, n_volcab)