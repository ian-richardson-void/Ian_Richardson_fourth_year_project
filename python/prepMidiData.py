import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import scandir
from music21 import converter, instrument, note, chord

fileName = "../training-data/beethoven" # Name of directory of data  
s_l = 128 # Sequence length of data

def midRead(name):
	# read midi files from directory, return midi objects
	files = scandir.scandir(name)
	
	midfiles = []
	for f in files:
		print "reading", f.path
		# convert using music21
		mmid = converter.parse(f.path) # this line takes a while
		midfiles.append(mmid)
	
	return midfiles
	
def getNotes(midfiles):
	# returns a list of all notes from a list of midi objects
	notes = []
	for f in midfiles:
		notes_to_parse = None
		try: # multiple instument parts
			partish = instrument.partitionByInstrument(f)
			notes_to_parse = partish.parts[0].recurse()
		except: # flat structure
			notes_to_parse = f.flat.notes
		
		for element in notes_to_parse:
			if isinstance(element, note.Note): # parse note
				notes.append(str(element.pitch))
			elif isinstance(element, chord.Chord): # parse chord
				notes.append('.'.join(str(n) for n in element.normalOrder))
	
	return notes
		
def data():
	# returns data sequences for use with neural network
	midFiles = midRead(fileName)
	notes = getNotes(midFiles)
	
	chars = sorted(list(set(notes))) # list of unique notes
	n_volcab = len(chars) # number of unique notes
	n_chars = len(notes) # total number of notes
	
	# create dictionary
	noteToInt = dict((n, i) for i, n in enumerate(chars))
	intToNote = dict((i, n) for i, n in enumerate(chars))
	
	# cutting data into sequence length for input x -> output y
	seq_length = s_l
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length):
		seq_in = notes[i:i + seq_length]
		seq_out = notes[i + seq_length]
		dataX.append([noteToInt[char] for char in seq_in])
		dataY.append(noteToInt[seq_out])
	n_patterns = len(dataX)
	
	# data in a format for network
	x = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	x = x/numpy.float32(n_volcab)
	y = np_utils.to_categorical(dataY)
	
	return x, y, dataX, n_volcab, intToNote

def getModel(x, y, n_volcab):
	# creates and returns Sequential LSTM model
	model = Sequential()
	model.add(LSTM(
		256, 
		input_shape=(x.shape[1], x.shape[2]), 
		return_sequences=True
	))
	model.add(Dropout(0.3))
	model.add(LSTM(
		512, 
		return_sequences=True
	))
	model.add(Dropout(0.3))
	model.add(LSTM(256))
	model.add(Dense(256))
	model.add(Dropout(0.3))
	model.add(Dense(n_volcab))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model