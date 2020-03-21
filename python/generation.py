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
from music21 import instrument, note, stream, chord

weightsName = "../weights/beethoven-S128-B128-L0.3661.hdf5" # name of weight checkpoint file
outputFileName = "../my-generated/beethoven-S128-B128-L0.3661-.mid" # name of file to be outputted

def gen(x, y, dataX, n_volcab, intToNote):
	# creation of lstm neural net
	model = prepMidiData.getModel(x, y, n_volcab)
	
	# load weights from file
	print "loading weights", weightsName
	model.load_weights(weightsName)
	
	# pick a seed
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	print pattern
	print "Seed:"
	print "\"", ''.join([intToNote[value] for value in pattern]), "\""
	
	# generate notes
	newNotes = []
	for i in range(500):
		X = numpy.reshape(pattern, (1, len(pattern), 1))
		X = X / float(n_volcab)
		prediction = model.predict(X, verbose=0)
		
		index = numpy.argmax(prediction)
		note = intToNote[index]
		newNotes.append(note)
		
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	
	print "Generation Complete, writing out to file "
	return newNotes
	
def createMidiFile(newNotes):
	offset = 0
	midOutput = []
	
	# loop through output and convert to midi objects
	for prediction in newNotes:
		# if chord
		if ("." in prediction) or prediction.isdigit():
			# getting notes in chord
			chordNotes = []
			for n in prediction.split("."):
				noteObj = note.Note(int(n))
				noteObj.storedInstrument = instrument.Piano()
				chordNotes.append(noteObj)
			
			# creating new chord object, appended to the output
			chordObj = chord.Chord(chordNotes)
			chordObj.offset = offset
			midOutput.append(chordObj)
		# if note
		else:
			# creating new note object, appended to the output
			noteObj = note.Note(prediction)
			noteObj.offset = offset
			noteObj.storedInstrument = instrument.Piano()
			midOutput.append(noteObj)
		
		offset += 0.5
	
	# outputting to midi file
	midStream = stream.Stream(midOutput)
	midStream.write("midi", fp=outputFileName)
	print "file written out"
	
x, y, dataX, n_volcab, intToNote = prepMidiData.data()
newNotes = gen(x, y, dataX, n_volcab, intToNote)
createMidiFile(newNotes)