4th year Individual project ROUS03
Using Deep Neural Networks to Generate Music
Ian Richardson 201603535

- There are 3 Python2 files in the directory "python": "training.py" and "generation.py" can be run from the console, "prepMidiData.py" will 
  only be used as a helper file for the other 2. 

- To run the project, the python files require you to have modules "numpy", "keras", "scandir" and "music21" installed, all available through pip. 

- Running the python file "training.py" will train the neural network on a given data set (specified at the top of "prepMidiData.py") and write 
  some weight checkpoint files out as ".hdf5" into the "weight-gen" directory, this may take some time especially on large data sets like 
  "training-data/beethoven", so I have provided a checkpoint file I trained on this data earlier (like blue peter), called 
  "beethoven-S128-B128-L0.3661.hdf5" in the "weights" directory. 

- Running the python file "generation.py" will use this checkpoint file (specified at the top of the file) and the data set to create a new 
  MIDI file in the directory "my-generated" named "beethoven-S128-B128-L0.3661-.mid" (also specified at the top of the file), this step can 
  be repeated with the same checkpoint file for a new different midi file, although be careful of overwritting previously generated music 
  with more files named the same. 

- The folder "my-generated" contains some MIDI files I have generated earlier (also like blue peter) that I liked, using the "beethoven" data 
  set and weights provided. 

- MIDI files can be opened and listened to on "Windows Media Player" which comes default on windows, on Linux a third party program will be 
  required to listen, I have been using a program called "timidity" which allows running the files from the command line.