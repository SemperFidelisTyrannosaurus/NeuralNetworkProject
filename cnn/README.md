# Train the model with the existing data set

Just run `python dna_cnn.py`.
To monitor the model run `tensorboard` in another terminal.
tensorboard will print a URL in the terminal. Copy that URL and paste it in a 
browser.

# Generate a new dataset

Import the `dna_dataset` module.
Call `dna_dataset.gen_datafiles()` from a script or python prompt.