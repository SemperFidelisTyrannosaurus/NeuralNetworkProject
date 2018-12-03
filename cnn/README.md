# Train the model with the existing data set

Just run `python dna_cnn.py`.
To monitor the model run `tensorboard` in another terminal.
tensorboard will print a URL in the terminal. Copy that URL and paste it in a browser.

# Generate a new dataset

Import the `dna_dataset` module.
Call `dna_dataset.gen_datafiles()` from a script or python prompt.

# Generating k-mer sequences

To generating k-mer sequences for a new set of filters call the `dna_dataset.gen_kmer_data(k)` with the new k value. The sequences are saved using [pickle](https://docs.python.org/3/library/pickle.html) as sorted lists for each class of data, and can be loaded with `dna_dataset.load_kmer_data(k)`.

To use this new file, change the `k` variable in the `dna_cnn.cnn_model_fn`.