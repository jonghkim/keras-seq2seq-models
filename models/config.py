class VanillaConfig():
    
    MODE = "train"

    BATCH_SIZE = 64  # Batch size for training.
    EPOCHS = 100  # Number of epochs to train for.
    LATENT_DIM = 256  # Latent dimensionality of the encoding space.
    NUM_SAMPLES = 700  # Number of samples to train on. 
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100

    def __repr__(self):
        arttribute = vars(self)
        arttribute = {k:v for k,v in arttribute.items() if not k.startswith("__")}
        return str(arttribute)
