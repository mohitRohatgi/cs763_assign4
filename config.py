class Config:
    def __init__(self):
        self.lr = 1e-3
        self.n_epoch = 500
        self.seq_length = 2948
        self.batch_size = 1
        self.n_layers = 1
        self.hidden_dim = 512
        self.num_classes = 2
        self.truncated_delta = 32
        self.vocab_size = 151
        self.embed_size = 151
        self.evaluate_every = 100
        self.bins = [50, 100, 150, 200, 250, 300, 250, 300, 350, 400, 450, 500, 550, 600, 650,
                     700, 800, 900, 1000, 1300, 1500, 2000, 2500]

    def __str__(self):
        return "{ lr:" + str(self.lr) + ", n_epoch = " + str(self.n_epoch) + ", batch_size = " + str(self.batch_size) +\
               ", n_layer = " + str(self.n_layers) + ", hidden_dim = " + str(self.hidden_dim) + ", truncated_delta = " \
               + str(self.truncated_delta) + ", bins = " + str(self.bins) + ", embed_size = " + str(self.embed_size) \
               + " }"
