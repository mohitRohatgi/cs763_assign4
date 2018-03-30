class Config:
    def __init__(self):
        self.lr = 1e-5
        self.n_epoch = 5000
        self.seq_length = 2948
        self.batch_size = 64
        self.n_layers = 3
        self.hidden_dim = 256
        self.num_classes = 2
        self.truncated_delta = 10
        self.vocab_size = 151
        self.embed_size = 151
        self.num_steps = 1500
        self.evaluate_every = 30

    def __str__(self):
        return "{ lr:" + str(self.lr) + ", n_epoch = " + str(self.n_layers) + ", batch_size = " + str(self.batch_size) +\
               ", n_layer = " + str(self.n_layers) + ", hidden_dim = " + str(self.hidden_dim) + ", truncated_delta = " \
               + str(self.truncated_delta) + ", num steps = " + str(self.num_steps) + ", embed_size = " + \
               str(self.embed_size) + " }"
