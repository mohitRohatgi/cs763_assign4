class Config:
    def __init__(self):
        self.lr = 1e-4
        self.n_epoch = 200
        self.seq_length = 2948
        self.batch_size = 1
        self.n_layers = 1
        self.hidden_dim = 100
        self.num_classes = 2
        self.truncated_delta = 32
        self.vocab_size = 151
        self.embed_size = 151
        self.evaluate_every = 100
        self.bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                     210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
                     410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
                     610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 730, 750, 800, 830, 860, 900, 930, 960, 1000,
                     1030, 1060, 1100]

    def __str__(self):
        return "{ lr:" + str(self.lr) + ", n_epoch = " + str(self.n_epoch) + ", batch_size = " + str(self.batch_size) +\
               ", n_layer = " + str(self.n_layers) + ", hidden_dim = " + str(self.hidden_dim) + ", truncated_delta = " \
               + str(self.truncated_delta) + ", bins = " + str(self.bins) + ", embed_size = " + str(self.embed_size) \
               + " }"
