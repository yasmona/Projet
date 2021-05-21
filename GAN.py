from data_loader import data_loader as dl
from generateur import Generateur as G
from discriminateur import Discriminateur as D
from keras.layers import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import sentence_bleu

class GAN():
    def __init__(self, data):
        self.tokenizer = Tokenizer()
        self.list_tweets = dl.nettoyer_data(data)
        self.sequences, self.vocab_size = dl.tokenizer_data(self.list_tweets, self.tokenizer)
        self.batch_size = dl.max_tweet(self.sequences)
        self.X_train, self.y_train, self.nb_tweet = dl.preparer_data(self.sequences, self.vocab_size, self.batch_size)

    def modele_gan(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        # connect them
        x0 = Input(shape=(self.batch_size, 1))
        x1 = Input(shape=(self.batch_size,))
        x2 = generator([x0, x1])
        print(x2[-1].shape)
        x3 = discriminator([x2, x1])
        print(x3.shape)
        model = Model([x0, x1], x3)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def generer_tweet(self, epoch, noise, sampled_labels, g_model, tokenizer):
        result = g_model.predict([noise, sampled_labels])
        res = []
        predicted_word = ""
        for j in range(self.batch_size):
            for word, index in tokenizer.word_index.items():
                if index == abs(int(result[0][j])):
                    predicted_word = predicted_word + ' ' + word
                    break
        res.append(predicted_word)
        return res

    def entrainer(self):
        # Store the losses
        d_losses = []
        g_losses = []

        Epoch = 10
        nb=int(self.nb_tweet/2)

        X = self.sequences[:nb]
        X = pad_sequences(X, maxlen=self.batch_size, padding='post')
        y = self.sequences[nb:]
        y = pad_sequences(y, maxlen=self.batch_size, padding='post')
        X_train = X.reshape(nb, self.batch_size, 1)
        y_train = y.reshape(nb, self.batch_size, 1)
        real = np.ones(shape=(nb, 1))
        fake = np.zeros(shape=(nb, 1))
        # Build and compile the generator
        generator = G.modele_generateur(self.batch_size)
        # Build and compile the discriminator
        discriminator = D.modele_discriminateur(self.batch_size)
        # Build and compile the combined model
        gan_model = self.modele_gan(generator, discriminator)
        ######################## Main training loop###########################
        for i in range(Epoch):
            ########################
            ###Train Discriminator##
            #######################
            # prepare real samples
            [tweets, labels] = X_train, y_train
            lab = labels[-1].reshape(1, self.batch_size, 1)
            noise = np.random.randint(self.vocab_size, size=(1, self.batch_size, 1))

            # Generate fake tweets
            gen_twt = generator.predict([noise, lab])
            f = fake[-1].reshape(1, 1)
            # Train the discriminator
            # both loss and accuracy are returned
            d_loss_real, d_acc_real = discriminator.train_on_batch([tweets, labels], real)
            d_loss_fake, d_acc_fake = discriminator.train_on_batch([gen_twt, lab], f)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc = 0.5 * (d_acc_real + d_acc_fake)

            #######################
            ### Train generator ###
            #######################
            noise = np.random.randint(self.vocab_size, size=(nb, self.batch_size, 1))
            y_gan = np.ones((nb, 1))

            lab = labels[np.random.randint(0, nb)].reshape(1, self.batch_size, 1)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([noise, labels], y_gan)
            noise = np.random.randint(self.vocab_size, size=(1, self.batch_size, 1))
            # Save the losses
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            print("d_loss", d_loss)
            print("d_acc:", d_acc)
            print("g_loss", g_loss)
            tweet=self.generer_tweet(i, noise, lab, generator, self.tokenizer)
        return tweet

if __name__ == '__main__':
    t=open("tweet.txt", "r")
    l=t.read()
    gan=GAN(l)
    print(gan.training()[0])
    score = sentence_bleu(l, gan.training()[0])
    print(score)
