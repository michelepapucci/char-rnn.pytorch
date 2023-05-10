# https://github.com/michelepapucci/char-rnn.pytorch/

import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm

from helpers import *
from model import *
from generate import *


def random_training_set(chunk_len, batch_size, file, file_len, use_cuda = False):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - (chunk_len + 1))
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        if(len(chunk[:-1]) < 50):
            print(chunk[:-1])
        elif(len(chunk[1:]) < 50):
            print(chunk[1:])
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if use_cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target, criterion, decoder, decoder_optimizer, batch_size, chunk_len, use_cuda=False):
    hidden = decoder.init_hidden(batch_size)
    if use_cuda:
        print(type(hidden))
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / chunk_len

def save(decoder):
    save_filename = 'model.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


def main(): 
    filepath = "training.csv"
    use_cuda = False
    batch_size = 500
    chunk_len = 50
    hidden_size = 200
    model = 'lstm'
    n_layers = 4
    learning_rate = 0.01
    n_epochs = 3000 # TODO provare ad aumentare? 
    print_every = 100

    file, file_len = read_file(filepath)

    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model=model,
        n_layers=n_layers,
    )
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        decoder.cuda()

    start = time.time()
    all_losses = []
    loss_avg = 0

    try:
        print("Training for %d epochs..." % n_epochs)
        for epoch in tqdm(range(1, n_epochs + 1)):
            loss = train(*random_training_set(chunk_len, batch_size, file, file_len), criterion, decoder, decoder_optimizer, batch_size, chunk_len)
            loss_avg += loss

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
                print(generate(decoder, 'Il', 100, cuda=use_cuda, stop_char="\n"))

        print("Saving...")
        save(decoder)
    except KeyboardInterrupt:
        print("Saving before quit...")
        save(decoder)

if __name__ == "__main__":
    main()
