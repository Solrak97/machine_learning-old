import torchvision
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from models import Generator, Discriminator
from torch.utils.data import DataLoader
from dataset import MNIST

# save it as a gif
# Esta idea se la robe a Juan
from moviepy.editor import ImageSequenceClip


# Esta wea no funciona, voy a escribir una desde 0
def model_accuracy(y_pred, y):
    y_pred = y_pred > 0.5
    return (y == y_pred).sum().item() / y.size(0)


def train_discriminator(opt, model, x_true, x_false, accuracy=None, max_iters=100, batch_size=1000):
    true_labels = torch.ones(x_true.shape[0], dtype=torch.float32)
    false_labels = torch.zeros(x_false.shape[0], dtype=torch.float32)

    x = torch.cat((x_true, x_false))
    y = torch.unsqueeze(torch.cat((true_labels, false_labels)), dim=1)

    model_data = DataLoader(MNIST(x, y), batch_size, shuffle=True)
    loss_fn = BCELoss()

    for epoch in range(max_iters):
        iteration_accuracy = 0
        iteration_loss = 0

        if accuracy is not None and iteration_accuracy > accuracy:
            break

        else:
            for i, data in enumerate(model_data):
                opt.zero_grad()
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                pred = model(images)
                loss = loss_fn(pred, labels)
                iteration_loss += loss
                iteration_accuracy += model_accuracy(pred, labels)

                loss.backward()
                opt.step()
    pass


def train_generator(opt, generator, classifier, accuracy=None, max_iters=100, batch_size=1000):
    batches = 10

    noise_loader = DataLoader(torch.rand(
                (batch_size * batches, 48, 1, 1)), batch_size)


    for epoch in range(max_iters):
        iteration_accuracy = 0
        loss_fn = BCELoss()

        if accuracy is not None and iteration_accuracy > accuracy:
            break

        else:            
            for i, data in enumerate(noise_loader):
                opt.zero_grad()
                images = data
                images = images.to(device)
                fake_labels = torch.ones((images.shape[0], 1), dtype=torch.float32).to(device)

                generated = generator(images)

                pred = classifier(generated)

                loss = loss_fn(pred, fake_labels)

                iteration_accuracy += model_accuracy(pred, fake_labels)

                loss.backward()
                opt.step()
    pass


# Data
train = torchvision.datasets.MNIST(".", download=True)
x = train.data.float()
x = torch.unsqueeze(x, dim=1)
y = train.targets

# Data separation
class_indexes = ((y == 5).nonzero(as_tuple=True)[0])
chosen_x = x[class_indexes]

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
gen = Generator()
disc = Discriminator()

# Model to device
device_generator = gen.to(device)
device_discriminator = disc.to(device)

# Model optimizers
disc_optimizer = Adam(device_discriminator.parameters(), 1e-4)
gen_optimizer = Adam(device_generator.parameters(), 1e-4)


epochs = 50
imgs = []
for epoch in range(epochs):
    print(f'Epoca #{epoch}')

    print('     Generando ruido')
    random_noise = torch.rand(chosen_x.shape[0], 48, 1, 1).to(device)
    generated_batch = device_generator(random_noise).cpu().detach()

    print('     Entrenando discriminador')
    train_discriminator(disc_optimizer, device_discriminator,
                        chosen_x, generated_batch, max_iters = 100)

    print('     Entrenando generador')
    train_generator(gen_optimizer, device_generator,
                    device_discriminator, max_iters = 100)

    print('     Generando imagen aleatoria')
    random_img_noise = torch.rand(1, 48, 1, 1).to(device)
    img = device_generator(random_img_noise)
    imgs.append(img.cpu().detach().numpy().reshape((28, 28, 1)))


print('Generando gif')
clip = ImageSequenceClip(imgs, fps=20)
clip.write_gif('1.gif', fps=5)
