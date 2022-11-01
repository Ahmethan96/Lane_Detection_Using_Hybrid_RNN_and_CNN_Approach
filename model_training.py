from dataset import RoadSequenceDataset, RoadSequenceDatasetList
from mod import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
from configuration import class_weights
import torch
import time
from configuration import arguments_setup


def training(arguments, epoch, model, train_loader, pc, opti, criter):
    time_past = time.time()
    model.training()
    for batch_idx,  sample_batched in enumerate(train_loader):
        data, target = sample_batched['data'].to(pc), sample_batched['label'].type(torch.LongTensor).to(pc)# LongTensor
        opti.zero_grad()
        output, x = mod(data)
        los = criter(output, target)
        los.backward()
        opti.step()
        if batch_idx % arguments.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), los.item()))

    time_1 = time.time() - time_past
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
                                                               time_1 // 60, time_1 % 60))
def parameters(model, layer):
    import torch.nn as neural
    modules = (neural.ReLU, neural.MaxPool2d, neural.Dropout2d, neural.UpsamplingBilinear2d)
    for call, module in model.named_children():
        if call in layer:
            for layer in module.children():
                if isinstance(layer, modules):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma
def validation(arguments, mod, validation_loader, pc, criter, best_acc):
    mod.eval()
    test_loss = 0
    correct_samples = 0
    with torch.no_grad():
        for sample_batched in validation_loader:
            data, target = sample_batched['data'].to(pc), sample_batched['label'].type(torch.LongTensor).to(pc)
            output,_ = mod(data)
            test_loss += criter(output, target).item()  # sum up batch loss
            prediction = output.max(1, keepdim=True)[1]
            correct_samples += prediction.eq(target.view_as(prediction)).sum().item()
    test_loss /= (len(validation_loader.dataset) / arguments.test_batch_size)
    val_accuracy = 100. * int(correct_samples) / (len(validation_loader.dataset) * configuration.label_height * configuration.label_width)
    print('\nAvg loss: , Accuracy: \n')


if __name__ == '__main__':
    arguments = arguments_setup()
    torch.manual_seed(arguments.seed)
    cuda = arguments.cuda and torch.cuda.is_available()
    pc = torch.device("cuda" if cuda else "cpu")

    # transform pictures into floatTensor
    tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    if arguments.mod == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
        loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=configuration.train_path, transforms=tranforms),
            batch_size=arguments.batch_size,shuffle=True,num_workers=configuration.data_loader_numworkers)
        validation_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=configuration.val_path, transforms=tranforms),
            batch_size=arguments.test_batch_size,shuffle=True,num_workers=configuration.data_loader_numworkers)
    else:
        loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=configuration.train_path, transforms=tranforms),
            batch_size=arguments.batch_size, shuffle=True, num_workers=configuration.data_loader_numworkers)
        validation_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=configuration.val_path, transforms=tranforms),
            batch_size=arguments.test_batch_size, shuffle=True, num_workers=configuration.data_loader_numworkers)

    #loading the model
    best_accurcy = 0
    mod = generate_model(arguments)
    opti = torch.optim.Adam(mod.parameters(), lr=arguments.lr)
    schedule = lr_scheduler.StepLR(opti, step_size=1, gamma=0.5)
    class_weights = torch.tensor(configuration.class_weights)
    criter = torch.nn.CrossEntropyLoss(weight=class_weights).to(pc)

    print(type(class_weights))
    print(type(criter))
    pretrained_dictonary = torch.load(configuration.pretrained_path)
    model_dictionary = mod.state_dict()

    pretrained_dictionary_1 = {k: v for k, v in pretrained_dictonary.items() if (k in model_dictionary)}
    model_dictionary.update(pretrained_dictionary_1)
    mod.load_state_dict(model_dictionary)

    # train phase
    for epoch in range(1, arguments.epochs + 1):
        schedule.step()
        training(arguments, epoch, mod, loader, pc, opti, criter)
        validation(arguments, mod, validation_loader, pc, criter, best_accurcy)