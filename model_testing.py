import torch
from configuration import arguments_setup
from dataset import RoadSequenceDataset, RoadSequenceDatasetList
from mod_1 import generate_model
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn as neural
def result(model_1, testing_loader, pc):
    model_1.eval()
    i = 0
    dic=[]
    with torch.no_grad():
        for batches in testing_loader:
            i+=1
            print(i)
            data_test, target = batches['data'].to(pc), batches['label'].type(torch.LongTensor).to(pc)
            output,feature = model_1(data_test)
            print(type(feature))
            dic.append(feature)
            prd = output.max(1, keepdim=True)[1]
            pict = torch.squeeze(prd).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            pict = Image.fromarray(pict.astype(np.uint8))

            data_test = torch.squeeze(data_test).cpu().numpy()
            if arguments.mod_1 == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
                data_test = np.transpose(data_test[-1], [1, 2, 0]) * 255
            else:
                data_test = np.transpose(data_test, [1, 2, 0]) * 255
            data_test = Image.fromarray(data_test.astype(np.uint8))
            row = pict.size[0]
            col = pict.size[1]
            for i in range(0, row):
                for j in range(0, col):
                    pict2 = (pict.getpixel((i, j)))
                    if (pict2[0] > 200 or pict2[1] > 200 or pict2[2] > 200):
                        data_test.putpixel((i, j), (234, 53, 57, 255))
            data_test = data_test.convert("RGB")
            data_test.save(config.save_path + "%s_data.jpg" % i)#red line on the original image
            pict.save(config.save_path + "%s_pred.jpg" % i)#prediction result

def evaluation_of_model(mod, testing_loader, pc, criter):
    test_loss = 0
    correct = 0
    error = 0
    mod.eval()
    i = 0
    precision_per = 0.0
    recall_sco = 0.0

    with torch.no_grad():
        for sample_batched in testing_loader:
            i+=1
            data, target = sample_batched['data'].to(pc), sample_batched['label'].type(torch.LongTensor).to(pc)
            output, feature = mod(data)
            pred = output.max(1, keepdim=True)[1]
            pict = torch.squeeze(pred).cpu().numpy()*255
            lab = torch.squeeze(target).cpu().numpy()*255
            pict = img.astype(np.uint8)
            lab = lab.astype(np.uint8)
            kernel = np.uint8(np.ones((3, 3)))

            test_loss += criter(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            label_precision = cv2.dilate(lab, kernel)
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            a = len(np.nonzero(img*label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b==0:
                error= error + 1
                continue
            else:
                precision_per += float(a / b)
            m = len(np.nonzero(pred_recall * lab)[1])
            g = len(np.nonzero(lab)[1])
            if g==0:
                error = error + 1
                continue
            else:
                recall_sco += float(m / g)
            F1_score= (2 * precision_per * recall_sco) / (precision_per + recall_sco)

    test_loss /= (len(testing_loader.dataset) / arguments.test_batch_size)
    test_acc = 100. * int(correct) / (len(testing_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(
        test_loss, int(correct), len(testing_loader.dataset), test_acc))

    precision_per = precision_per / (len(testing_loader.dataset) - error)
    recall_sco = recall_sco / (len(testing_loader.dataset) - error)
    F1_score = F1_score / (len(testing_loader.dataset) - error)
    print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision_per, recall_sco, F1_score))

def parameters(mod, layer_name):
    modules_skipped = (neural.ReLU,neural.MaxPool2d,neural.Dropout2d,neural.UpsamplingBilinear2d)
    for call_2, module in mod.named_children():
        if call_2 in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma

if __name__ == '__main__':
    arguments = arguments_setup()
    torch.manual_seed(arguments.seed)
    cuda = arguments.cuda and torch.cuda.is_available()
    pc = torch.device("cuda" if cuda else "cpu")

    o_tranforms = transforms.Compose([transforms.ToTensor()])

    if arguments.mod_1 == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
        test_loader=torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=config.test_path, transforms=o_tranforms),
            batch_size=arguments.test_batch_size, shuffle=False, num_workers=1)
    else:
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=config.test_path, transforms=o_tranforms),
            batch_size=arguments.test_batch_size, shuffle=False, num_workers=1)

    mod_1 = generate_model(arguments)
    class_weight = torch.Tensor(config.class_weight)
    criter = torch.nn.CrossEntropyLoss(weight=class_weight).to(pc)

    pretrained_dict = torch.load(config.pretrained_path)
    model_dictionary = mod_1.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dictionary)}
    model_dictionary.update(pretrained_dict_1)
    mod_1.load_state_dict(model_dictionary)
    result(mod_1, test_loader, pc)
    evaluation_of_model(mod_1, test_loader, pc, criter)
