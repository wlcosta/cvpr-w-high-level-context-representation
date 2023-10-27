import json
import torch
import dgl
import yaml
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import DiscreteLoss, ContinuousLoss_SL1, ContinuousLoss_L2
from utils.emotic import EmoticDataset, ind2cat, ind2vad, test_scikit_ap, test_vad
from models.graph_models import GIN, DefaultGCNModel, get_gcn_model
cfg = yaml.safe_load(open('config.yaml', 'r'))


def build_optimizer(params, optimizer, learning_rate, weight_decay):
    """Builds optimizer.

    Arguments:
        params -- model's parameters.
        optimizer -- optimizer name
        learning_rate -- chosen lr
        weight_decay -- chosen weight_decay

    Returns:
        built optimizer
    """
    if optimizer == 'adam':
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        return optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=learning_rate, weight_decay=weight_decay)


def main():
    train_cat = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_train_cat.npy'))
    test_cat = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_test_cat.npy'))
    val_cat = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_val_cat.npy'))
    train_cont = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_train_cont.npy'))
    test_cont = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_test_cont.npy'))
    val_cont = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_val_cont.npy'))

    train_graphs = dgl.data.utils.load_graphs(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_train_graph_nlp.bin'))

    test_graphs = dgl.data.utils.load_graphs(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_test_graph_nlp.bin'))

    val_graphs = dgl.data.utils.load_graphs(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_val_graph_nlp.bin'))

    train_context_nlp_caption = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_train_context_nlp_caption.npy'))

    test_context_nlp_caption = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_test_context_nlp_caption.npy'))

    val_context_nlp_caption = np.load(
        os.path.join('dataset',
                     cfg['dataset']['name']+cfg['dataset']['output_dir'],
                     cfg['dataset']['name']+'_val_context_nlp_caption.npy'))

    print(f"Loaded annotations.")
    print(len(train_cat))

    train_dataset = EmoticDataset(train_cat, train_cont)
    test_dataset = EmoticDataset(test_cat, test_cont)
    val_dataset = EmoticDataset(val_cat, val_cont)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Working on {device}')

    model = GIN(50, 40, 26)
    model = model.to(device)

    opt = build_optimizer(model.parameters(), "rmsprop", 0.0001, 0.0006)
    cont_loss = ContinuousLoss_SL1()
    disc_loss = DiscreteLoss("static")
    for epoch in range(200):
        running_loss = 0.0
        running_cat_loss = 0.0
        running_cont_loss = 0.0
        indx = 0

        cat_preds = np.zeros((len(train_dataset), 26))
        cat_labels = np.zeros((len(train_dataset), 26))
        cont_preds = np.zeros((len(train_dataset), 3))
        cont_labels = np.zeros((len(train_dataset), 3))

        model.train()

        with tqdm(total=len(train_loader), position=0, leave=True) as pbar:
            pbar.set_description(f"Train {epoch}/200")
            for label_cat, label_cont, indexes in iter(train_loader):
                opt.zero_grad()
                graph_batch = []
                for index in indexes:
                    graph = train_graphs[0][index]
                    graph = graph.to(device)
                    graph_batch.append(graph)
                graphs = dgl.batch(graph_batch)
                feats = graphs.ndata.pop('x')
                feats = feats.type(torch.float)
                feats = feats.to(device)
                pred_cat, pred_cont = model(graphs, feats)
                pred_cat = pred_cat.to("cpu")
                pred_cont = pred_cont.to("cpu")
                disc_loss_ = disc_loss(pred_cat, label_cat)
                cont_loss_ = cont_loss(pred_cont*10, label_cont*10)
                loss = (0.3*disc_loss_) + (1-0.3)*cont_loss_
                running_loss += loss.item()
                running_cat_loss += disc_loss_.item()
                running_cont_loss += cont_loss_.item()

                loss.backward()
                opt.step()

                cat_preds[indx: (indx + pred_cat.shape[0]),
                          :] = pred_cat.to("cpu").data.numpy()
                cat_labels[indx: (indx + label_cat.shape[0]),
                           :] = label_cat.to("cpu").data.numpy()
                cont_preds[indx: (indx + pred_cont.shape[0]),
                           :] = pred_cont.to("cpu").data.numpy() * 10
                cont_labels[indx: (indx + label_cont.shape[0]),
                            :] = label_cont.to("cpu").data.numpy() * 10
                indx = indx + pred_cat.shape[0]
                pbar.update(1)

        cat_preds = cat_preds.transpose()
        cat_labels = cat_labels.transpose()
        cont_preds = cont_preds.transpose()
        cont_labels = cont_labels.transpose()

        train_map = test_scikit_ap(cat_preds, cat_labels, ind2cat)
        train_vad = test_vad(cont_preds, cont_labels, ind2vad)

        print(
            f'Epoch: {epoch} :: train_mAP: {train_map}\tloss: {running_loss}')

        running_loss = 0.0
        running_cat_loss = 0.0
        running_cont_loss = 0.0
        indx = 0

        cat_preds = np.zeros((len(test_dataset), 26))
        cat_labels = np.zeros((len(test_dataset), 26))
        cont_preds = np.zeros((len(test_dataset), 3))
        cont_labels = np.zeros((len(test_dataset), 3))

        model.eval()
        with tqdm(total=len(test_loader), position=1, leave=True) as pbar:
            with torch.no_grad():
                pbar.set_description(f"Test {epoch}/{200}")
                for label_cat, label_cont, indexes, test_caption in iter(test_loader):
                    graph_batch = []
                    for index in indexes:
                        graph = test_graphs[0][index]
                        graph = graph.to(device)
                        graph_batch.append(graph)
                    graphs = dgl.batch(graph_batch)
                    feats = graphs.ndata.pop('x')
                    feats = feats.type(torch.float)
                    feats = feats.to(device)
                    pred_cat, pred_cont = model(graphs, feats)
                    pred_cat = pred_cat.to("cpu")
                    pred_cont = pred_cont.to("cpu")
                    disc_loss_ = disc_loss(pred_cat, label_cat)
                    cont_loss_ = cont_loss(pred_cont*10, label_cont)
                    loss = (0.3*disc_loss_) + (1-0.3)*cont_loss_
                    running_loss += loss.item()
                    running_cat_loss += disc_loss_.item()
                    running_cont_loss += cont_loss_.item()

                    cat_preds[indx: (indx + pred_cat.shape[0]),
                              :] = pred_cat.to("cpu").data.numpy()
                    cat_labels[indx: (indx + label_cat.shape[0]),
                               :] = label_cat.to("cpu").data.numpy()
                    cont_preds[indx: (indx + pred_cont.shape[0]),
                               :] = pred_cont.to("cpu").data.numpy() * 10
                    cont_labels[indx: (indx + label_cont.shape[0]),
                                :] = label_cont.to("cpu").data.numpy() * 10
                    indx = indx + pred_cat.shape[0]
                    pbar.update(1)

        cat_preds = cat_preds.transpose()
        cat_labels = cat_labels.transpose()
        cont_preds = cont_preds.transpose()
        cont_labels = cont_labels.transpose()

        test_map_v = test_scikit_ap(cat_preds, cat_labels, ind2cat)
        test_vad_v = test_vad(cont_preds, cont_labels, ind2vad)

        print(
            f'Epoch: {epoch} :: test_mAP: {test_map_v}\tloss: {running_loss}')

    torch.save(model.state_dict(), 'model_graphs.pth')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    predictions_dict = {}
    for label_cat, label_cont, index, test_caption in iter(test_loader):
        graphs = test_graphs[0][index]
        feats = graphs.ndata.pop('x')
        feats = feats.type(torch.float)
        feats = feats.to(device)
        pred_cat, pred_cont = model(graphs, feats)
        pred_cat = pred_cat.to("cpu")
        pred_cont = pred_cont.to("cpu")

        predictions_dict[index] = {
            'caption': test_caption,
            'label_cat': label_cat,
            'pred_cat': pred_cat
        }

    json_ = json.dumps(predictions_dict)
    with open('predictions.json', 'w') as f:
        f.write(json_)


if __name__ == '__main__':
    main()
