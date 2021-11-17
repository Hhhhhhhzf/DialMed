"""
@Time: 2021/11/15 15:49
@desc:
"""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def train():
    setup_seed(args.seed)
    model_path = ''
    kg_info = KGInfo('')
    kg_embedding = pickle.load(open('', 'rb'))
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    label_data = LabelData(label_path='')
    train_dataset, dev_dataset, test_dataset = PretrainedDataset(os.path.join(args.data_path, 'train.txt'), tokenizer,
                                                                 kg_info, label_data), \
                                               PretrainedDataset(os.path.join(args.data_path, 'dev.txt'), tokenizer,
                                                                 kg_info, label_data), \
                                               PretrainedDataset(os.path.join(args.data_path, 'test.txt'), tokenizer,
                                                                 kg_info, label_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=PretrainedDataset.pretrained_collate,
                                  pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True,
                                collate_fn=PretrainedDataset.pretrained_collate,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=PretrainedDataset.pretrained_collate,
                                 pin_memory=True)
    hyper_parameters = {'epochs': args.epoch, 'lr': args.lr}
    device = torch.device('cuda')
    if args.trained_model == '':
        pretrained_model = AutoModel.from_pretrained(model_path, config=config)
    else:
        pretrained_model = load_model(os.path.join('', args.trained_model), map_location=lambda storage, loc: storage.cuda())
        if hasattr(pretrained_model, 'module'):
            pretrained_model = getattr(pretrained_model, 'module').model.to(device)
        else:
            pretrained_model = pretrained_model.to(device)
    model = DDN(
        model=pretrained_model,
        n_label=len(label_data.label_to_id),
        hidden_size=config.hidden_size,
        output_size=500,
        entity_embedding=kg_embedding['entity_embedding'],
        relation_embedding=kg_embedding['relation_embedding'],
        entity_nums=len(kg_info.entity2id),
        relation_nums=len(kg_info.id2relation),
        graph_adj=kg_info.graph_adj.to(device),
        dim=500,
        config=config).to(device)
    model = DataParallel(model)
    best_model = None
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gama)
    loss_func = torch.nn.BCELoss().to(device)
    epochs = hyper_parameters['epochs']
    idx_train = idx_dev = 0
    max_dev_jaccard = 0
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    logger = MyLogger(os.path.join(args.log_name+time_str+'.log'))
    logger.log(str(sys.argv))
    logger.log(str(model))
    test_loss, test_jaccard = [], []
    test_p, test_r, test_f1 = [], [], []
    for epoch in range(epochs):
        logger.log('epoch: {}'.format(epoch))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        train_loss, train_jaccard, dev_loss, dev_jaccard = [], [], [], []
        train_p, train_r, train_f1, dev_p, dev_r, dev_f1 = [], [], [], [], [], []
        model.train()
        for input_ids, token_type_ids, attention_mask, labels, adjs, cls_ids, node_ids, kg_adjs, disease_ids in tqdm(train_dataloader):
            input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
            attention_mask, labels = attention_mask.to(device), labels.to(device)
            disease_ids = disease_ids.to(device)
            outputs = model(input_ids, token_type_ids, attention_mask, adjs, cls_ids, node_ids, kg_adjs, disease_ids)
            result = logits_to_multi_hot(outputs)
            optimizer.zero_grad()
            loss = loss_func(outputs, labels.float())
            batch_labels = labels.data.cpu().numpy()
            p, r, f1, j = get_metrics(result, batch_labels)
            train_p.append(p), train_r.append(r), train_f1.append(f1), train_jaccard.append(j)
            train_loss.append(loss.item())

            idx_train += 1
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.log('train_loss: {}'.format(np.mean(train_loss)))
        logger.log('train_prescription: {}'.format(np.mean(train_p)))
        logger.log('train_recall: {}'.format(np.mean(train_r)))
        logger.log('train_f1: {}'.format(np.mean(train_f1)))
        logger.log('train_jaccard_score: {}'.format(np.mean(train_jaccard)))
        if dev_dataloader:
            with torch.no_grad():
                model.eval()
                for input_ids, token_type_ids, attention_mask, labels, adjs, cls_ids, node_ids, kg_adjs, disease_ids in tqdm(dev_dataloader):
                    input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
                    attention_mask, labels = attention_mask.to(device), labels.to(device)
                    disease_ids = disease_ids.to(device)
                    outputs = model(input_ids, token_type_ids, attention_mask, adjs, cls_ids, node_ids, kg_adjs, disease_ids)
                    result = logits_to_multi_hot(outputs)
                    loss = loss_func(outputs, labels.float())
                    labels = labels.data.cpu().numpy()
                    p, r, f1, j = get_metrics(result, labels)
                    dev_p.append(p), dev_r.append(r), dev_f1.append(f1), dev_jaccard.append(j)
                    dev_loss.append(loss.item())
                    idx_dev += 1
                dev_j = np.mean(dev_jaccard)
                logger.log('dev_loss: {}'.format(np.mean(dev_loss)))
                logger.log('dev_prescription: {}'.format(np.mean(dev_p)))
                logger.log('dev_recall: {}'.format(np.mean(dev_r)))
                logger.log('dev_f1: {}'.format(np.mean(dev_f1)))
                logger.log('dev_jaccard_score: {}'.format(dev_j))
                if max_dev_jaccard < dev_j:
                    max_dev_jaccard = dev_j
                    best_model = model
    if test_dataloader:
        with torch.no_grad():
            best_model.eval()
            for input_ids, token_type_ids, attention_mask, labels, adjs, cls_ids, node_ids, kg_adjs, disease_ids in tqdm(test_dataloader):
                input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
                attention_mask, labels = attention_mask.to(device), labels.to(device)
                disease_ids = disease_ids.to(device)
                outputs = best_model(input_ids, token_type_ids, attention_mask, adjs, cls_ids, node_ids, kg_adjs, disease_ids)
                result = logits_to_multi_hot(outputs)
                loss = loss_func(outputs, labels.float())
                labels = labels.data.cpu().numpy()
                p, r, f1, j = get_metrics(result, labels)
                test_p.append(p), test_r.append(r), test_f1.append(f1), test_jaccard.append(j)
                test_loss.append(loss.item())
            test_j = np.mean(test_jaccard)
            logger.log('test_loss: {}'.format(np.mean(test_loss)))
            logger.log('test_prescription: {}'.format(np.mean(test_p)))
            logger.log('test_recall: {}'.format(np.mean(test_r)))
            logger.log('test_f1: {}'.format(np.mean(test_f1)))
            logger.log('test_jaccard_score: {}'.format(test_j))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", '-cu', type=str, default=0)
    parser.add_argument('--seed', '-s', type=int, default=20)
    parser.add_argument('--lr', '-l', type=float, default=2e-5)
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--hidden_size', '-hs', type=int, default=256)
    parser.add_argument('--output_size', '-os', type=int, default=512)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--log_name', '-lg', type=str, default='log-')
    parser.add_argument('--trained_model', '-tm', type=str, default='')
    parser.add_argument('--data_path', '-p', type=str, help='dataset path')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'

    from utils import get_metrics, logits_to_multi_hot, setup_seed, MyLogger, load_model
    import os
    import time
    from model import DDN
    import numpy as np
    from data import KGInfo, PretrainedDataset, LabelData
    import sys
    from torch.nn import DataParallel
    from torch.utils.data import DataLoader, Dataset, random_split
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    from tqdm import tqdm
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    milestones = [20, 30, 40]
    gama = 0.5

    train()
