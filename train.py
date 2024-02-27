from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import os
import datasets
from tqdm import tqdm
import argparse
import wandb

from eval import evaluate

import utils


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--data', type=str, default='WebOfScience')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--early-stop', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, default='auto')
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--arch', type=str, default='bert-base-uncased')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', default=True, action='store_false')
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--loss', default='ZMLCE', type=str)
    parser.add_argument('--adv', default=False, action='store_true')
    parser.add_argument('--adv-weight', type=float, default=1)
    return parser


class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    if args.name == 'auto':
        args.name = '-'.join(['auto', args.model, 'adv' if args.adv else 'base', str(args.seed), str(args.adv_weight)])
    if args.wandb:
        wandb.init(config=args, project='new-htc')
    print(args)
    utils.seed_torch(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}
    # label_dict.update({0: '<s>', 2: '</s>', 1: 'PAD'})

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    num_class = 0
    children = set()
    path_list = []
    value2slot = {}
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            children.add(v)
            if num_class < v:
                num_class = v
    path_list = [(i, v) for v, i in value2slot.items()]
    num_class += 1
    depth2label = None

    if 'bert' in args.model:
        if os.path.exists(os.path.join(data_path, 'bert')):
            dataset = datasets.load_from_disk(os.path.join(data_path, 'bert'))
        else:
            dataset = datasets.load_dataset('json',
                                            data_files={'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                        'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                        'test': 'data/{}/{}_test.json'.format(args.data, args.data), })


            def data_map_function(batch, tokenizer):
                new_batch = {}
                new_batch.update(tokenizer(batch['token'], truncation=True, padding='max_length'))
                new_batch['labels'] = []
                for b in batch['label']:
                    new_batch['labels'].append([0 for _ in range(num_class)])
                    for i in b:
                        new_batch['labels'][-1][i] = 1
                return new_batch


            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(os.path.join(data_path, 'bert'))

        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])

        if args.model == 'bert':
            from model.bert import HTCModel
        elif args.model == 'hibert':
            from model.bert_new import HTCModel
        else:
            raise NotImplementedError
    elif 'prompt' in args.model:
        for i in range(num_class):
            if i not in value2slot:
                value2slot[i] = -1


        def get_depth(x):
            depth = 0
            while value2slot[x] != -1:
                depth += 1
                x = value2slot[x]
            return depth


        depth_dict = {i: get_depth(i) for i in range(num_class)}
        max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
        depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

        if args.model != 'single_prompt':
            for depth in depth2label:
                for l in depth2label[depth]:
                    path_list.append((num_class + depth, l))

            if os.path.exists(os.path.join(data_path, 'prompt')):
                dataset = datasets.load_from_disk(os.path.join(data_path, 'prompt'))
            else:
                dataset = datasets.load_dataset('json',
                                                data_files={
                                                    'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                    'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                    'test': 'data/{}/{}_test.json'.format(args.data, args.data), })

                prefix = []
                for i in range(max_depth):
                    prefix.append(tokenizer.vocab_size + num_class + i)
                    prefix.append(tokenizer.vocab_size + num_class + max_depth)
                prefix.append(tokenizer.sep_token_id)


                def data_map_function(batch, tokenizer):
                    new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                    for l, t in zip(batch['label'], batch['token']):
                        new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)])
                        for d in range(max_depth):
                            for i in depth2label[d]:
                                new_batch['labels'][-1][d][i] = 0
                            for i in l:
                                if new_batch['labels'][-1][d][i] == 0:
                                    new_batch['labels'][-1][d][i] = 1
                        new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y]

                        tokens = tokenizer(t, truncation=True)
                        new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix)
                        new_batch['input_ids'][-1].extend(
                            [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                        new_batch['attention_mask'].append(
                            tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                        new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                        new_batch['token_type_ids'].append([0] * 512)

                    return new_batch


                dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
                dataset.save_to_disk(os.path.join(data_path, 'prompt'))
            from model.prompt import HTCModel
        else:
            for l in range(num_class):
                path_list.append((num_class, l))
            if os.path.exists(os.path.join(data_path, 'single_prompt')):
                dataset = datasets.load_from_disk(os.path.join(data_path, 'single_prompt'))
            else:
                dataset = datasets.load_dataset('json',
                                                data_files={
                                                    'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                    'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                    'test': 'data/{}/{}_test.json'.format(args.data, args.data), })

                prefix = []
                prefix.append(tokenizer.vocab_size + num_class)
                prefix.append(tokenizer.vocab_size + num_class + 1)
                prefix.append(tokenizer.sep_token_id)


                def data_map_function(batch, tokenizer):
                    new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                    for l, t in zip(batch['label'], batch['token']):
                        new_batch['labels'].append([0 for _ in range(num_class)])
                        for i in l:
                            new_batch['labels'][-1][i] = 1

                        tokens = tokenizer(t, truncation=True)
                        new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix)
                        new_batch['input_ids'][-1].extend(
                            [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                        new_batch['attention_mask'].append(
                            tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                        new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                        new_batch['token_type_ids'].append([0] * 512)

                    return new_batch


                dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
                dataset.save_to_disk(os.path.join(data_path, 'single_prompt'))
            from model.single_prompt import HTCModel
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)

    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    update_step = 0
    loss = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))

    test = DataLoader(dataset['test'], batch_size=32, shuffle=False)
    model = HTCModel.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                     use_graph=args.graph, data_path=data_path, label_mask=[],
                                     depth2label=depth2label, loss=args.loss, double_inf=args.adv)
    model.init_embedding()

    model.to('cuda')
    if args.wandb:
        wandb.watch(model)
    if args.adv:
        model.graph_encoder_two.load_state_dict(model.graph_encoder.state_dict(), strict=False)
        if hasattr(model, 'classifier_two'):
            model.classifier_two.load_state_dict(model.classifier.state_dict())
    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.adv:
        discriminator = nn.Sequential(
            nn.Linear(model.config.hidden_size,
                      model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(model.config.hidden_size, 1)
        )
        discriminator.to(args.device)
        optimizer_d = Adam(discriminator.parameters(), lr=args.lr)

    save = Save(model, optimizer, None, args)

    model.to('cuda')

    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    update_step = 0
    loss = 0
    train_gen = True
    gen_step = 0

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break

        model.train()
        with tqdm(train) as p_bar:
            for batch in p_bar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(**batch,
                               )

                if args.adv:
                    if args.model != 'prompt':
                        predict = batch['labels'] + 1
                    else:
                        predict = batch['labels'] * (batch['labels'] >= 0)
                        predict = predict.view(batch['labels'].size(0), len(depth2label), num_class).sum(dim=1)

                    generator_output = model(**batch, predict=predict)

                    # Train discriminator
                    optimizer_d.zero_grad()
                    fake_out = discriminator(output['label_emb'].detach())
                    fake_loss_d = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))

                    label_emb = generator_output['label_emb'].detach()
                    real_out = discriminator(label_emb)
                    real_loss_d = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                    loss_d = real_loss_d + fake_loss_d
                    loss_d.backward()
                    optimizer_d.step()

                    # Train generator
                    optimizer.zero_grad()
                    loss_g = generator_output['loss']

                    # Train model
                    loss = output['loss'].item()
                    if epoch != 0:
                        model_adv_out = discriminator(output['label_emb'])
                        model_adv_loss = F.binary_cross_entropy_with_logits(model_adv_out,
                                                                            torch.ones_like(model_adv_out))
                        total_loss = output['loss'] + model_adv_loss * args.adv_weight + loss_g
                        model_adv_loss = model_adv_loss.item()
                    else:
                        model_adv_loss = 0
                        total_loss = output['loss'] + loss_g

                    total_loss.backward()
                    optimizer.step()
                    optimizer_d.zero_grad()

                    if args.wandb:
                        wandb.log({'loss': loss,
                                   'loss_d': loss_d.item(), 'loss_g': loss_g.item(),
                                   'adv_loss': model_adv_loss})
                    p_bar.set_description(
                        'loss:{:.4f} loss_d:{:.4f} loss_g:{:.4f} adv_loss:{:.4f}'.format(loss,
                                                                                         loss_d.item(), loss_g.item(),
                                                                                         model_adv_loss))
                else:
                    optimizer.zero_grad()
                    output['loss'].backward()
                    optimizer.step()
                    loss = output['loss'].item()
                    if args.wandb:
                        wandb.log({'loss': loss, })
                    p_bar.set_description(
                        'loss:{:.4f}'.format(loss, ))

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(input_ids=batch['input_ids'],
                                                    attention_mask=batch['attention_mask'],
                                                    token_type_ids=batch['token_type_ids'], depth2label=depth2label,
                                                    max_length=80,
                                                    )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1})
        early_stop_count += 1

        if epoch != 0:
            if macro_f1 > best_score_macro:
                best_score_macro = macro_f1
                save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
                early_stop_count = 0

            if micro_f1 > best_score_micro:
                best_score_micro = micro_f1
                save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
                early_stop_count = 0
        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_{}.pt'.format(epoch)))
        if args.wandb:
            wandb.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        torch.cuda.empty_cache()

    # test
    model.eval()


    def test_function(extra):
        checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra)),
                                map_location='cpu')
        model.load_state_dict(checkpoint['param'])
        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                output_ids, logits = model.generate(input_ids=batch['input_ids'],
                                                    attention_mask=batch['attention_mask'],
                                                    token_type_ids=batch['token_type_ids'], depth2label=depth2label,
                                                    max_length=80,
                                                    # predict=batch['labels'] + 1
                                                    )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)
            prefix = 'test' + extra
        if args.wandb:
            wandb.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})


    test_function('_macro')
