import torch
import os
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

from eval import evaluate
from train import parse
import utils
import random

if __name__ == '__main__':
    utils.seed_torch(3)
    parser = parse()
    parser.add_argument('--extra', type=str, default='_macro')
    args = parser.parse_args()

    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location='cpu')
    batch_size = args.batch
    data_path = args.data
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    args = parser.parse_args(namespace=args)
    print(args)
    data_path = os.path.join('data', args.data if 'data' in args else data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)

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
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])

        from model.single_prompt import HTCModel
    model = HTCModel.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                     use_graph=args.graph, data_path=data_path, label_mask=[],
                                     depth2label=depth2label, loss=args.loss, double_inf=args.adv)
    model.init_embedding()

    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'.format(extra)),
                            map_location='cpu')

    model.load_state_dict(checkpoint['param'])
    model.to('cuda')

    test = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    model.eval()
    pred = []
    gold = []
    father_count = 0
    father_false = 0
    # gold_total = 0
    with torch.no_grad(), tqdm(test) as pbar:
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
    with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
        print('macro', macro_f1, 'micro', micro_f1, file=f)
        print(scores['full'], file=f)

    with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
        print('macro', macro_f1, 'micro', micro_f1, file=f)
        print('all', scores['full'][2], file=f)
        depth_macro = [[] for _ in range(max_depth + 1)]
        depth_r = [[] for _ in range(max_depth + 1)]
        depth_p = [[] for _ in range(max_depth + 1)]
        depth_g = [[] for _ in range(max_depth + 1)]
        for i in scores['full'][2]:
            v = scores['full'][2][i]
            i = int(i.split('_')[-1])
            depth_r[depth_dict[i]].append(scores['full'][3][i])
            depth_p[depth_dict[i]].append(scores['full'][4][i])
            depth_g[depth_dict[i]].append(scores['full'][5][i])
            depth_macro[depth_dict[i]].append(v)
            if depth_dict[i] > 3:
                depth_r[-1].append(scores['full'][3][i])
                depth_p[-1].append(scores['full'][4][i])
                depth_g[-1].append(scores['full'][5][i])
                depth_macro[-1].append(v)
        for d in range(max_depth + 1):
            p = sum(depth_r[d]) / sum(depth_p[d])
            r = sum(depth_r[d]) / sum(depth_g[d])
            micro_f1 = 2 * p * r / (p + r)
            print(d, sum(depth_macro[d]) / len(depth_macro[d]), micro_f1, file=f)

        from collections import defaultdict

        count = defaultdict(int)
        dataset = datasets.load_from_disk(os.path.join(data_path, 'bert'))
        for i in dataset['train']:
            for l in i['label']:
                count[l] += 1
        print(count)
        count_ = []
        for i in scores['full'][2]:
            v = scores['full'][2][i]
            i = int(i.split('_')[-1])
            count_.append((v, count[i]))
        sorted(count_, key=lambda x: x[1])
        l = len(count_)
        print(count_, file=f)
        print(sum([i[0] for i in count_[:l // 5]]) / len(count_[:l // 5]), file=f)
        print(sum([i[0] for i in count_[l // 5:l * 2 // 5]]) / len(count_[l // 5:l * 2 // 5]), file=f)
        print(sum([i[0] for i in count_[l * 2 // 5:l * 3 // 5]]) / len(count_[l * 2 // 5:l * 3 // 5]), file=f)
        print(sum([i[0] for i in count_[l * 3 // 5:l * 4 // 5]]) / len(count_[l * 3 // 5:l * 4 // 5]), file=f)
        print(sum([i[0] for i in count_[l * 4 // 5:]]) / len(count_[l * 4 // 5:]), file=f)
