import run_eval
import torch
import argparse
import numpy as np
import utils
from pathlib import Path
from datetime import datetime


def ensemble(target_path, out_dir):
    for model_dir in filter(lambda d:d.is_dir() and d.name != 'pasted', out_dir.iterdir()):
        r = {}
        for replica_dir in filter(lambda d:d.is_dir() and d.name.isdigit(), model_dir.iterdir()):
            for pkl in replica_dir.glob('*.distance'):
                target = pkl.name.split('.')[0]
                dis = np.load(pkl, allow_pickle=True)

                if target in r:
                    r[target].append(dis)
                else:
                    r[target] = [dis]

        ensemble_dir = model_dir / 'ensemble'
        ensemble_dir.mkdir(exist_ok=True)
        for k, v in r.items():
            ensemble_file = ensemble_dir / f'{k}.distance'
            ensemble_dis = sum(v) / len(v)
            ensemble_dis.dump(ensemble_file)

    targets_weight = {data['domain_name']:{'weight':data['num_alignments'][0,0], 'seq':data['sequence']} for data in np.load(target_path, allow_pickle=True)}
    ensemble_dir = out_dir / 'Distogram' / 'ensemble'
    paste_dir = out_dir / 'pasted'
    paste_dir.mkdir(exist_ok=True)
    targets = set([t.split("-")[0] for t in targets_weight.keys()])

    for target in targets:
        combined_cmap = np.load(ensemble_dir / f'{target}.distance', allow_pickle=True)
        counter_map = np.ones_like(combined_cmap[:, :, 0:1])
        seq = targets_weight[target]['seq']
        target_domains = utils.generate_domains(target, seq)

        for domain in sorted(target_domains, key=lambda x: x["name"]):
            if domain["name"] == target: continue

            crop_start, crop_end = domain["description"]
            domain_dis = np.load(ensemble_dir / f'{domain["name"]}.distance', allow_pickle=True)
            weight = targets_weight[domain["name"]]['weight']
            weight_matrix_size = crop_end - crop_start + 1
            weight_matrix = np.ones((weight_matrix_size, weight_matrix_size), dtype=np.float32) * weight
            combined_cmap[crop_start - 1:crop_end, crop_start - 1:crop_end, :] += (domain_dis * np.expand_dims(weight_matrix, 2))
            counter_map[crop_start - 1:crop_end, crop_start - 1:crop_end, 0] += weight_matrix

        combined_cmap /= counter_map
        combined_cmap.dump(paste_dir / f'{target}.distance')
        contact_probs = combined_cmap[:,:,:19].sum(-1)
        utils.save_rr_file(contact_probs, seq, target, paste_dir / f'{target}.rr')
        utils.plot_contact_map(target, [contact_probs, combined_cmap], paste_dir / f'{target}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alphafold - PyTorch version')
    parser.add_argument('-i', '--input', type=str, required=True, help='target protein, support both .pkl or .tfrec format')
    parser.add_argument('-o', '--out', type=str, default='', help='output dir')
    parser.add_argument('-m', '--model', type=str, default='model', help='model dir')
    parser.add_argument('-r', '--replica', type=str, default='0', help='model replica')
    parser.add_argument('-t', '--type', type=str, choices=['D', 'B', 'T'], default='D', help='model type: D - Distogram, B - Background, T - Torsion')
    parser.add_argument('-e', '--ensemble', default=False, action='store_true', help='ensembling all replica outputs')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='debug mode')
    args = parser.parse_args()

    DEBUG = args.debug
    TARGET_PATH = args.input
    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    OUT_DIR = Path(args.out) if args.out else Path(f'contacts_{TARGET}_{timestr}')

    if args.ensemble:
        ensemble(TARGET_PATH, OUT_DIR)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TARGET = TARGET_PATH.split('/')[-1].split('.')[0]
        REPLICA = args.replica
        if args.type == 'D':
            MODEL_TYPE = 'Distogram'
            MODEL_PATH = Path(args.model) / '873731'
        elif args.type == 'B':
            MODEL_TYPE = 'Background'
            MODEL_PATH = Path(args.model) / '916425'
        elif args.type == 'T':
            MODEL_TYPE = 'Torsion'
            MODEL_PATH = Path(args.model) / '941521'
        OUT_DIR = OUT_DIR / MODEL_TYPE / REPLICA
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f'Input file: {TARGET_PATH}')
        print(f'Output dir: {OUT_DIR}')
        print(f'{MODEL_TYPE} model: {MODEL_PATH}')
        print(f'Replica: {REPLICA}')
        print(f'Device: {DEVICE}')

        run_eval(TARGET_PATH, MODEL_PATH, REPLICA, OUT_DIR, DEVICE)