import argparse
import itertools
import json
import logging
import pathlib
import sys

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import tqdm

import models

from transformers import AutoTokenizer
from ard_dataset import ARDDataset
import orjson


def get_parser(
    parser=argparse.ArgumentParser(
        description="Run a reverse dictionary baseline.\nThe task consists in reconstructing an embedding from the glosses listed in the datasets"
    ),
):
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="whether to eval a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--from_pretrained", action="store_true", help="whether to load pretrained weights"
    )
    parser.add_argument(
        "--model_name", type=str, help="HF model name"
    )
    parser.add_argument(
        "--resume_train",
         type=pathlib.Path,  
         help="where the model & vocab is saved",
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="path to the train file",
    )
    parser.add_argument(
        "--target_arch",
        type=str,
        default="electra",
        choices=("sgns", "electra", "bertseg", "bertmsa"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        choices=(300, 256, 768),
        help="dimension of embedding",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / f"revdict-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / f"revdict-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        default=pathlib.Path("revdict-baseline-preds.json"),
        help="where to save predictions",
    )   
    return parser

def rank_cosine(preds, targets):
    assocs = F.normalize(preds) @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float()
    assert ranks.numel() == preds.size(0)
    ranks = ranks.mean().item()
    return ranks / preds.size(0)

def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    ## make datasets
    train_dataset = ARDDataset(args.train_file)
    valid_dataset = ARDDataset(args.dev_file)
    
    ## make dataloader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    ## make summary writer
    summary_writer = SummaryWriter(args.save_dir / args.summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    ## Hyperparams
    logger.debug("Setting up training environment")

    model = models.ARBERTRevDict(args).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)     
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 30, 1.0e-4, 0.9, 0.999, 1.0e-6
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    vec_tensor_key = f"{args.target_arch}_tensor"

    best_cosine = 0

    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        for ids, word, gloss, electra, bertseg, bertmsa in train_dataloader:
            optimizer.zero_grad()

            word_tokens = tokenizer(word, padding=True, return_tensors='pt').to(args.device)
            gloss_tokens = tokenizer(gloss, padding=True, return_tensors='pt').to(args.device)

            if args.target_arch == "electra":
                target_embs = torch.stack(electra, dim=1).to(args.device)
            elif args.target_arch =="bertseg":
                target_embs = torch.stack(bertseg, dim=1).to(args.device)
            elif args.target_arch =="bertmsa":
                target_embs = torch.stack(bertmsa, dim=1).to(args.device)

            # print(gloss_tokens)

            target_embs = target_embs.float()
            pred = model(**gloss_tokens)
            loss = criterion(pred, target_embs)
            loss.backward()
            # keep track of the train loss for this step
            next_step = next(train_step)
            summary_writer.add_scalar(
                "revdict-train/cos",
                F.cosine_similarity(pred, target_embs).mean().item(),
                next_step,
            )
            summary_writer.add_scalar("revdict-train/mse", loss.item(), next_step)
            optimizer.step()
            pbar.update(target_embs.size(0))

        pbar.close()
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss, sum_cosine, sum_rnk = 0.0, 0.0, 0.0
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch}",
                    total=len(valid_dataset),
                    disable=None,
                    leave=False,
                )
                for ids, word, gloss, electra, bertseg, bertmsa in valid_dataloader:
                    word_tokens = tokenizer(word, padding=True, return_tensors='pt').to(args.device)
                    gloss_tokens = tokenizer(gloss, max_length=512, padding=True, truncation=True, return_tensors='pt').to(args.device)
                    if args.target_arch == "electra":
                        target_embs = torch.stack(electra, dim=1).to(args.device)
                    elif args.target_arch == "bertseg":
                        target_embs = torch.stack(bertseg, dim=1).to(args.device)
                    elif args.target_arch == "bertmsa":
                        target_embs = torch.stack(bertmsa, dim=1).to(args.device)
                    # else:
                    #     target_embs = torch.stack(sgns, dim=1).to(args.device)

                    target_embs = target_embs.float()
                    pred = model(**gloss_tokens)
                    sum_dev_loss += (
                        F.mse_loss(pred, target_embs, reduction="none").mean(1).sum().item()
                    )
                    sum_cosine += F.cosine_similarity(pred, target_embs).sum().item()
                    sum_rnk += rank_cosine(pred, target_embs)
                    pbar.update(target_embs.size(0))

                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch} cos: "+str(sum_cosine / len(valid_dataset))+" mse: "+str( sum_dev_loss / len(valid_dataset) )+" rnk: "+str(sum_rnk/ len(valid_dataset))+ " sum_rnk: "+str(sum_rnk)+" len of dev: "+str(len(valid_dataset)) +"\n",
                    total=len(valid_dataset),
                    disable=None,
                    leave=False,
                )

                if sum_cosine >= best_cosine:
                    best_cosine = sum_cosine
                    print(f"Saving Best Checkpoint at Epoch {epoch}")
                    model.save(args.save_dir)
                    

                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar(
                    "revdict-dev/cos", sum_cosine / len(valid_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse", sum_dev_loss / len(valid_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/rnk", sum_rnk / len(valid_dataset), epoch
                )
                pbar.close()
                model.train()

        # model.save(args.save_dir / "modelepoch.pt")
            
    # 5. save result
    # model.save(args.save_dir / "model.pt")


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
        ## make datasets
    test_dataset = ARDDataset(args.test_file, is_test=True)
    
    ## make dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    ## Hyperparams
    logger.debug("Setting up training environment")

    model = models.ARBERTRevDict(args).load(f"{args.save_dir}")
    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)     

    vec_tensor_key = f"{args.target_arch}_tensor"

    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset))
        for ids, words, gloss in test_dataloader:
            word_tokens = tokenizer(words, padding=True, return_tensors='pt').to(args.device)
            gloss_tokens = tokenizer(gloss, max_length=512, padding=True, truncation=True, return_tensors='pt').to(args.device)
            # print(gloss_tokens)

            vecs = model(**gloss_tokens)
            # Extract the last hidden states
            for id, word, vec in zip(ids, words, vecs.unbind()):
                predictions.append(
                    {"id": id, "word": word, args.target_arch: vec.view(-1).tolist()}
                )
            pbar.update(vecs.size(0))
        pbar.close()

    logger.debug("writing predction file") 
    # with open(args.save_dir /args.pred_file, "w") as ostr:
    #     json.dump( predictions, ostr)
    with open(args.save_dir / args.pred_file, "wb") as ostr:  # Note the "wb" mode for orjson
        ostr.write(orjson.dumps(predictions))
    logger.debug("writing finished") 





def main(args):
    torch.autograd.set_detect_anomaly(True)
    if args.do_train:
        logger.debug("Performing revdict training")
        train(args)
    if args.do_pred:
        logger.debug("Performing revdict prediction")
        pred(args)
        logger.debug("Prediction finished")

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
