import torch.nn.functional as F


def multitask_loss(preds, labels):
    loss = []
    for i in range(len(preds)):
        loss.append(F.cross_entropy(preds[i], labels))
    loss = torch.sum(torch.stack(loss))
    return loss

def pairwise_ranking_loss(preds):
    """
        preds:
            list of scalar Tensor.
            Each value represent the probablity of each class
                e.g) class = 3
                    preds = [logits1[class], logits2[class]]
    """
    if len(preds) <= 1:
        return 0
    else:
        loss = []
        for i in range(len(preds)-1):
            rank_loss = (preds[i]-preds[i+1] + 0.05).clamp(min = 0)
            # 0.05 margin
            loss.append(rank_loss)
        loss = torch.sum(torch.stack(loss))
        return loss
