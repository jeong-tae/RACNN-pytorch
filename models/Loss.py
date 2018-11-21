import torch
import torch.nn.functional as F

def multitask_loss(preds, labels):
    loss = []
    for i in range(len(preds)):
        loss.append(F.cross_entropy(preds[i], labels))
    #loss = torch.sum(torch.stack(loss))
    return loss

def pairwise_ranking_loss(preds, size_average = True):
    """
        preds:
            list of scalar Tensor.
            Each value represent the probablity of each class
                e.g) class = 3
                    preds = [logits1[class], logits2[class]]
    """
    if len(preds) <= 1:
        return torch.zeros(1).cuda()
    else:
        losses = []
        for pred in preds:
            loss = []
            for i in range(len(pred)-1):
                rank_loss = (pred[i]-pred[i+1] + 0.05).clamp(min = 0)
                # 0.05 margin
                loss.append(rank_loss)
            loss = torch.sum(torch.stack(loss))
            losses.append(loss)
        losses = torch.stack(losses)
        if size_average:
            losses = torch.mean(losses)
        else:
            losses = torch.sum(losses)
        return losses

if __name__ == '__main__':
    print(" [*] Loss test...")
    # assume that batch_size = 2
    logits = [torch.randn(2, 10), torch.randn(2, 10), torch.randn(2, 10)] 
    target_cls = torch.LongTensor([3, 2])
    preds = []
    for i in range(len(target_cls)):
        pred = [logit[i][target_cls[i]] for logit in logits]
        preds.append(pred)
    loss_cls = multitask_loss(logits, target_cls)
    loss_rank = pairwise_ranking_loss(preds)
    print(" [*] Loss_cls:", loss_cls)
    print(" [*] Loss_rank:", loss_rank)
