from torch import sum

def cal_consistency(top1_preds_ids, label_ids):
    bz, n_rel = top1_preds.shape
    consist = torch.sum(top1_preds[:, 0] == top1_preds[:, 1:], 1) / (n_rel-1) # bz

    n_all_pairs = n_rel*(n_rel - 1)/2
    all_consist = torch.sum(torch.sum(top1_preds.unsqueeze(2) == top1_preds.unsqueeze(1), dim=-1)-1, dim=-1)/n_all_pairs # bz

    correct_pairs = torch.sum(top1_preds == label_ids.reshape(-1,1), dim=-1)
    acc_consist = correct_pairs * (correct_pairs-1) / 2 
    
    return consist, all_consist, acc_consist