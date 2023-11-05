import pandas as pd
from matplotlib import pyplot as plt

def get_output_files(test, qrels_out='qrels', run_out='run', threshold=4.0):
    qrels = pd.DataFrame()
    qrels['qid'] = test['user_id'].values
    qrels['iter'] = [0 for _ in range(len(test))]
    qrels['docno'] = test['item_id'].values
    qrels['rel']= [1 if i >= threshold else 0 for i in test['rating'].values]
    # qrels['rel'] = test['rating'].values
    qrels.to_csv(qrels_out, sep=' ', header=False, index=False)

    run = pd.DataFrame()
    run['qid'] = test['user_id'].values
    run['iter'] = [-1 for _ in range(len(test))]
    run['docno'] = test['item_id'].values
    run['rank'] = [-1 for _ in range(len(test))]
    run['sim'] = test['preds'].values
    run['run_id'] = [-1 for _ in range(len(test))]
    run.to_csv(run_out, sep=' ', header=False, index=False)


def get_loss_curve(train_losses, val_losses, outfile='loss_curve.png'):
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MSELoss')

    plt.savefig(outfile)