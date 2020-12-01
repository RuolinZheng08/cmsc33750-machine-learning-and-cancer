from utils import *

def main():
    if len(sys.argv) < 2:
        print('python generate.py [cvae | cgan]')
        return

    EXPERIMENT = sys.argv[1]
    SAVEDIR = EXPERIMENT + '_data' # save to [cvae|cgan]_data
    SAVELABEL = EXPERIMENT + '_labels.csv' # [cvae|cgan]_train_labels.csv

    # load best model
    best_cache = torch.load(os.path.join(OUTDIR, EXPERIMENT, 'best_model.pth'))
    model = None
    if EXPERIMENT.startswith('cvae'):
        model = ConditionalConvVAE(N_LATENT, N_IN_CHANNELS, N_CLASSES).cuda()
        model.load_state_dict(best_cache['model_state_dict'])
    elif EXPERIMENT.startswith('cgan'): # load generator
        model = ConditionalConvGenerator(N_LATENT, N_IN_CHANNELS, N_CLASSES).cuda()
        model.load_state_dict(best_cache['generator_state_dict'])
    else:
        print('Invalid model name:', EXPERIMENT)
        return

    try:
        os.mkdir(os.path.join(INDIR, SAVEDIR))
    except:
        pass

    # load original data to determine the number of positive and negative samples
    dataset = TumorDataset(os.path.join(INDIR, 'tumor_train_labels.csv'), os.path.join(INDIR, 'tumor_data'))
    counts = dataset.annotations.label.value_counts()
    num_zeros, num_ones = counts[0], counts[1]
    labels = torch.cat([torch.zeros(num_zeros, dtype=torch.long), torch.ones(num_ones, dtype=torch.long)], dim=0)
    ids = np.arange(len(dataset))

    # generate [cvae|cgan]_train_labels.csv
    df = pd.DataFrame({'id': ids, 'label': labels})
    df.to_csv(os.path.join(INDIR, SAVELABEL), index=False)

    # generate data and save to [cvae|cgan]_data
    batch_size = 1000
    num_batches = len(dataset) //  batch_size
    print('Num batches: ', num_batches)
    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            start = batch_idx * batch_size
            end = start + batch_size
            if EXPERIMENT.startswith('cvae'): # convert labels to one-hot
                batch_labels = F.one_hot(labels[start : end], N_CLASSES).cuda()
                data = model.generate(batch_size, batch_labels)
            else:
                data = model(batch_size, labels[start : end].cuda())
            for idx, arr in zip(ids[start : end], data):
                img = transforms.ToPILImage()(arr.cpu())
                img.save(os.path.join(INDIR, SAVEDIR, '{}.tif'.format(idx)))

if __name__ == '__main__':
    main()
