from utils import *

def main():
    if len(sys.argv) < 2:
        print('python eval.py [classifier_baseline | classifier_cvae | classifier_cgan]')
        return

    EXPERIMENT = sys.argv[1]
    writer = SummaryWriter(os.path.join(OUTDIR, EXPERIMENT))

    # load data
    transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor()
    ])
    test_dataset = TumorDataset(os.path.join(INDIR, 'tumor_test_labels.csv'),
                    os.path.join(INDIR, 'tumor_data/'), transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = create_classifier(N_IN_CHANNELS)
    best_cache = torch.load(os.path.join(OUTDIR, EXPERIMENT, 'best_model.pth'))
    model.load_state_dict(best_cache['model_state_dict'])
    criterion = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        # a single batch
        for data, labels in test_loader:
            x = data.cuda()
            y = labels.cuda()
            preds = model(x).squeeze().cpu()
            loss = criterion(preds, y.float())
            auc = roc_auc_score(labels, preds)
    print('test AUC', auc)
    writer.add_scalar('AUC/test', auc)

    # confusion matrix
    cfmat = confusion_matrix(labels, np.where(preds > 0.5, 1, 0))
    fig = sns.heatmap(cfmat, annot=True, cmap='Blues')
    fig2 = sns.heatmap(cfmat / np.sum(cfmat), annot=True, fmt='.2%', cmap='Blues')
    writer.add_figure('ConfusionMatrix/num', fig)
    writer.add_figure('ConfusionMatrix/percent', fig2)
    np.save(os.path.join(OUTDIR, EXPERIMENT, 'test_preds.npy'), preds)

if __name__ == '__main__':
    main()
