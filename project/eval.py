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
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset) // 2, shuffle=False)

    model = create_classifier(N_IN_CHANNELS)
    best_cache = torch.load(os.path.join(OUTDIR, EXPERIMENT, 'best_model.pth'))
    model.load_state_dict(best_cache['model_state_dict'])
    criterion = nn.BCELoss()
    model.eval()

    preds_list = []
    total_loss = 0
    with torch.no_grad():
        # two batches
        for data, labels in test_loader:
            x = data.cuda()
            y = labels.cuda()
            preds = model(x).squeeze()
            loss = criterion(preds, y.float())
            preds_list.append(preds.cpu().numpy())
            total_loss += loss.item()

    preds = np.hstack(preds_list)
    labels = test_dataset.annotations.label
    auc = roc_auc_score(labels, preds)
    writer.add_scalar('loss/test', total_loss)
    writer.add_scalar('AUC/test', auc)

    preds_class = np.where(preds > 0.5, 1, 0) # threshold 0.5
    accu = accuracy_score(labels, preds_class)
    print('test loss {:f}, AUC {:f}, accuracy: {:f}'.format(total_loss, auc, accu))
    writer.add_scalar('accuracy/test', accu)

    # confusion matrix
    cfmat = confusion_matrix(labels, preds_class)
    fig = plt.figure()
    sns.heatmap(cfmat, annot=True, fmt='d', cmap='Blues')
    writer.add_figure('ConfusionMatrix/num', fig)
    fig2 = plt.figure()
    sns.heatmap(cfmat / np.sum(cfmat), annot=True, fmt='.2%', cmap='Blues')
    writer.add_figure('ConfusionMatrix/percent', fig2)
    np.save(os.path.join(OUTDIR, EXPERIMENT, 'test_preds.npy'), preds)

if __name__ == '__main__':
    main()
