from utils import *

def main():
    if len(sys.argv) < 3:
        print('python train.py [cvae | cgan | classifier_baseline | classifier_cvae | classifier_cgan] [data_dir]')
        return

    EXPERIMENT = sys.argv[1]
    if 'classifier' in EXPERIMENT:
        DATADIR = sys.argv[2]
    else:
        DATADIR = 'tumor_data'

    print(EXPERIMENT, DATADIR)

    # load data
    transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor()
    ])
    dataset = TumorDataset(os.path.join(INDIR, 'tumor_train_labels.csv'),
    os.path.join(INDIR, DATADIR), transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    writer = SummaryWriter(os.path.join(OUTDIR, EXPERIMENT))

    if 'classifier' in EXPERIMENT:
        print('Train classifier')
        DEV_SIZE = len(dataset) // 5
        train, dev = random_split(dataset, [len(dataset) - DEV_SIZE, DEV_SIZE],
        generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        dev_loader = DataLoader(dataset=dev, batch_size=len(dev), shuffle=False)

        model = create_classifier(N_IN_CHANNELS)
        criterion = nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.5, 0.999))

        best_auc = 0
        for epoch in tqdm(range(N_EPOCHS)):
            dev_auc = train_classifier(epoch, model, opt, criterion, train_loader, dev_loader, writer)
            # tqdm.set_description('Dev AUC {:f}'.format(dev_auc))
            if dev_auc > best_auc: # save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': opt.state_dict()
                    },
                os.path.join(OUTDIR, EXPERIMENT, 'best_model.pth'))
                best_auc = dev_auc

    else:
        if EXPERIMENT == 'cvae':
            model = ConditionalConvVAE(N_LATENT, N_IN_CHANNELS, N_CLASSES).cuda()
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)

            best_loss = 0
            for epoch in tqdm(range(N_EPOCHS)):
                curr_loss = train_cvae(epoch, model, opt, dataloader, writer)
                # tqdm.set_description('Loss {:f}'.format(curr_loss))
                if curr_loss < best_loss:
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'opt_state_dict': opt.state_dict()
                            },
                        os.path.join(OUTDIR, EXPERIMENT, 'best_model.pth'))
                    best_loss = curr_loss

        elif EXPERIMENT == 'cgan':
            generator = ConditionalConvGenerator(N_LATENT, N_IN_CHANNELS, N_CLASSES, IMG_SIZE).cuda()
            gopt = torch.optim.Adam(generator.parameters(), lr=5e-4, betas=(0.5, 0.999))
            discriminator = ConditionalConvDiscriminator(N_LATENT, N_IN_CHANNELS, N_CLASSES, IMG_SIZE).cuda()
            dopt = torch.optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999))
            criterion = torch.nn.BCEWithLogitsLoss()

            best_loss = 0
            for epoch in tqdm(range(N_EPOCHS)):
                curr_loss = train_cgan(epoch, generator, discriminator, gopt, dopt, criterion, dataloader, writer)
                # tqdm.set_description('Generator loss {:f}'.format(curr_loss))
                if curr_loss < best_loss:
                    torch.save({
                            'epoch': epoch,
                            'generator_state_dict': generator.state_dict(),
                            'discriminator_state_dict': discriminator.state_dict(),
                            'gopt_state_dict': gopt.state_dict(),
                            'dopt_state_dict': dopt.state_dict()
                            },
                        os.path.join(OUTDIR, EXPERIMENT, 'best_model.pth'))
                    best_loss = curr_loss

        else:
            print('Invalid model name')

    writer.close()

if __name__ == '__main__':
    main()
