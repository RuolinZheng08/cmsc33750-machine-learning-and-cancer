DIR = '../../input/'
N_IN_CHANNELS = 3 # RGB
N_CLASSES = 2 # binary classification
BATCH_SIZE = 32
IMG_SIZE = 96
CROP_SIZE = 64

transform = transforms.Compose([
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
])

dataset = TumorDataset(DIR + 'tumor_train_labels.csv',
                      DIR + 'tumor_data/', transform=transform)

dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

DEV_SIZE = len(dataset) // 5

train, dev = random_split(dataset, [len(dataset) - DEV_SIZE, DEV_SIZE],
generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev, batch_size=len(dev), shuffle=False, num_workers=4)

writer = SummaryWriter(os.path.join(OUTDIR, EXPERIMENT))

cvae = ConditionalConvVAE(100, N_IN_CHANNELS, N_CLASSES).cuda()
opt = torch.optim.Adam(vae.parameters(), lr=5e-4)

generator = ConditionalConvGenerator(100, N_IN_CHANNELS, N_CLASSES, IMG_SIZE).cuda()
gopt = torch.optim.Adam(generator.parameters(), lr=5e-4, betas=(0.5, 0.999))
discriminator = ConditionalConvDiscriminator(100, N_IN_CHANNELS, N_CLASSES, IMG_SIZE).cuda()
dopt = torch.optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999))
criterion = torch.nn.BCEWithLogitsLoss()

model = create_classifier(N_IN_CHANNELS)
criterion = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.5, 0.999))

best_auc = 0
for epoch in range(10):
    dev_auc = train_classifier(epoch, model, opt, criterion, train_loader, dev_loader, writer)
    if dev_auc > best_auc: # save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()
            },
        os.path.join(OUTDIR, EXPERIMENT, 'best_model.pth'))
        best_auc = dev_auc

test_dataset = TumorDataset(DIR + 'tumor_test_labels.csv',
                      DIR + 'tumor_data/', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                         shuffle=False, num_workers=4)

