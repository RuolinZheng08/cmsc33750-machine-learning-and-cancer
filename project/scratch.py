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

cvae = ConditionalConvVAE(100, N_IN_CHANNELS, N_CLASSES).cuda()
opt = torch.optim.Adam(vae.parameters(), lr=5e-4)

