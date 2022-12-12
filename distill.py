import torch
from torchvision.models import resnet50

from vit_pytorch.distill import DistillableViT, DistillWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher = resnet50(pretrained = True)

# teacher = teacher.to(device)

v = DistillableViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# v = v.to(device)

distiller = DistillWrapper(
    student = v,
    teacher = teacher,
    temperature = 3,           # temperature of distillation
    alpha = 0.5,               # trade between main loss and distillation loss
    hard = False               # whether to use soft or hard distillation
)

optimizer = torch.optim.AdamW(v.parameters(),lr=1e-4)

# class DistillRandom(Dataset):
#     def __init__(self, img_size, img_number):
#         self.img_size = img_size
#         self.img_number = img_number

#     def __len__(self):
#         return img_number

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

num_epochs = 100000
running_loss = 0.0
train_total = 0

for epoch in range(0, num_epochs):
    
    img = torch.randn(2, 3, 256, 256)
    labels = torch.randint(0, 1000, (2,))

    loss = distiller(img, labels)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    train_total += 1
    if epoch%100==0:
        print(f'Loss: {(running_loss/train_total):.4f}')
        running_loss = 0.0
        train_total = 0


# pred = v(img) # (2, 1000)