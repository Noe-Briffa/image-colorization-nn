from torchvision import datasets, transforms
import os

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

save_dir = "./cifar10_test_images"
os.makedirs(save_dir, exist_ok=True)

for i, (img, label) in enumerate(train_dataset):
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(f"{save_dir}/test_{i:05d}.png")

print("Images sauvegardées dans", save_dir)
