import io, json, os, zipfile
from PIL import Image
from torchvision.datasets import MNIST

def main(dest='datasets/mnist-32x32.zip', root='./mnist_raw'):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ds = MNIST(root=root, train=True, download=True)  # 60k training images
    labels = []
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_STORED) as z:
        for i, (img, y) in enumerate(ds):
            canvas = Image.new('L', (32, 32), 0)       # black 32x32 canvas
            canvas.paste(img.convert('L'), (2, 2))     # center the 28x28 digit
            buf = io.BytesIO()
            canvas.convert('RGB').save(buf, format='PNG')  # 3 channels
            name = f'{i // 1000:05d}/img{i:08d}.png'
            z.writestr(name, buf.getvalue())
            labels.append([name, int(y)])
        z.writestr('dataset.json', json.dumps({'labels': labels}))
    print(f'Wrote {len(labels)} images to {dest}')

if __name__ == '__main__':
    main()
