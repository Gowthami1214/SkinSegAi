import torch, sys
sys.path.insert(0, '.')
from src.model import load_checkpoint

device = torch.device('cpu')
files = [
    'results/mobilenetv3_unet_50_best.pth',
    'results/mobilenetv3_unet_100_best.pth',
    'results/mobilenetv3_unet_250_best.pth',
]

for fname in files:
    try:
        raw = torch.load(fname, map_location='cpu', weights_only=False)
        epoch = raw.get('epoch')
        bvd   = raw.get('best_val_dice')
        bvi   = raw.get('best_val_iou')
        print(fname)
        print(f'  epoch={epoch}  best_val_dice={bvd:.4f}  best_val_iou={bvi:.4f}')

        result = load_checkpoint(fname, device)
        model  = result['model']
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out  = model(x)
            prob = torch.sigmoid(out)
            mask = (prob > 0.5).float()
        pct = mask.mean().item() * 100
        print(f'  output={tuple(out.shape)}  lesion={pct:.1f}%  PASS')
    except Exception as e:
        print(f'  FAIL: {e}')
    print()
