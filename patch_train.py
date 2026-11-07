src = open('train.py').read()
marker = "    c.G_kwargs.NoiseDimension = NoiseDimension"
assert marker in src, "marker line not found - add the preset manually"
assert "MNIST-32" not in src, "already patched"

block = '''    if opts.preset == 'MNIST-32':
        WidthPerStage = [256, 256, 256, 256]
        BlocksPerStage = [1, 1, 1, 1]
        CardinalityPerStage = [16, 16, 16, 16]
        FP16Stages = [-1, -2, -3]
        NoiseDimension = 64

        if opts.cond:
            c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
            c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]

        ema_nimg = 500 * 1000
        decay_nimg = 5e6

        c.ema_scheduler = { 'base_value': 0, 'final_value': ema_nimg, 'total_nimg': decay_nimg }
        c.aug_scheduler = { 'base_value': 0, 'final_value': 0.3, 'total_nimg': decay_nimg }
        c.lr_scheduler = { 'base_value': 2e-4, 'final_value': 5e-5, 'total_nimg': decay_nimg }
        c.gamma_scheduler = { 'base_value': 0.05, 'final_value': 0.005, 'total_nimg': decay_nimg }
        c.beta2_scheduler = { 'base_value': 0.9, 'final_value': 0.99, 'total_nimg': decay_nimg }

'''
open('train.py', 'w').write(src.replace(marker, block + marker, 1))
print("train.py patched")
