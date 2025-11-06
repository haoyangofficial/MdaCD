# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)



    # --- Params & FLOPs from the first real test batch (safe, no device side effects) ---
    '''import torch
    from torch import nn
    from mmengine.hooks import Hook
    try:
        from mmengine.model import is_model_wrapper
    except Exception:
        is_model_wrapper = None

    def _unwrap(m):
        if is_model_wrapper is not None:
            try:
                return m.module if is_model_wrapper(m) else m
            except Exception:
                pass
        return getattr(m, 'module', m)

    def _fmt(n):
        if n is None: return "N/A"
        units = ['F','KF','MF','GF','TF','PF']
        i = 0
        while n >= 1000 and i < len(units)-1:
            n /= 1000.0; i += 1
        return f"{n:.3f}{units[i]}"

    def _first_tensor(x):
        if torch.is_tensor(x): return x
        if isinstance(x, (list, tuple)):
            for xi in x:
                t = _first_tensor(xi)
                if t is not None: return t
        if isinstance(x, dict):
            for v in x.values():
                t = _first_tensor(v)
                if t is not None: return t
        return None

    class ComplexityHook(Hook):
        """Accumulate FLOPs from actual first test pass; print once."""
        priority = 'ABOVE_NORMAL'
        def __init__(self):
            self._armed = False
            self._printed = False
            self._handles = []
            self._flops = 0
            self._input_shape = None
            self._param_count = None

        # register hooks just before iter 0 runs
        def before_test_iter(self, runner, batch_idx, data_batch):
            if self._armed or batch_idx != 0:
                return

            # only rank 0 prints
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                    return
            except Exception:
                pass

            core = _unwrap(runner.model).eval()

            # record learnable parameters now
            self._param_count = sum(p.numel() for p in core.parameters() if p.requires_grad)

            # capture the real input shape we are about to use
            x = data_batch.get('inputs', None)
            if isinstance(x, (list, tuple)): x = x[0]
            if torch.is_tensor(x):
                if x.dim() == 3:  # CHW -> NCHW
                    x = x.unsqueeze(0)
                self._input_shape = tuple(x.shape)

            # ---- define robust hooks (ignore weird outputs) ----
            def conv2d_hook(m: nn.Conv2d, inp, out):
                try:
                    x0 = _first_tensor(inp)
                    y0 = _first_tensor(out)
                    if x0 is None or y0 is None: return
                    N, Cin, Hin, Win = x0.shape
                    N, Cout, Hout, Wout = y0.shape
                    kH, kW = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size))
                    groups = m.groups if m.groups else 1
                    macs = N * Cout * Hout * Wout * (Cin // groups) * kH * kW
                    bias = (N * Cout * Hout * Wout) if m.bias is not None else 0
                    self._flops += 2 * macs + bias
                except Exception:
                    pass  # never break the run

            def convt_hook(m: nn.ConvTranspose2d, inp, out):
                try:
                    x0 = _first_tensor(inp); y0 = _first_tensor(out)
                    if x0 is None or y0 is None: return
                    N, Cin, Hin, Win = x0.shape
                    N, Cout, Hout, Wout = y0.shape
                    kH, kW = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size))
                    groups = m.groups if m.groups else 1
                    macs = N * Cin * Hout * Wout * (Cout // groups) * kH * kW
                    bias = (N * Cout * Hout * Wout) if m.bias is not None else 0
                    self._flops += 2 * macs + bias
                except Exception:
                    pass

            def linear_hook(m: nn.Linear, inp, out):
                try:
                    x0 = _first_tensor(inp); y0 = _first_tensor(out)
                    if x0 is None or y0 is None: return
                    batch = int(x0.numel() / x0.shape[-1])
                    macs = batch * m.in_features * m.out_features
                    bias = batch * m.out_features if m.bias is not None else 0
                    self._flops += 2 * macs + bias
                except Exception:
                    pass

            def upsample_hook(m: nn.Upsample, inp, out):
                try:
                    y0 = _first_tensor(out)
                    if y0 is None: return
                    # nearest/bilinear approx: 1 op per output element
                    self._flops += int(y0.numel())
                except Exception:
                    pass
            # -----------------------------------------------

            # attach to heavy ops only
            for mod in core.modules():
                if isinstance(mod, nn.Conv2d):
                    self._handles.append(mod.register_forward_hook(conv2d_hook))
                elif isinstance(mod, nn.ConvTranspose2d):
                    self._handles.append(mod.register_forward_hook(convt_hook))
                elif isinstance(mod, nn.Linear):
                    self._handles.append(mod.register_forward_hook(linear_hook))
                elif isinstance(mod, nn.Upsample):
                    self._handles.append(mod.register_forward_hook(upsample_hook))

            self._armed = True  # hooks are live; the upcoming normal forward will trigger them

        # after iter 0 finishes its NORMAL forward, print & clean up
        def after_test_iter(self, runner, batch_idx, data_batch, outputs=None):
            if self._printed or batch_idx != 0:
                return

            for h in self._handles:
                try: h.remove()
                except Exception: pass
            self._handles.clear()

            # only rank 0 prints
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                    return
            except Exception:
                pass

            print("\n" + "="*60)
            print(f"Model complexity (first test batch input shape: {self._input_shape})")
            print("- Learnable parameters:", f"{(self._param_count or 0):,}")
            print("- FLOPs (Conv/Linear [+upsample approx], 2 ops per MAC):", _fmt(self._flops))
            print("  (Norm/activations/softmax and tiny ops are ignored.)")
            print("="*60 + "\n")

            self._printed = True

    runner.register_hook(ComplexityHook(), priority='ABOVE_NORMAL')'''
    # --- end block ---




    # start testing
    runner.test()


if __name__ == '__main__':
    main()
