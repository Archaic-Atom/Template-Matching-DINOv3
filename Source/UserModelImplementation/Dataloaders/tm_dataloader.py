# -*- coding: utf-8 -*-
"""Dataloader handler for Template-Matching-DINOv3.

In test mode each rank writes a per-rank CSV with one row per sample
(``index, sar_name, rgb_name, gt_x, gt_y, pred_x, pred_y, score``). After
the loop, rank 0 merges all per-rank CSVs into a single
``test_<dataset>_<timestamp>.csv`` (sorted by sample index) and deletes the
temp files. The decoding of model output uses ``decode_primary`` from the
Models side so test results stay aligned with the training-time
``accuracy()`` a0 metric.

测试模式下每个 rank 各自写一个 per-rank CSV，包含
``index, sar_name, rgb_name, gt_x, gt_y, pred_x, pred_y, score``。
循环结束后由 rank 0 合并为一个 ``test_<dataset>_<timestamp>.csv``
（按 index 排序）并删除临时文件。模型输出的解码统一调用
``decode_primary``，保证测试导出与训练 ``accuracy()`` 的 a0 完全一致。
"""
import csv
import glob
import os
import time

import torch
import JackFramework as jf  # type: ignore

try:
    from ._tm_datasets import TM_Dataset
except ImportError:
    from _tm_datasets import TM_Dataset

try:
    from ._tm_webdataset import build_tm_webdataset
except ImportError:
    try:
        from _tm_webdataset import build_tm_webdataset
    except ImportError:
        build_tm_webdataset = None


# Module-level handle to the active test-mode dataloader so the model's
# ``postprocess()`` hook can trigger merge/cleanup without an explicit
# reference. Set in ``get_train_dataset(is_training=False)``, cleared by
# ``finalize_active_test()``.
# 模块级句柄，让 model.postprocess 能在不持有 dataloader 引用的情况下
# 触发合并/清理。
_ACTIVE_TEST_LOADER = None


CSV_HEADER = (
    'index', 'sar_name', 'rgb_name',
    'gt_x', 'gt_y', 'pred_x', 'pred_y', 'score',
)


def _sanitize(name: str) -> str:
    """Make a string filesystem-safe (no /, \\, spaces).

    把字符串变成文件名安全的形式。
    """
    safe = str(name)
    for ch in ('/', '\\', ' ', '\t', '\n'):
        safe = safe.replace(ch, '_')
    return safe or 'unknown'


def finalize_active_test() -> None:
    """Trigger merge/cleanup on the currently active test dataloader.

    Called from the model's ``postprocess()`` hook so we don't have to
    add a new method to JackFramework's DataHandlerTemplate. Idempotent;
    safe to call when no test is active.

    由 model.postprocess 调用，避免改动 JackFramework 的抽象基类。
    无 test 在跑时是 no-op，可重复调用。
    """
    global _ACTIVE_TEST_LOADER
    loader = _ACTIVE_TEST_LOADER
    _ACTIVE_TEST_LOADER = None
    if loader is None:
        return
    loader._finalize_test()


class TMDataloader(jf.UserTemplate.DataHandlerTemplate):
    """Template-matching dataloader. Stage 2 test mode writes CSVs."""

    ID_SAR, ID_RGB, ID_LABEL = 0, 1, 2
    ID_SAR_NAME, ID_RGB_NAME = 3, 4
    ID_MODEL, ID_PRED = 0, 0
    # supplement layout in test mode: [sar_names, rgb_names, labels]
    ID_SPLIT_SAR_NAME, ID_SPLIT_RGB_NAME, ID_SPLIT_LABEL = 0, 1, 2

    def __init__(self, args: object) -> None:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset, self.__val_dataset = None, None

        # Test-mode state. Initialised in ``get_train_dataset`` when
        # ``is_training=False``.
        # 测试态字段；只在 is_training=False 的 get_train_dataset 中初始化。
        self._is_test_mode = False
        self._test_rank = 0
        self._test_tmp_path = None
        self._test_tmp_fh = None
        self._test_tmp_writer = None

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        if self.__use_webdataset():
            self.__train_dataset = build_tm_webdataset(
                args.trainListPath, is_train=is_training,
                is_pre_train=args.pre_trained,
                is_stage_2=bool(getattr(args, 'is_stage_2', False)),
                distributed=bool(getattr(args, 'dist', False)),
                shuffle_buffer=int(getattr(args, 'shuffle_buffer', 100000)),
                batch_size=int(getattr(args, 'batchSize', 1)),
                num_workers=int(getattr(args, 'dataloaderNum', 1)))
        else:
            self.__train_dataset = TM_Dataset(
                args.trainListPath, is_training, args.pre_trained,
                is_stage_2=bool(getattr(args, 'is_stage_2', False)))

        if not is_training:
            # ``--mode test`` reuses get_train_dataset for the test set.
            # ``--mode test`` 复用 train 路径加载测试集。
            self._open_test_writer()

        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        args = self.__args
        if self.__use_webdataset():
            self.__val_dataset = build_tm_webdataset(
                args.valListPath, is_train=False,
                is_pre_train=args.pre_trained,
                is_stage_2=bool(getattr(args, 'is_stage_2', False)),
                distributed=bool(getattr(args, 'dist', False)),
                shuffle_buffer=int(getattr(args, 'shuffle_buffer', 100000)),
                batch_size=int(getattr(args, 'batchSize', 1)),
                num_workers=int(getattr(args, 'dataloaderNum', 1)))
        else:
            self.__val_dataset = TM_Dataset(
                args.valListPath, False, args.pre_trained,
                is_stage_2=bool(getattr(args, 'is_stage_2', False)))
        return self.__val_dataset

    def __use_webdataset(self) -> bool:
        if getattr(self.__args, 'dataloaderType', 'mapstyle') != 'iterable':
            return False
        if build_tm_webdataset is None:
            raise ImportError(
                "dataloaderType=iterable requires the 'webdataset' package "
                "and _tm_webdataset.py to be importable")
        return True

    # ------------------------------------------------------------------
    # Batch splitting — test mode pushes LABEL into supplement
    # ------------------------------------------------------------------

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        if is_training:
            return [batch_data[self.ID_SAR], batch_data[self.ID_RGB]], \
                [batch_data[self.ID_LABEL]]
        # Test/val mode — labels live in supplement so ``save_result``
        # can pair them with predictions for CSV export.
        # 测试/验证模式把 label 放到 supplement，方便 save_result 写入 CSV。
        return [batch_data[self.ID_SAR], batch_data[self.ID_RGB]], \
            [batch_data[self.ID_SAR_NAME], batch_data[self.ID_RGB_NAME],
             batch_data[self.ID_LABEL]]

    # ------------------------------------------------------------------
    # Result display (unchanged from prior version)
    # ------------------------------------------------------------------

    def show_train_result(self, epoch: int, loss: list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)
        info_str = self.__result_str.training_result_str(
            epoch, loss[self.ID_MODEL], acc[self.ID_MODEL], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss: list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)
        info_str = self.__result_str.training_result_str(
            epoch, loss[self.ID_MODEL], acc[self.ID_MODEL], duration, False)
        jf.log.info(info_str)

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)
        return self.__result_str.training_intermediate_result(
            epoch, loss[self.ID_MODEL], acc[self.ID_MODEL])

    # ------------------------------------------------------------------
    # Per-rank tmp CSV writer
    # ------------------------------------------------------------------

    def _open_test_writer(self) -> None:
        """Open a per-rank tmp CSV with header.

        Each rank gets its own ``.tmp_test_<dataset>_rank{N}.csv``. The
        final merged file (named with a timestamp) is produced in
        ``_finalize_test`` by rank 0.

        每个 rank 开一个 ``.tmp_test_<dataset>_rank{N}.csv``。
        最终带时间戳的合并文件由 rank 0 在 _finalize_test 里生成。
        """
        global _ACTIVE_TEST_LOADER
        args = self.__args
        out_dir = args.outputDir
        os.makedirs(out_dir, exist_ok=True)
        rank = int(os.environ.get('RANK', '0'))
        dataset = _sanitize(getattr(args, 'dataset', 'dataset'))
        tmp_name = f'.tmp_test_{dataset}_rank{rank}.csv'
        tmp_path = os.path.join(out_dir, tmp_name)

        # Overwrite any leftover tmp from a prior aborted run.
        # 覆盖上一次异常退出留下的 tmp。
        fh = open(tmp_path, 'w', newline='')
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)

        self._is_test_mode = True
        self._test_rank = rank
        self._test_tmp_path = tmp_path
        self._test_tmp_fh = fh
        self._test_tmp_writer = writer
        _ACTIVE_TEST_LOADER = self
        jf.log.info(f'[test] rank{rank} writing tmp CSV: {tmp_path}')

    # ------------------------------------------------------------------
    # Per-batch save (called by JackFramework's _save_result)
    # ------------------------------------------------------------------

    def save_result(self, output_data: object, supplement: list,
                    img_id: int, model_id: int) -> None:
        """Write one CSV row per sample from the decoded primary head.

        The model's ``inference()`` attaches ``_test_pred_xy`` and
        ``_test_score`` to the output dict using the same decoder as
        training-time ``accuracy()``, so what we write here is exactly
        what the model would have reported as a0.

        模型的 inference() 已经把主 head 解码结果挂到 output 上了，
        这里只负责按样本逐行写 CSV，确保和训练 a0 同口径。

        Args:
            output_data: Single model's output dict (framework already
                indexed into the per-model list).
            supplement: ``[sar_names, rgb_names, labels]`` (labels is a
                (B, 2) tensor of (gt_x, gt_y)).
            img_id: Framework-computed batch identifier
                = ``rank + iteration * (batchSize * gpu)``.
            model_id: Index in the user-models list (always 0 here).
        """
        if not self._is_test_mode or self._test_tmp_writer is None:
            return
        if model_id != self.ID_MODEL:
            return

        if '_test_pred_xy' not in output_data:
            jf.log.warning(
                '[test] save_result: _test_pred_xy missing in output; '
                'is the model running in stage_2 + test mode? Skipping.')
            return

        sar_names = supplement[self.ID_SPLIT_SAR_NAME]
        rgb_names = supplement[self.ID_SPLIT_RGB_NAME]
        labels = supplement[self.ID_SPLIT_LABEL]

        pred_xy = output_data['_test_pred_xy'].detach().cpu()
        score = output_data['_test_score'].detach().cpu()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu()
        else:
            labels = torch.as_tensor(labels)

        batch = pred_xy.shape[0]
        for b in range(batch):
            # Per-sample globally-unique index: img_id is unique per
            # (rank, iteration), so (img_id * batch + b) is unique across
            # ranks and iterations.
            # 每个样本的全局唯一 index。
            sample_idx = int(img_id) * batch + b
            row = (
                sample_idx,
                sar_names[b], rgb_names[b],
                int(labels[b, 0].item()), int(labels[b, 1].item()),
                f'{float(pred_xy[b, 0].item()):.2f}',
                f'{float(pred_xy[b, 1].item()):.2f}',
                f'{float(score[b].item()):.2f}',
            )
            self._test_tmp_writer.writerow(row)

    # ------------------------------------------------------------------
    # Test finalize: rank-0 merge + sort + cleanup
    # ------------------------------------------------------------------

    def _finalize_test(self) -> None:
        """Close tmp writers, barrier all ranks, then rank 0 merges.

        关闭 tmp 写入流；所有 rank barrier 同步；rank 0 合并所有
        rank 的 tmp 文件，按 sample index 排序，写最终 csv，删除 tmp。
        """
        # Close own tmp file first so contents are flushed to disk.
        # 先关闭自己的 tmp 文件，确保数据 flush。
        if self._test_tmp_fh is not None:
            try:
                self._test_tmp_fh.close()
            except Exception:
                pass
            self._test_tmp_fh = None
            self._test_tmp_writer = None

        # Barrier so rank 0 doesn't start merging before others finish writing.
        # 同步等所有 rank 写完。
        try:
            import torch.distributed as dist
            if getattr(self.__args, 'dist', False) and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception as exc:
                    jf.log.warning(f'[test] finalize barrier failed: {exc}')
        except Exception:
            pass

        if self._test_rank != 0:
            self._is_test_mode = False
            return

        # Rank 0: merge all per-rank tmp files.
        args = self.__args
        out_dir = args.outputDir
        dataset = _sanitize(getattr(args, 'dataset', 'dataset'))
        tmp_glob = os.path.join(out_dir, f'.tmp_test_{dataset}_rank*.csv')
        tmp_files = sorted(glob.glob(tmp_glob))
        if not tmp_files:
            jf.log.warning(
                f'[test] finalize: no tmp files matched {tmp_glob}; '
                'nothing to merge.')
            self._is_test_mode = False
            return

        timestamp = time.strftime('%Y%m%d-%H%M%S')
        final_name = f'test_{dataset}_{timestamp}.csv'
        final_path = os.path.join(out_dir, final_name)

        rows = []
        for tf in tmp_files:
            try:
                with open(tf, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for r in reader:
                        if not r:
                            continue
                        rows.append(r)
            except Exception as exc:
                jf.log.warning(f'[test] failed reading {tf}: {exc}')

        # Sort by integer index column (column 0).
        # 按 index 列（第 0 列）数值升序排序。
        try:
            rows.sort(key=lambda r: int(r[0]))
        except (ValueError, IndexError) as exc:
            jf.log.warning(
                f'[test] sort by index failed ({exc}); writing unsorted.')

        # Re-index 0..N-1 (sequential, contiguous) after sort.
        # 排序后重编号为连续的 0..N-1。
        for new_idx, r in enumerate(rows):
            r[0] = str(new_idx)

        try:
            with open(final_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                writer.writerows(rows)
            jf.log.info(
                f'[test] merged {len(rows)} rows from {len(tmp_files)} '
                f'rank tmp file(s) → {final_path}')

            # Only delete tmps if merge succeeded.
            # 合并成功才删 tmp。
            for tf in tmp_files:
                try:
                    os.remove(tf)
                except OSError as exc:
                    jf.log.warning(f'[test] failed removing {tf}: {exc}')
        except Exception as exc:
            jf.log.error(
                f'[test] failed writing final CSV {final_path}: {exc}. '
                f'Tmp files left in place: {tmp_files}')

        self._is_test_mode = False
