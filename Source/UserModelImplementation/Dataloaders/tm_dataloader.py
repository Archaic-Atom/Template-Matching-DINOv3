# -*- coding: utf-8 -*-
import os
import time
import torch
import JackFramework as jf  # type: ignore
# import UserModelImplementation.user_define as user_def
try:
    from ._tm_datasets import TM_Dataset
except ImportError:
    from _tm_datasets import TM_Dataset


class TMDataloader(jf.UserTemplate.DataHandlerTemplate):
    ID_SAR, ID_RGB, ID_LABEL = 0, 1, 2
    ID_SAR_NAME, ID_RGB_NAME = 3, 4
    ID_MODEL, ID_PRED = 0, 0
    ID_SPLIT_SAR_NAME, ID_SPLIT_RGB_NAME = 0, 1
    ID_X, ID_Y = 1, 2

    def __init__(self, args: object) -> None:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset, self.__val_dataset = None, None
        self.__imgs_num = 0
        self.__test_result = None

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        self.__train_dataset = TM_Dataset(
            args.trainListPath, is_training, args.pre_trained)
        if not is_training:
            t = time.strftime("%Y%m%d-%H%M%S")
            with open(os.path.join(args.outputDir, t + '.txt'), 'w') as f:
                self.__test_result = f
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        args = self.__args
        self.__val_dataset = TM_Dataset(args.valListPath, False)
        return self.__val_dataset

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        if is_training:
            return [batch_data[self.ID_SAR], batch_data[self.ID_RGB]],\
                [batch_data[self.ID_LABEL]]
        return [batch_data[self.ID_SAR], batch_data[self.ID_RGB], batch_data[self.ID_LABEL]], \
            [batch_data[self.ID_SAR_NAME], batch_data[self.ID_RGB_NAME]]

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(
            epoch, loss[self.ID_MODEL], acc[self.ID_MODEL], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(
            epoch, loss[self.ID_MODEL], acc[self.ID_MODEL], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        pred = output_data[self.ID_MODEL][self.ID_PRED]
        batch_size, _, _ = pred.shape
        sar_name, rgb_name = supplement[self.ID_SPLIT_SAR_NAME], supplement[self.ID_SPLIT_RGB_NAME]
        pred_y, pred_x = self._argmax(pred, self.ID_Y), self._argmax(pred, self.ID_X)
        for i in range(batch_size):
            if img_id == self.__imgs_num:
                self.__test_result.write(
                    sar_name[i] + ' ' + rgb_name[i] + ' '
                    + str(pred_y[i].item()) + ' ' + str(pred_x[i].item()) + '\n')
        self.__imgs_num += 1

    @staticmethod
    def _argmax(x, ax):
        y = torch.max(x, ax)[0]
        x = torch.argmax(y, dim=1).float()
        return x

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        return self.__result_str.training_intermediate_result(
            epoch, loss[self.ID_MODEL], acc[self.ID_MODEL])
