import os
import logging
from datetime import datetime
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class Exp_Conf(object):
    def __init__(self, PY_FILE, SUPER_PARAM, sys_argv, dataSet='OSINT'):
        self.SUPER_PARAM = SUPER_PARAM
        self.sys_argv = sys_argv
        self.LOAD_WEIGHTS_FLAG = False
        if len(self.sys_argv) >= 2 and 'load_weights=1' in self.sys_argv[1]:
            self.LOAD_WEIGHTS_FLAG = True
        self.TIME_PREFIX = ''
        if len(self.sys_argv) >= 2 and 'time_prefix=1' in self.sys_argv[1]:
            self.TIME_PREFIX = datetime.now().strftime("%Y%m%d%H%M") + '_'
        self.PY_FILE = PY_FILE
        self.FILE_PREFIX = self.TIME_PREFIX + self.PY_FILE + '_' + self.SUPER_PARAM
        self.TRAIN_LOG_FILE = 'LOG/' + self.FILE_PREFIX + '_training.log.txt'
        self.MODEL_FILE = os.path.join('MODEL', self.FILE_PREFIX + '_model_weights.h5')
        self.FMODEL_FILE = os.path.join('FMODEL', self.FILE_PREFIX + '_fmodel_weights.h5')
        self.CONFUSION_MATRIX_FILE = 'LOG/' + self.FILE_PREFIX + '_Confusion_Matrix.png'
        self.CONFUSION_MATRIX_LOG_FILE = 'LOG/' + self.FILE_PREFIX + '_Confusion_Matrix.txt'

        # 标签配置
        self._init_label_orders(dataSet)

        # 配置日志记录
        logging.basicConfig(
            filename=self.TRAIN_LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _init_label_orders(self, dataSet):
        """初始化标签顺序配置"""
        self.label_order = [
            'w32.virut', 'cryptowall', 'bedep', 'hesperbot', 'tempedre', 'beebone',
            'volatile', 'bamital', 'proslikefan', 'corebot', 'fobber', 'padcrypt', 'ramdo',
            'matsnu', 'nymaim', 'geodo', 'dircrypt', 'cryptolocker', 'pushdo', 'locky',
            'dyre', 'pykspa', 'qadars', 'suppobox', 'shifu', 'symmi', 'kraken',
            'shiotob-urlzone-bebloh', 'qakbot', 'ranbyus', 'simda', 'murofet', 'tinba',
            'necurs', 'ramnit', 'post', 'banjori'
        ]
        if dataSet == '360':
            self.label_order = [
                'tofsee', 'gspy', 'proslikefan', 'vidro', 'bamital', 'padcrypt',
                'pykspa_v2_real', 'tempedreve', 'vawtrak', 'fobber_v1', 'fobber_v2',
                'nymaim', 'dircrypt', 'conficker', 'pykspa_v2_fake', 'matsnu', 'chinad',
                'dyre', 'cryptolocker', 'locky', 'qadars', 'suppobox', 'shifu', 'symmi',
                'ranbyus', 'murofet', 'necurs', 'virut', 'gameover', 'ramnit', 'simda',
                'pykspa_v1', 'tinba', 'rovnix', 'emotet', 'banjori'
            ]

    def log(self, info):
        logging.info(info)

    def check_load_weights(self, fmodel_flag=False):
        target_file = self.FMODEL_FILE if fmodel_flag else self.MODEL_FILE
        return self.LOAD_WEIGHTS_FLAG and os.path.exists(target_file)

    def load_weights(self, model, fmodel_flag=False):
        target_file = self.FMODEL_FILE if fmodel_flag else self.MODEL_FILE
        model.load_weights(target_file)
        self.log(f"Loaded weights from {target_file}")

    def save_weights(self, model, fmodel_flag=False):
        target_file = self.FMODEL_FILE if fmodel_flag else self.MODEL_FILE
        os.makedirs(os.path.dirname(target_file), exist_ok=True)  # 修复：补全右括号
        model.save_weights(target_file)
        self.log(f"Saved weights to {target_file}")

    def log_train(self, history):
        metrics = history.history.keys()
        for epoch, epoch_data in enumerate(history.epoch, 1):
            log_str = f"Epoch {epoch}/{len(history.epoch)}: " + ", ".join(
                [f"{k}={v[epoch - 1]:.4f}" for k, v in history.history.items()]
            )
            logging.info(log_str)
        logging.info("Training completed")

    def log_evaluate(self, loss, accuracy):
        logging.info(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def log_test(self, y_true_classes, y_pred_classes, target_names):
        try:
            # 检查输入类型
            if len(y_true_classes) == 0 or len(y_pred_classes) == 0:
                raise ValueError("输入标签不能为空")

            # 转换输入为numpy数组
            y_true_classes = np.asarray(y_true_classes)
            y_pred_classes = np.asarray(y_pred_classes)

            # 处理字符串标签
            if y_true_classes.dtype.kind in ['U', 'O']:  # 字符串类型
                y_true_classes = np.array([np.where(target_names == x)[0][0] for x in y_true_classes])
            if y_pred_classes.dtype.kind in ['U', 'O']:  # 字符串类型
                y_pred_classes = np.array([np.where(target_names == x)[0][0] for x in y_pred_classes])

            # 转换为整数
            y_true = y_true_classes.astype(int)
            y_pred = y_pred_classes.astype(int)

            # 获取实际存在的标签
            present_labels = np.unique(np.concatenate([y_true, y_pred]))
            valid_target_names = [target_names[i] for i in present_labels]

            # 生成分类报告
            report = classification_report(
                y_true, y_pred,
                target_names=valid_target_names,
                labels=present_labels,
                digits=4
            )
            logging.info("\nClassification Report:\n" + report)

            # 生成混淆矩阵
            self._generate_confusion_matrix(y_true, y_pred, valid_target_names)

        except Exception as e:
            logging.error(f"评估失败: {str(e)}\n"
                          f"y_true_classes 类型: {y_true_classes.dtype if len(y_true_classes) > 0 else 'empty'}\n"
                          f"y_true_classes 示例: {y_true_classes[:5] if len(y_true_classes) > 0 else 'empty'}\n"
                          f"y_pred_classes 类型: {y_pred_classes.dtype if len(y_pred_classes) > 0 else 'empty'}\n"
                          f"y_pred_classes 示例: {y_pred_classes[:5] if len(y_pred_classes) > 0 else 'empty'}")
            raise

    def _generate_confusion_matrix(self, y_true, y_pred, target_names):
        """生成并保存混淆矩阵"""
        try:
            # 创建标签映射
            label_mapping = {label: idx for idx, label in enumerate(target_names)}
            valid_labels = [label for label in self.label_order if label in label_mapping]
            valid_indices = [label_mapping[label] for label in valid_labels]

            # 生成并处理混淆矩阵
            conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(target_names)))
            conf_matrix_reordered = conf_matrix[np.ix_(valid_indices, valid_indices)]
            conf_matrix_norm = conf_matrix_reordered.astype('float') / conf_matrix_reordered.sum(axis=1)[:, np.newaxis]

            # 保存数据
            np.savetxt(
                self.CONFUSION_MATRIX_LOG_FILE,
                conf_matrix_norm,
                delimiter='\t',
                fmt='%.4f'
            )

            # 绘制热力图
            plt.figure(figsize=(16, 14))
            sns.heatmap(
                conf_matrix_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=valid_labels,
                yticklabels=valid_labels
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Normalized Confusion Matrix')
            plt.tight_layout()
            plt.savefig(self.CONFUSION_MATRIX_FILE)
            plt.close()

        except Exception as e:
            logging.error(f"生成混淆矩阵错误: {str(e)}")
            raise