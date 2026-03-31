import pandas as pd
import numpy as np
from keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import os
import sys
from tensorflow.python.keras.regularizers import l2
from Exp_Conf import *
from sklearn.decomposition import PCA
from keras import initializers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
import tensorflow as tf
import keras as K
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as KK


SUPER_PARAM = '-'
LOSS_WEIGHT = 0.9
if len(sys.argv) >= 2:
    LOSS_WEIGHT = float(sys.argv[1])

EPOCH = 5
BATCH_SIZE = 32

PY_FILE = os.path.basename(__file__).replace(".py", "")
ExpConf = Exp_Conf(PY_FILE, SUPER_PARAM, sys.argv, dataSet='360')

ExpConf.log("LOSS_WEIGHT="+str(LOSS_WEIGHT)+", EPOCH="+str(EPOCH)+"\r\n")

# 读取数据
df = pd.read_csv('./processed_data_360.csv')
# df = pd.read_csv('/home/pycharm_project_279/dataset/processed_data.csv')

# 读取域名数据
all_chars = set(''.join(df['Domain']))
char_to_index = {char: idx for idx, char in enumerate(sorted(all_chars))}
char_to_index['_'] = 38  # 填充字符
max_length = max(len(domain) for domain in df['Domain'])



    
class AttentionWeightedAverage2(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage2, self).__init__(**kwargs)
 
    def build(self, input_shape):
        # 检查输入维度
        assert len(input_shape) == 3
        # 添加权重，形状为 (input_dim, 1)
        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='attention_weights',
                                 initializer=self.init,
                                 trainable=True)
        # 设置输入规范
        self.input_spec = [InputSpec(ndim=3)]
        super(AttentionWeightedAverage2, self).build(input_shape)
 
    def call(self, x, mask=None):
        # 计算logits，形状为 (batch_size, seq_len, 1) -> (batch_size, seq_len)
        logits = tf.squeeze(tf.matmul(x, self.W), axis=-1)
        
        # 应用softmax得到注意力权重
        ai = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
        if mask is not None:
            ai *= tf.cast(mask, tf.float32)  # 应用mask
        att_weights = ai / (tf.reduce_sum(ai, axis=-1, keepdims=True) + tf.keras.backend.epsilon())
        
        # 计算加权输入和结果
        weighted_input = x * tf.expand_dims(att_weights, axis=-1)
        result = tf.reduce_sum(weighted_input, axis=1)
        
        # 根据需要返回结果和注意力权重
        if self.return_attention:
            return [result, att_weights]
        return result

        
def domain_to_indices(domain):
    indices = [char_to_index.get(char, 38) for char in domain]
    return indices + [38] * (max_length - len(indices))

df['DomainIndices'] = df['Domain'].apply(domain_to_indices)
domain_indices_array = np.array(df['DomainIndices'].tolist()).astype(int)

#读取新增tld&len的token
# 将com_tld_len字段的字符串从逗号分隔，转换为列表
df['com_tld_len'] = df['com_tld_len'].apply(lambda x: x.split(','))

# 读取token数据
all_tokens = set(token for sublist in df['com_tld_len'] for token in sublist)
token_to_index = {token: idx for idx, token in enumerate(sorted(all_tokens))}
def tokens_to_indices(tokens):
    indices = [token_to_index.get(token, len(token_to_index)) for token in tokens]
    return indices

df['TokenIndices'] = df['com_tld_len'].apply(tokens_to_indices)
token_indices_array = np.array(df['TokenIndices'].tolist()).astype(int)
token_indices_dim = token_indices_array.shape[1]
token_len=6

#读取2gram
all_2grams = set(gram for sublist in df['2gram'].apply(lambda x: x.split(',')) for gram in sublist)
two_gram_to_index = {gram: idx for idx, gram in enumerate(sorted(all_2grams))}
two_gram_to_index['--'] = len(two_gram_to_index)
def parse_string_list(s):
      return s.split(',')
# 计算每个字符串列表的长度，并找出最大值
max_2gram_length = max(len(parse_string_list(row)) for row in df['2gram'])
def two_grams_to_indices(grams):
    grams_list = grams.split(',')
    indices = [two_gram_to_index.get(gram, len(two_gram_to_index)) for gram in grams_list]
    # 填充到统一长度
    return indices + [len(two_gram_to_index)] * (max_2gram_length - len(indices))

df['TwoGramIndices'] = df['2gram'].apply(two_grams_to_indices)
two_gram_indices_array = np.array(df['TwoGramIndices'].tolist()).astype(int)

#读取3gram
all_3grams = set(gram for sublist in df['3gram'].apply(lambda x: x.split(',')) for gram in sublist)
three_gram_to_index = {gram: idx for idx, gram in enumerate(sorted(all_3grams))}
three_gram_to_index['---'] = len(three_gram_to_index)
max_3gram_length = max(len(parse_string_list(row)) for row in df['3gram'])
def three_grams_to_indices(grams):
    grams_list = grams.split(',')
    indices = [three_gram_to_index.get(gram, len(three_gram_to_index)) for gram in grams_list]
    # 填充到统一长度
    return indices +  [len(three_gram_to_index)] * (max_3gram_length - len(indices))

df['ThreeGramIndices'] = df['3gram'].apply(three_grams_to_indices)
three_gram_indices_array = np.array(df['ThreeGramIndices'].tolist()).astype(int)

# 读取TLD数据并进行独热编码 sparse=False
tld_encoder = OneHotEncoder(sparse=False)
tld_encoded = tld_encoder.fit_transform(df[['TLD']].values.reshape(-1, 1))
tld_input_dim = tld_encoded.shape[1]

#读取word token
all_wt = set(wt for sublist in df['Word Token'].apply(lambda x: x.split(',')) for wt in sublist)
wt_to_index = {wt: idx for idx, wt in enumerate(sorted(all_wt))}
wt_to_index['-'] = len(wt_to_index)
max_wt_length = max(len(parse_string_list(row)) for row in df['Word Token'])
def wt_to_indices(wt):
    wt_list = wt.split(',')
    indices = [wt_to_index.get(wt, len(wt_to_index)) for wt in wt_list]
    # 填充到统一长度
    return indices +  [len(wt_to_index)] * (max_wt_length - len(indices))

df['WtIndices'] = df['Word Token'].apply(wt_to_indices)
wt_indices_array = np.array(df['WtIndices'].tolist()).astype(int)

# 读取TLD数据并进行独热编码 sparse=False
tld_encoder = OneHotEncoder(sparse=False)
tld_encoded = tld_encoder.fit_transform(df[['TLD']].values.reshape(-1, 1))
tld_input_dim = tld_encoded.shape[1]

# 读取Feature数据
feature_array = df['Feature'].apply(lambda x: np.array(eval(x))).tolist()
feature_array = np.array(feature_array)
feature_array_dim = feature_array.shape[1]
# 对标签进行独热编码
label_binarizer = LabelBinarizer()
labels_one_hot = label_binarizer.fit_transform(df['Label'])

combined_data=np.hstack((domain_indices_array, tld_encoded, feature_array,token_indices_array,two_gram_indices_array,three_gram_indices_array,wt_indices_array))
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    combined_data, labels_one_hot, test_size=0.2, random_state=42)
# 对数据集进行过采样
sampling_strategy ={0: 205 , 1: 452007 , 2: 12800 , 3: 462 , 4: 340 , 5: 15796 , 6: 20595 , 7: 20160 , 8: 1245 , 9: 20805 , 10: 6136 , 11: 20645 , 12: 8522 , 13: 12194 , 14: 1396 , 15: 28082 , 16: 35379 , 17: 16613 , 18: 1066 , 19: 62425 , 20: 18523 , 21: 3350 , 22: 6682 , 23: 3444 , 24: 21072 , 25: 1564 , 26: 57565 , 27: 23924 , 28: 2922 , 29: 9212 , 30: 23619 , 31: 2852 , 32: 4360 , 33: 20660 , 34: 31734 , 35: 1504 , 36: 20555}
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
#X_train, y_train = ros.fit_resample(X_train, y_train)

# 重新分割合并后的数据
X_train_domain_end = max_length
X_train_tld_end = X_train_domain_end + tld_input_dim
X_train_feature_end = X_train_tld_end + feature_array_dim
X_train_token_end = X_train_feature_end + token_indices_dim
X_train_two_gram_end = X_train_token_end + max_2gram_length # 2gram的长度
X_train_three_gram_end =X_train_two_gram_end+max_3gram_length
X_train_wt_end=X_train_three_gram_end+max_wt_length

X_train_domain = X_train[:, :X_train_domain_end]
X_train_tld = X_train[:, X_train_domain_end:X_train_tld_end]
X_train_feature = X_train[:, X_train_tld_end:X_train_feature_end]
X_train_token_indices = X_train[:, X_train_feature_end:X_train_token_end]
X_train_two_gram_indices = X_train[:, X_train_token_end:X_train_two_gram_end]
X_train_three_gram_indices = X_train[:, X_train_two_gram_end:X_train_three_gram_end]
X_train_wt_indices = X_train[:, X_train_three_gram_end:X_train_wt_end]

X_test_domain = X_test[:, :X_train_domain_end]
X_test_tld = X_test[:, X_train_domain_end:X_train_tld_end]
X_test_feature = X_test[:, X_train_tld_end:X_train_feature_end]
X_test_token_indices = X_test[:, X_train_feature_end:X_train_token_end]
X_test_two_gram_indices = X_test[:, X_train_token_end:X_train_two_gram_end]
X_test_three_gram_indices =X_test[:, X_train_two_gram_end:X_train_three_gram_end]
X_test_wt_indices = X_test[:, X_train_three_gram_end:X_train_wt_end]

# TLD的Dense层处理
tld_input = Input(shape=(tld_input_dim,))
tld_dense = Dense(64, activation='relu')(tld_input)
# tld_dense=AttentionWeightedAverage2()(tf.expand_dims(tld_dense, axis=1))
#tld_dense = Dense(64, activation='relu')(tld_dense)

# Feature的Dense层处理
feature_input = Input(shape=(feature_array.shape[1],))
feature_dense = Dense(64, activation='relu')(feature_input)
# feature_dense=AttentionWeightedAverage2()(tf.expand_dims(feature_dense, axis=1))
#feature_dense = Dense(64, activation='relu')(feature_dense)

# Domain,新增tld&len,2gram,3gram,word token特征的Embedding和LSTM处理
embedding_dim = 32
domain_input = Input(shape=(max_length,))
domain_embedding = Embedding(input_dim=len(char_to_index) + 1, output_dim=50, input_length=max_length)(domain_input)
domain_lstm = LSTM(100, return_sequences=False)(domain_embedding)
#domain_lstm=AttentionWeightedAverage2()(tf.expand_dims(domain_lstm, axis=1))
#domain_lstm = Dense(64, activation='relu')(domain_lstm)


token_input = Input(shape=(token_len,))
token_embedding =Embedding(input_dim=len(token_to_index) + 1, output_dim=embedding_dim, input_length=token_len)(token_input)
token_lstm = LSTM(64, return_sequences=False)(token_embedding)
#token_lstm = Dropout(0.1)(token_lstm)
# token_lstm=AttentionWeightedAverage2()(tf.expand_dims(token_lstm, axis=1))
#token_lstm = Dense(64, activation='relu')(token_lstm)

twogram_input = Input(shape=(max_2gram_length,))
twogram_embedding =Embedding(input_dim=len(two_gram_to_index) + 1, output_dim=8, input_length=max_2gram_length)(twogram_input)
twogram_lstm = LSTM(64, return_sequences=False)(twogram_embedding) #two_gram_flat = Flatten()(two_gram_embedding)
#twogram_lstm = Dropout(0.2)(twogram_lstm)
# twogram_lstm=AttentionWeightedAverage2()(tf.expand_dims(twogram_lstm, axis=1))
#twogram_lstm = Dense(64, activation='relu')(twogram_lstm)

threegram_input = Input(shape=(max_3gram_length,))
threegram_embedding =Embedding(input_dim=len(three_gram_to_index) + 1, output_dim=8, input_length=max_3gram_length)(threegram_input)
threegram_lstm = LSTM(64, return_sequences=False)(threegram_embedding)
#threegram_lstm = Dropout(0.2)(threegram_lstm)
# threegram_lstm=AttentionWeightedAverage2()(tf.expand_dims(threegram_lstm, axis=1))
#threegram_lstm = Dense(64, activation='relu')(threegram_lstm)


wt_input = Input(shape=(max_wt_length,))
wt_embedding =Embedding(input_dim=len(wt_to_index) + 1, output_dim=embedding_dim, input_length=max_wt_length)(wt_input)
wt_lstm = LSTM(64, return_sequences=False)(wt_embedding)
#wt_lstm = Dropout(0.8)(wt_lstm)
# wt_lstm=AttentionWeightedAverage2()(tf.expand_dims(wt_lstm, axis=1))
#wt_lstm = Dense(64, activation='relu')(wt_lstm)

# 合并处理后的数据
feature_4ch = Concatenate()([domain_lstm, token_lstm, twogram_lstm, feature_dense])

feature_merge1 = AttentionWeightedAverage2()(tf.expand_dims(feature_4ch, axis=1))
feature_merge2 = AttentionWeightedAverage2()(tf.expand_dims(feature_4ch, axis=1))

feature_merge3 = Concatenate()([feature_merge1, feature_merge2])
feature_merge3 = AttentionWeightedAverage2()(tf.expand_dims(feature_merge3, axis=1))

softmax_out1 = Dense(labels_one_hot.shape[1], activation='softmax', name='out1')(feature_merge1)
softmax_out2 = Dense(labels_one_hot.shape[1], activation='softmax', name='out2')(feature_merge2)
softmax_out3 = Dense(labels_one_hot.shape[1], activation='softmax', name='out3')(feature_merge3)

loss_weights = {
    'out1': 1,
    'out2': 1,
    'out3': 1
}


outputs=[softmax_out1, softmax_out2, softmax_out3]
# 定义模型
model = Model(inputs=[domain_input, tld_input, feature_input,token_input,twogram_input,threegram_input,wt_input], outputs=outputs )
#featureEx_model = Model(inputs=[domain_input, tld_input, feature_input,token_input ,twogram_input,threegram_input,wt_input] ,outputs=feature)
print(model.summary())
# 编译模型
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


mse = MeanSquaredError()
# 定义Autoencoder重构损失
def autoencoder_loss(y_true, y_pred):
    # 确保 y_true 和 y_pred 是 float32 类型
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 计算交叉熵损失
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # 获取真实类别的索引（one-hot 编码 -> 类别索引）
    true_class_indices = tf.argmax(y_true, axis=1)

    # 获取预测类别的索引（softmax 输出 -> 类别索引）
    pred_class_indices = tf.argmax(y_pred, axis=1)
    # cost_matrix = np.random.rand(37, 37)
    cost_matrix = np.loadtxt("BSL_Confusion_Matrix360.txt", delimiter='\t')
    np.fill_diagonal(cost_matrix, 0)

    # 根据真实类别和预测类别索引，从代价矩阵中提取误分类代价
    batch_size = tf.shape(y_true)[0]
    batch_indices = tf.cast(tf.range(batch_size), tf.int64)
    indices = tf.stack([true_class_indices, pred_class_indices], axis=1)
    misclassification_cost = tf.gather_nd(cost_matrix, indices)

    # 总损失 = 交叉熵损失 + 误分类代价
    loss = tf.cast(cross_entropy, tf.float64) + tf.cast(misclassification_cost, tf.float64)
    return loss

loss_functions = [
    tf.keras.losses.categorical_crossentropy,  # softmax_out的损失
    lambda y_true, y_pred: autoencoder_loss(y_true, y_pred),
    tf.keras.losses.categorical_crossentropy,
]
model.compile(optimizer=Adam(), loss=loss_functions, loss_weights=loss_weights, metrics=['accuracy'])

# 加载权重或训练模型
if ExpConf.check_load_weights():
    ExpConf.load_weights(model)
else:
    history = model.fit( [X_train_domain, X_train_tld, X_train_feature,X_train_token_indices,X_train_two_gram_indices,X_train_three_gram_indices,X_train_wt_indices], y_train, epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.2, verbose=1
    )
    ExpConf.save_weights(model)
    ExpConf.log_train(history)

# 评估模型
#loss, accuracy = model.evaluate([X_test_domain, X_test_tld, X_test_feature,X_test_token_indices,X_test_two_gram_indices,X_test_three_gram_indices,X_test_wt_indices], y_test)
#ExpConf.log_evaluate(loss, accuracy)

# 预测并记录结果
y_pred = model.predict([X_test_domain, X_test_tld, X_test_feature,X_test_token_indices,X_test_two_gram_indices,X_test_three_gram_indices,X_test_wt_indices,])[2]
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
ExpConf.log_test(y_true_classes, y_pred_classes, label_binarizer.classes_)

# # 保存特征提取模型
# if ExpConf.check_load_weights(fmodel_flag=True):
#     ExpConf.load_weights(featureEx_model, fmodel_flag=True)
# else:
#     ExpConf.save_weights(featureEx_model, fmodel_flag=True)

'''
# 使用特征处理模型对数据集进行数值型转换
feature_outputs = featureEx_model.predict([domain_indices_array, tld_encoded, feature_array,token_indices_array,two_gram_indices_array,three_gram_indices_array,wt_indices_array ])
new_df = pd.DataFrame(feature_outputs, columns=[f'feature_{i}' for i in range(feature_outputs.shape[1])])
new_df['Label'] = df['Label']
# 数值化结果保存
new_df.to_csv('dataset/'+ExpConf.FILE_PREFIX+'.csv', index=False)
'''
