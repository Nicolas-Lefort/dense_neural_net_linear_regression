'''
Deep Neural Network with mixed Numerical-Categorical Input

https://www.kaggle.com/shivachandel/kc-house-data
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Note: the second dataset is really small and ML ensemble method shall be explored instead of DL

- collect data into pandas
- split data into categorical/numerical sets
- eliminate low features/columns and deal with missing values
- remove outliers and normalize (numerical) data
- run a random search model using keras tuners
- run a mixed inputs model
- save and load models

mode 0 : run a simple model
mode 1 : run a random search
mode 2 : run a mixed-input model

output figures:
"_features_before_cleaning.png"
"_features_after_cleaning.png"
"_correlation_before_cleaning.png"
"_correlation_after_cleaning.png"
"_optimization.png"
"_pred_vs_true.png"
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from kerastuner.tuners import RandomSearch
from keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping


def prepare_data(df, target, project_name):
    # resume
    show_data_summary(df, target)
    # remove poor data
    df = df.drop_duplicates()
    # keep where target not NaN
    df = df[df[target].notna()]
    # process data according to types
    df_numerical = process_numerical_data(df, target)
    df_categorical = process_categorical_data(df, target)
    # join back data
    df = df_numerical.join(df_categorical)
    # resume
    show_data_summary(df, target)
    # remove target columns from features
    X = df.drop(target, axis=1)
    # isolate target
    Y = df[target].to_frame()
    # train_test_split from sklearn
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # identify outliers
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(x_train)
    mask = yhat == 1
    # select all rows that are not outliers
    x_train=x_train[mask]
    y_train=y_train[mask]
    # normalize numerical data
    x_train_scaled, x_test_scaled = scale(x_train, x_test)
    # input shape
    input_shape = len(x_train_scaled.columns)
    # plot data after pre-processing
    plot_correlation_matrix(x_train_scaled.join(y_train), target, project_name, title="after_cleaning")
    plot_features(x_train_scaled.join(y_train), target, project_name, title="after_cleaning")
    # create datasets
    dataset, dataset_num, dataset_hot = build_dataset(x_train_scaled, x_test_scaled, y_train, y_test, input_shape)

    return [dataset, dataset_num, dataset_hot]

def build_dataset(x_train, x_test, y_train, y_test, input_shape):
    dataset = [x_train, x_test, y_train, y_test]
    dataset_num = get_numerical_data(dataset, target)
    dataset_hot = get_categorical_data(dataset, target)
    dataset.append(input_shape)
    dataset_num.append(len(dataset_num[0].columns))
    dataset_hot.append(len(dataset_hot[0].columns))
    return [dataset, dataset_num, dataset_hot]

def get_numerical_data(dataset, target):
    dataset_num = []
    for df in dataset:
        if target in df.columns and df[target].dtype not in ['int64', 'float64']:
            dataset_num.append(df.select_dtypes(include=['int64', 'float64']).join(df[target].to_frame()))
        else:
            dataset_num.append(df.select_dtypes(include=['int64', 'float64']))
    return dataset_num

def get_categorical_data(dataset, target):
    dataset_hot = []
    # 'uint8' in the defaults dtype resulting from get_dummies
    for df in dataset:
        if target in df.columns and df[target].dtype not in ['uint8', 'uint8']:
            dataset_hot.append(df.select_dtypes(include=['uint8']).join(df[target].to_frame()))
        else:
            dataset_hot.append(df.select_dtypes(include=['uint8']))
    return dataset_hot

def scale(x_train, x_test):
    # split data types
    x_train_numerical = x_train.select_dtypes(include=['int64', 'float64'])
    x_train_categorical = x_train.select_dtypes(include=['uint8'])
    x_test_numerical = x_test.select_dtypes(include=['int64', 'float64'])
    x_test_categorical = x_test.select_dtypes(include=['uint8'])
    # scale numerical input
    scaler = MinMaxScaler()
    x_train_numerical_scaled = scaler.fit_transform(x_train_numerical)
    x_test_numerical_scaled = scaler.transform(x_test_numerical)
    # convert back to dataframe
    x_train_numerical_scaled = pd.DataFrame(x_train_numerical_scaled, index=x_train_numerical.index, columns=x_train_numerical.columns)
    x_test_numerical_scaled = pd.DataFrame(x_test_numerical_scaled, index=x_test_numerical.index, columns=x_test_numerical.columns)
    # join back data
    x_train_scaled = x_train_numerical_scaled.join(x_train_categorical)
    x_test_scaled = x_test_numerical_scaled.join(x_test_categorical)

    return x_train_scaled, x_test_scaled

def process_categorical_data(df, target):
    # select categorical types + target
    df = df.select_dtypes(include=['object', 'bool']).join(df[target].to_frame())
    n, p = df.shape
    # lower all string
    df = df.applymap(lambda s:s.lower() if type(s) == str else s)
    # one hot encoding
    X = df.drop(target, axis=1)
    X = pd.get_dummies(X, drop_first=False)
    Y = df[target].to_frame()
    model = LinearRegression()
    # fit the model
    model.fit(X, Y)
    # summarize feature importance
    df = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
    df = df.abs().sort_values(ascending = False, by="Coefficient").head(5*p)
    top_features=df.index.tolist()
    X = X[top_features]

    return X

def process_numerical_data(df, target):
    # select numerical types
    df = df.select_dtypes(include=['int64', 'float64'])
    df0 = df
    # plott correlation matrix
    plot_correlation_matrix(df, target, project_name, title="before_cleaning")
    plot_features(df, target, project_name, title="before_cleaning")
    # remove low features/columns
    min_corr = 0.1
    df_corr = df.corr().abs()[target].sort_values(ascending=False).to_frame()
    low_feature = df_corr.index[df_corr[target] < min_corr].tolist()
    # manage missing values
    s = df.isna().sum()
    s = s[s > 0].sort_values(ascending = False)
    df = pd.DataFrame({'feature':s.index, 'number_missing':s.values})
    #df.columns = ['feature','number_missing']
    df['percentage_missing'] = (100.0*df['number_missing'])/len(df0)
    missing = pd.Series(df.percentage_missing.values,index=df.feature).to_dict()

    for feature, rate in missing.items():
        # remove feature with missing rate > 80 %
        if rate > 80:
            low_feature.append(feature)
        # remove feature with missing rate > 60 % and  correlation factor < 0.3
        if rate >= 60 and df.corr().abs()[target].loc[feature]<=0.3:
            low_feature.append(feature)
        if rate >= 50 and df.corr().abs()[target].loc[feature]<=0.4:
            low_feature.append(feature)

    # remove low_feature from dataframe
    df = df0.drop(low_feature, axis=1)
    print("removed features :" , low_feature)
    # impute residual missing values / if enough data, just drop with df.dropna()
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df)
    df = pd.DataFrame(imputer.transform(df), index=df.index, columns=df.columns)
    # plot features and correlations
    plot_correlation_matrix(df, target, project_name, title="after_cleaning")
    plot_features(df, target, project_name, title="after_cleaning")

    return df


def create_model_num(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Dense(units=100, activation='relu')(inputs)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dropout(rate=0.4)(x)
    outputs = Dense(units=1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def create_model_hot(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Dense(units=200, activation='relu')(inputs)
    x = Dense(units=200, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=40, activation='relu')(x)
    x = Dropout(rate=0.4)(x)
    outputs = Dense(units=1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def create_simple_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Dense(units=200, activation='relu')(inputs)
    x = Dense(units=200, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=20, activation='relu')(x)
    x = Dropout(rate=0.4)(x)
    outputs = Dense(units=1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def create_mixed_model(model1, model2):
    inputs = concatenate([model1.output, model2.output])
    x = Dense(100, activation="relu")(inputs)
    x = Dense(100, activation="relu")(x)
    x = Dense(100, activation="relu")(x)
    x = Dense(50, activation="relu")(x)

    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=[model1.input, model2.input], outputs=outputs)

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mean_squared_error'])

    return model


def train_model(model, x_train, x_test, y_train, y_test):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.getcwd(),
                                                 save_weights_only=True,
                                                 verbose=1)
    # train the model
    history = model.fit(x=x_train,
                    y=y_train,
                    validation_data=(x_test,y_test),
                    batch_size=128,
                    epochs=200,
                    callbacks = [checkpoint])

    return model, history

def train_mixed_model(model_mix, x_train_num, x_train_hot, x_test_num, x_test_hot, y_train, y_test):
    print(model_mix.summary())
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.getcwd(),
                                                 save_weights_only=True,
                                                 verbose=1)
    # train the model
    history = model_mix.fit(x=[x_train_num, x_train_hot],
                    y=y_train,
                    validation_data=([x_test_num, x_test_hot],y_test),
                    batch_size=128,
                    epochs=200,
                    callbacks = [checkpoint])

    return model_mix, history

def build_model_v1(hp):
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(
            Dense(
                units=hp.Int('units_' + str(i),min_value=50,max_value=300,step=10),
                activation='relu'))
        model.add(
            Dropout(
                hp.Float('dropout_',min_value=0.0,max_value=0.8,default=0.5,step=0.1)))

    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse',
                  metrics=['mean_squared_error'])

    return model


def model_search(x_train, x_test, y_train, y_test, project_name):
    tuner = RandomSearch(
    build_model_v1,
    overwrite=True,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=3,
    directory="random_search",
    project_name=project_name)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="random_search_untrained_"+project_name+'.h5',
                                                 save_weights_only=False,
                                                 verbose=1,
                                                 save_best_only=True)

    tuner.search(x_train, y_train, epochs=4, validation_data=(x_test, y_test),
                 use_multiprocessing=True,batch_size=256, callbacks=[checkpoint])

    return tuner.get_best_models(num_models=1)[0]

def plot_features(df, target, project_name, title):
    top_features = df.corr().abs()[target].sort_values(ascending=False).to_frame()
    top_features = top_features.drop(target, axis=1)
    plt.figure(figsize=(10,5))
    cols = top_features.index.values.tolist()
    sns.pairplot(df,
                 x_vars=cols[:10],
                 y_vars=[target],
                 height=2)
    plt.title(title)
    plt.savefig(project_name+'_num_features_'+title+'.png')

def plot_metrics(model, history, project_name):
    list_metrics = ["loss"]#model.metrics_names
    df_results = pd.DataFrame(history.history)
    for metric in list_metrics:
        df_mectric= df_results[[metric,'val_'+ metric]]
        df_mectric.plot(title='Model ' + metric, figsize=(12, 8)).set(xlabel='Epoch', ylabel=metric)
    plt.savefig(project_name+'_optimization.png')

def plot_pred_true(model, x_test, y_test, project_name):
    predictions = model.predict(x_test).reshape(-1)
    # Visualizing Our predictions
    plt.figure(figsize=(10,5))
    plt.scatter(y_test,predictions)
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    # Perfect predictions
    plt.plot(y_test,y_test,'r')
    plt.savefig(project_name+'_pred_vs_true.png')

def plot_correlation_matrix(df, target, project_name, title):
    top_features = df.corr().abs()[target].sort_values(ascending=False).head(30)
    plt.figure(figsize=(5,10))
    sns.heatmap(top_features.to_frame(),cmap='rainbow',annot=True,annot_kws={"size": 16},vmin=0)
    plt.title(title)
    plt.savefig(project_name+'_correlation_'+title+'.png')

def plot(model, history, x_test, y_test, project_name):
    # plot metrics
    plot_metrics(model, history, project_name)
    # plot prediction vs true
    plot_pred_true(model, x_test, y_test, project_name)


def show_data_summary(df, target):
    print("**********           START            **********")
    print(df.info())
    print("************************************************")
    print(df.corr().abs()[target].sort_values(ascending=False).head(30))
    print("************************************************")
    print(df.isnull().sum())
    print("************************************************")
    print(df.columns)
    print("************************************************")
    print(df.describe())
    print("**********            END             **********")

if __name__ == "__main__":
    project_name = "house"
    # import data
    df = pd.read_excel("kc_house_data.xls")
    # feature to predict
    target = "price"
    # clean and normalize data
    dataset, dataset_num, dataset_hot = prepare_data(df=df, target=target, project_name=project_name)
    x_train, x_test, y_train, y_test, input_shape = dataset
    x_train_num, x_test_num, y_train_num, y_test_num, input_shape_num = dataset_num
    x_train_hot, x_test_hot, y_train_hot, y_test_hot, input_shape_hot = dataset_hot

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # 0 <= mode <= 2
    mode = 1
    # mode 0 : ceate and train a model
    if mode == 0:
        prefix = "my_simple_model"
        model = create_simple_model(input_shape=input_shape)
        model, history = train_model(model, x_train, x_test, y_train, y_test)
        model.save(prefix + "_" + project_name + '.h5')
        plot(model, history, x_test, y_test, project_name)
    # mode 1 : let tensorflow search for a model and save it to project_name
    if mode == 1:
        prefix = "my_random_search"
        model = model_search(x_train, x_test, y_train, y_test, project_name)
        model, history = train_model(model, x_train, x_test, y_train, y_test)
        model.save(prefix + "_" +  project_name + '.h5')
        plot(model, history, x_test, y_test, project_name)
    # mode 3 : run a mixed input model
    if mode == 2:
        prefix = "my_mixed_model"
        model_num = create_model_num(input_shape_num)
        model_hot = create_model_hot(input_shape_hot)
        model_mix = create_mixed_model(model_num, model_hot)
        model_mix, history = train_mixed_model(model_mix, x_train_num, x_train_hot, x_test_num, x_test_hot, y_train, y_test)
        model_mix.save(prefix + "_" + project_name + '.h5')
        plot(model=model_mix, history=history, x_test=[x_test_num, x_test_hot], y_test=y_test, project_name=project_name)


