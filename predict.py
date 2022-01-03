import pandas_ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def get_prediction(symbol='A', dataset='NYSE'):
    data_paths = [r'NYSE\back_data.csv', r'NASDAQ\back_data.csv']
    if dataset == 'NYSE':
        data = pd.read_csv(data_paths[0])
    else:
        data = pd.read_csv(data_paths[1])
    data = data.rename(columns={'Unnamed: 0': 'Index'})
    data = data.set_index('Index')

    
    selection =  data.loc[data['symbol'] == symbol].reset_index()
    selection.ta.ema(close='close', length=10, append=True)
    selection = selection.iloc[10:].set_index('Index').reset_index()
    X_train, X_test, y_train, y_test = train_test_split(selection[['close']], selection[['EMA_10']], test_size=0.2)
    
    print(X_test.describe())
    print(X_train.describe())

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Printout relevant metrics
    print("Model Coefficients:", round(model.coef_[0][0], 4))
    print("Mean Absolute Error:",round(mean_absolute_error(y_test, y_pred), 4))
    print("Coefficient of Determination:", round(r2_score(y_test, y_pred),4))

    prediction_next_price = model.predict([selection[['close']].iloc[-1]])
    all_predictions = np.array(model.predict(selection[['close']]))
    results = [[round(prediction_next_price[0][0], 2), round(model.coef_[0][0], 4), round(mean_absolute_error(y_test, y_pred), 4), round(r2_score(y_test, y_pred), 4)]]
    results_df = pd.DataFrame(results, columns=['Predicted Current Price', 'Model Coefficient', 'Mean Absolute Error', 'Coefficient of Determination'])
    return results_df, all_predictions
    
    # print(selection['close'].shape[0])
    # print(all_predictions.shape[0])
    # # Plot outputs
    # plt.scatter(selection['close'], selection['EMA_10'], color="black")
    # plt.plot(selection['close'], results_df['All Predictions'][0], color="blue", linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.show()


    # data.ta.ema(close='close', length=10, append=True)

    # print(data.head(20))

#get_prediction()