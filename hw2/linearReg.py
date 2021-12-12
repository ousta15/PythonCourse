import numpy as np, pandas as pd
from scipy.stats import t

def linearReg(X,y):

    if len(X) == len(y):

        variable_names = X.columns
        total = pd.concat([X, y], axis=1)
        total = total.dropna()

        if len(total)>0:

            try:

                X = total.iloc[:,:-1].to_numpy()
                y = total.iloc[:,-1:].to_numpy()

                X = np.hstack((np.ones(len(X)).reshape(len(X),1),X))

                beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),y)
                yhat = np.matmul(X,beta)
                beta = beta.reshape(1,-1)
                e = y - yhat
                e_var = np.matmul(np.transpose(e),e) / (len(y)-len(X[0])-1)
                b_var = np.diag(e_var*np.linalg.inv(np.matmul(np.transpose(X),X)))
                b_std_error = np.sqrt(b_var)
                t_stat = np.abs(beta / b_std_error)

                results = (np.vstack((beta,b_std_error,t_stat)).transpose())

                df_results = pd.DataFrame(results,columns=["coef","std_err","t_stat"]).round(2)
                df_results["p_value"] = t.cdf(-abs(df_results["t_stat"]), len(X)-2).round(2)
                df_results["[0.025"] = round(df_results["coef"] - 1.96 * df_results["std_err"],4)
                df_results["0.975]"] = round(df_results["coef"] + 1.96 * df_results["std_err"],4)
                df_results = df_results.rename(index={0:"const"})

                for i in range(len(variable_names)):
                    df_results = df_results.rename(index={i+1: variable_names[i]})

            except:
                raise TypeError("Data should includes only integer and/or float data type")


            return df_results



        else: return "All rows contain at least one NAN value."

    else:
        return "Number of rows of X and y should be the same."


