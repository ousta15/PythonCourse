import numpy as np, pandas as pd

def linearReg(X,y):
    X = X.to_numpy()
    y = y.to_numpy()

    if len(X) == len(y):

        X = np.hstack((np.ones(len(X)).reshape(len(X),1),X))

        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),y)
        yhat = np.matmul(X,beta)
        e = y - yhat
        e_var = np.matmul(np.transpose(e),e) / (len(y)-len(X[0])-1)
        b_var = np.diag(e_var*np.linalg.inv(np.matmul(np.transpose(X),X)))
        b_std_error = np.sqrt(b_var)
        t_stat = beta / b_std_error

        results = np.vstack((beta,b_std_error,t_stat)).transpose()

        df_results = pd.DataFrame(results,columns=["beta","std_err","t_stat"])
        df_results["[0.025"] = df_results["t_stat"] - 1.96*df_results["std_err"]
        df_results["0.975]"] = df_results["t_stat"] + 1.96 * df_results["std_err"]

        return df_results

    else:
        raise Exception("Number of rows of X and y should be the same.")


