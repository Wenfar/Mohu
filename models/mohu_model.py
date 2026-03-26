import numpy as np

class IterativeMohuDecision:
    def __init__(self, n, number, tol=1e-6, max_iter=100):
        self.n = n
        self.number = number
        self.tol = tol
        self.max_iter = max_iter

        self.x = np.zeros((number, n))
        self.y = np.zeros(number)
        self.a = np.zeros((n, n))
        self.b = np.zeros(n)
        self.u = np.zeros(number)
        self.beta = np.zeros(n + 1)

    # def setdatax(self, X):
    #     self.x = np.array(X, dtype=float)
    #     # 每次设置新数据时，重置 u 为全1向量
    #     self.number = self.x.shape[0]
    #     self.u = np.ones(self.number)  # 重置隶属度
    def setdatax(self, X):
        self.x = np.array(X, dtype=float)
        self.number = self.x.shape[0]
        self.n = self.x.shape[1]          # ← 关键修复：同步更新特征数

        # 同步重置所有依赖 n 的结构
        self.a = np.zeros((self.n, self.n))
        self.b = np.zeros(self.n)
        self.u = np.ones(self.number)
        self.beta = np.zeros(self.n + 1)  # 重置 beta（截距 + n个系数）

    def setdatay(self, y):
        self.y = np.array(y, dtype=float)
        # 这里不需要重置 u，因为 setdatax 已经做了
        
    def setdataxishu(self, beta_init):
        self.beta = np.array(beta_init, dtype=float)

    def _single_update(self):
        """
        执行一次迭代更新，计算新的回归系数
        该方法实现了加权最小二乘法的迭代更新过程
        """
        for i in range(self.number):
            # 计算当前样本的预测值
            y_hat = self.beta[0]
            for j in range(self.n):
                # 累加特征值与对应系数的乘积
                y_hat += self.beta[j + 1] * self.x[i, j]
            # 计算并存储权重，权重基于实际值与预测值的差异
            self.u[i] = np.exp(-(self.y[i] - y_hat) ** 2)

        # 构建正规方程的系数矩阵a
        for i in range(self.n):
            for j in range(self.n):
                # 计算各种加权和
                sumu = np.sum(self.u)
                sumxx = np.sum(self.u * self.x[:, i] * self.x[:, j])
                sumx1 = np.sum(self.u * self.x[:, i])
                sumx2 = np.sum(self.u * self.x[:, j])
                # 计算矩阵a的元素
                self.a[i, j] = sumu * sumxx - sumx1 * sumx2

        # 构建正规方程的常数向量b
        for i in range(self.n):
            # 计算各种加权和
            sumu = np.sum(self.u)
            sumxy = np.sum(self.u * self.x[:, i] * self.y)
            sumx = np.sum(self.u * self.x[:, i])
            sumy = np.sum(self.u * self.y)
            # 计算向量b的元素
            self.b[i] = sumu * sumxy - sumx * sumy

        #beta_rest = np.linalg.solve(self.a, self.b)
        lambda_reg = 1e-6
        beta_rest = np.linalg.solve(
            self.a + lambda_reg * np.eye(self.n),
            self.b
        )

        #隶属度加权均值
        sumu = np.sum(self.u)#分母
        # 检查sumu是否接近于零，避免除以零错误
        if sumu < 1e-10:  # 如果sumu太小，使用简单的平均值
            avg_x = np.mean(self.x, axis=0)
            avg_y = np.mean(self.y)
        else:#计算每个指标的加权均值x_j
            avg_x = np.array([
                np.sum(self.u * self.x[:, j]) / sumu
                for j in range(self.n)
            ])
            #计算目标变量的加权均值y
            avg_y = np.sum(self.u * self.y) / sumu
        #计算截距beta0
        beta0 = avg_y - np.dot(beta_rest, avg_x)
        #截距写入结果向量
        beta_new = np.zeros(self.n + 1)
        beta_new[0] = beta0
        beta_new[1:] = beta_rest

        return beta_new

    def fit(self):
        for k in range(self.max_iter):
            beta_new = self._single_update()
            if np.linalg.norm(beta_new - self.beta) < self.tol:
                self.beta = beta_new
                break
            self.beta = beta_new
        return self.beta

    def predict(self, X_new):
        X_new = np.asarray(X_new)
        return self.beta[0] + X_new @ self.beta[1:]
    
    #普通最小二乘回归（零次回归）
    def ols_regression(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])#加一列常数项
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        return beta

    #误差指标
    def calc_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    def calc_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def add_outliers(X, y, ratio=0.15, scale=5):
        X_new = X.copy()
        y_new = y.copy()
        n_out = int(len(y) * ratio)
        idx = np.random.choice(len(y), n_out, replace=False)
        X_new[idx] *= scale
        y_new[idx] *= scale
        return X_new, y_new
    

    #数据验证。对比模糊回归和OSL回归的性能，评估拟合效果和鲁棒性（异常值影响）
    def validate_fuzzy_model(X, y, fuzzy_model, ols_model):
    # 原始数据
        beta_f = fuzzy_model.fit()
        ols_model.fit(X, y)

        rmse_f = np.sqrt(np.mean((y - fuzzy_model.predict(X))**2))
        rmse_o = np.sqrt(np.mean((y - ols_model.predict(X))**2))

        r2_f = r2_score(y, fuzzy_model.predict(X))
        r2_o = r2_score(y, ols_model.predict(X))

        # 异常扰动
        X_bad, y_bad = add_outliers(X, y, ratio=0.2, scale=5)

        beta_f_bad = fuzzy_model.fit(X_bad, y_bad)
        ols_model.fit(X_bad, y_bad)

        delta_f = np.linalg.norm(beta_f - beta_f_bad)
        delta_o = np.linalg.norm(ols_model.coef_ - ols_model.coef_)

        return {
            "RMSE_fuzzy": rmse_f,
            "RMSE_OLS": rmse_o,
            "R2_fuzzy": r2_f,
            "R2_OLS": r2_o,
            "delta_fuzzy": delta_f,
            "delta_OLS": delta_o
        }
    
    #内部验证。只验证模糊回归自身，评估拟合效果RMSE、R2。
    def validate(self, X, y):
        """
        模糊回归只用于验证指标体系是否能有效解释目标变量，不参与赋权
        """
        y_pred = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return {
            "rmse": rmse,
            "r2": r2,
            "coef": self.beta
        }



  

    




