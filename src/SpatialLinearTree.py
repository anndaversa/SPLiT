import logging
logging1 = logging.getLogger('SP')
from sklearn.linear_model import *
from .lineartree import LinearTreeRegressor
from .utils import *


class SPLIT():
    def __init__(self, X_train, y_train, X_test, y_test, X_valid, y_valid, num_targets, spatial=False, drop_train_node=False, id_key="", distance_matrix=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.spatial = spatial
        self.drop_train_node = drop_train_node
        self.spatial_leaf_count = 0
        self.num_targets = num_targets
        self.id_key = id_key
        self.distance_matrix = distance_matrix


    def check_node_spatial(self, Node, leaves, val_score, weighted_error_leaves=None):
        if self.spatial:
            X_train_sp = add_features_closeness(self.X_train.iloc[Node.samples_id, :], self.X_train, self.distance_matrix)
            y_train_leaves = self.y_train.iloc[Node.samples_id, :]
            cols = [col for col in X_train_sp.columns if "right" in col]

        m = LinearRegression()
        X_train_sp = X_train_sp.reindex(X_train_sp.columns, axis=1)
        m.fit(X_train_sp, y_train_leaves)
        coeff = m.coef_
        intercept = m.intercept_

        X_val = self.X_valid.iloc[leaves.loc[lambda x: x == Node.id].index, :].copy()
        y_val = self.y_valid.iloc[leaves.loc[lambda x: x == Node.id].index, :].copy()

        if self.spatial:
            X_valid_sp = add_features_closeness(X_val, self.X_valid, self.distance_matrix)
            predValid = m.predict(X_valid_sp)

        real_pred_valid = pd.concat([y_val.reset_index(), pd.DataFrame(predValid).round(3)], axis=1).iloc[:, 1:]
        real_pred_valid["leaf"] = Node.id

        val_score_spatial = real_pred_valid.groupby("leaf", group_keys=False).apply(apply_score)[["rse"]].drop_duplicates()

        
        logging1.info(f'NO SPATIAL {np.round(val_score[val_score["leaf"] == Node.id]["rse"].values[0], 3)} '
                        f'- {self.spatial, np.round(val_score_spatial["rse"].values[0], 3)}')
        rse_check = val_score[val_score["leaf"] == Node.id]["rse"].values

        if (val_score_spatial["rse"].values < rse_check):
            Node.model.coef_ = coeff
            Node.model.intercept_ = intercept
            Node.model.n_features_in_ = coeff.shape[1]
            Node.spatial_info = X_train_sp[cols].copy()
            self.spatial_leaf_count += 1

            # update val_score
            val_score.loc[val_score.set_index("leaf").index == Node.id, "rse"] = val_score_spatial["rse"].values[0]

        logging1.info(f"Num coeff: {Node.model.coef_.shape}")
        logging1.info("-------")

        return val_score


    def ModLinearTree(self):
        regr = LinearTreeRegressor(LinearRegression())
        regr.set_params(**{"min_samples_leaf":0.05, "n_jobs":-1})
        regr.fit(self.X_train.drop(["date",self.id_key], axis=1,errors='ignore'), self.y_train)

        # For each node, save the indices of instances that fell into that node in 'samples_id'
        save_index_sample(regr, self.X_train.drop(["date",self.id_key], axis=1,errors='ignore'))

        y_pred = regr.predict(self.X_valid.drop(["date",self.id_key], axis=1,errors='ignore'))

        # Return the index of the leaf that each sample of Validation is predicted as
        leaves = pd.Series(regr.apply(self.X_valid.drop(["date",self.id_key], axis=1,errors='ignore')), name='leaf')

        real_pred = pd.concat([self.y_valid.reset_index(), pd.DataFrame(y_pred).round(3)], axis=1)
        real_pred_leaves = pd.concat([real_pred, leaves], axis=1).iloc[:, 1:]

        # compute error for each leaf node
        val_score = real_pred_leaves.groupby("leaf", group_keys=False).apply(apply_score)[
            ["leaf", "rse"]].drop_duplicates()

        if self.spatial != False:
            # Compute spatial features for the training data only on leaves where a validation instance has fallen.
            # Save the lat,long + spatial feat structure in the "spatial_info" attribute of the Node class.
            for L in regr._leaves.values():
                if L.id in val_score.leaf.values:
                    val_score = self.check_node_spatial(L, leaves, val_score)

            print(f"Leaves with spatial features {self.spatial_leaf_count}/{len(val_score.leaf.unique())}")


        spatial_count = 0
        for L in regr._leaves.values():
            if L.spatial_info is not None:
                spatial_count += 1

        spatial_global = [spatial_count, len(regr._leaves), (spatial_count/ len(regr._leaves))*100]

        global_wins = self.spatial_leaf_count 
        global_attempts = len(val_score.leaf.unique()) 

        spatial_wins_attempts = [self.spatial_leaf_count, 
                              len(val_score.leaf.unique()),(global_wins/global_attempts) * 100]
        return regr, spatial_global, spatial_wins_attempts

    def predict(self, regr, distance_matrix):
        leaves_test = pd.Series(regr.apply(self.X_test.drop(["date",self.id_key], axis=1,errors='ignore')), name='leaf')
        real_pred_test = pd.DataFrame()
        for L in regr._leaves.values():
            if L.id in leaves_test.unique():
                X_test_m = self.X_test.iloc[leaves_test.loc[lambda x: x == L.id].index, :].copy()
                y_test_m = self.y_test.iloc[leaves_test.loc[lambda x: x == L.id].index, :].copy()

                if L.spatial_info is None:
                    y_test_pred = L.model.predict(X_test_m.drop(["date", self.id_key], axis=1,errors='ignore').values)
                elif self.spatial:
                    X_test_sp = add_features_closeness(X_test_m, self.X_test,distance_matrix)
                    y_test_pred = L.model.predict(X_test_sp.values)
                real_pred = pd.concat([y_test_m.reset_index(), pd.DataFrame(y_test_pred).round(3)], axis=1)
                real_pred_test = pd.concat([real_pred_test,real_pred], axis=0, ignore_index=True)

        return real_pred_test





