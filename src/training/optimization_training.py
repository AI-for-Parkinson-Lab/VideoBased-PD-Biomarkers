import os
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lightgbm as lgb
import ast  
from sklearn.preprocessing import label_binarize
import optuna
#from optuna.integration import SklearnPruningCallback
from tqdm import tqdm
import gc
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (255, 0, 0)  # vibrant red

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.results = []  
        self.hyperparams_results = []  
        self.id2vid = pd.read_csv(config['id2vid'], header=None)
        self.video_labels = pd.read_csv(config['vid2score'])
        features_csv_path = os.path.join(self.config['save_path'], 'combined_features.csv')
        df = pd.read_csv(features_csv_path)       
        self.patient_ids = df['ids'].astype(str).str.strip("[]'").unique()
        self.set_paths()
        self.n_jobs = 1
        self.trials = 100
        self.random_state = 42
        self.stratified_split = True
    def set_paths(self):

        self.dynamic_csv = os.path.join(self.config['save_path'], 'dynamic_save.csv')
        features_csv_path = os.path.join(self.config['save_path'], 'combined_features.csv')
        self.features_df = pd.read_csv(features_csv_path)

    def _extract_features_from_csv(self, video_path):

        matching_row = self.features_df[self.features_df['video_path'] == video_path]
        
        if matching_row.empty:
            #print(f"No matching row found for video path: {video_path}")
            return None
        else:

            features = matching_row.iloc[0, 3:].values
            return features

    def evaluate_model(self, name, model, X_train, y_train, X_test, y_test, fold, classification_type, best_params):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
              
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        acceptable_accuracy = np.mean(np.abs(y_pred - y_test) <= 1)

        if len(np.unique(y_test)) > 1:
           all_classes = np.arange(5)  # Assuming your 5 classes are labeled as 0, 1, 2, 3, 4
           y_test_binarized = label_binarize(y_test, classes=all_classes)
   
           # Initialize y_pred_proba_full with zeros for all classes
           y_pred_proba_full = np.zeros((y_pred.shape[0], len(all_classes)))
   
           present_classes = model.classes_
           for idx, cls in enumerate(present_classes):
               class_index = np.where(all_classes == cls)[0][0]  # Find the corresponding index in all_classes
               y_pred_proba_full[:, class_index] = y_pred_proba[:, idx]
   
           valid_columns = []
           for i in range(y_test_binarized.shape[1]):
               unique_values = np.unique(y_test_binarized[:, i])
               if len(unique_values) > 1:
                   valid_columns.append(i)
   
           if valid_columns:
               # Select only valid columns from y_test_binarized and y_pred_proba_full
               y_test_filtered = y_test_binarized[:, valid_columns]
               y_pred_proba_filtered = y_pred_proba_full[:, valid_columns]
   
               # Calculate the ROC AUC score for the valid columns
               auc = roc_auc_score(y_test_filtered, y_pred_proba_filtered, multi_class='ovr')
           else:
               # Set AUC to NaN if no valid columns are left
               auc = np.nan
        else:
           # Set AUC to NaN if only one class is present in y_test
           auc = np.nan
       

       
        # Initialize default hyperparameter values
        result_row = {
            'fold': fold,
            'ids': self.test_patients,
            'classification_type': classification_type,
            'model': name,
            'kernel': np.nan,
            'C': np.nan,
            'gamma': np.nan,
            'n_estimators': np.nan,
            'max_depth': np.nan,
            'min_samples_split': np.nan,
            'min_samples_leaf': np.nan,
            'max_features': np.nan,
            'num_leaves': np.nan,
            'learning_rate': np.nan,
            'min_child_samples': np.nan,
            'subsample': np.nan,
            'colsample_bytree': np.nan,
            'reg_alpha': np.nan,
            'reg_lambda': np.nan,
            'solver': np.nan,
            'penalty': np.nan,
            'max_iter': np.nan,
            'AUC': auc,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'acceptable_accuracy': acceptable_accuracy,
            'f1': f1,
            'precision': precision,

        }
    
        for param, value in best_params.items():
            if param in result_row:
                result_row[param] = value
    
   
        # Store the result row for dynamic saving
        self.result_row = result_row
        self.results.append(result_row)
    def ordinal_binary_classification_fs_our(self, model_name, best_params, X_train, y_train, X_test, y_test , fold ,classification_type):
            models = []
            thresholds = [1, 2, 3, 4]
            aucs = []
            for threshold in thresholds:
                y_train_binary = (np.array(y_train) >= threshold).astype(int)
                y_test_binary = (np.array(y_test) >= threshold).astype(int)
        
                if model_name == 'Logistic Regression':
                    model = LogisticRegression(random_state=self.random_state, **best_params)

                elif model_name == 'Random Forest':
                    model = RandomForestClassifier(random_state=self.random_state, **best_params)

                elif model_name == 'LightGBM':
                    model = lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1, **best_params)

        
                model.fit(X_train, y_train_binary)
                models.append(model)
        
                    
        
            # Get P(y >= k) for each threshold
            proba_thresholds = [model.predict_proba(X_test)[:, 1] for model in models]
            proba_thresholds = np.vstack(proba_thresholds)  # shape: (4, N)
        
            # Compute ordinal class probabilities
            n_samples = proba_thresholds.shape[1]
            ordinal_probs = np.zeros((n_samples, 5))
            ordinal_probs[:, 0] = 1 - proba_thresholds[0]
            for i in range(1, 4):
                ordinal_probs[:, i] = proba_thresholds[i - 1] - proba_thresholds[i]
            ordinal_probs[:, 4] = proba_thresholds[3]
        
            # Final prediction = argmax(P(y = k))
            y_pred = np.argmax(ordinal_probs, axis=1)
        

            acceptable_accuracy = np.mean(np.abs(y_pred - y_test) <= 1)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
        
        
     
        
            result_row = {
                'fold': fold,
                'ids': self.test_patients,
                'classification_type': classification_type,
                'model': model_name,
                'kernel': np.nan,
                'C': np.nan,
                'gamma': np.nan,
                'n_estimators': np.nan,
                'max_depth': np.nan,
                'min_samples_split': np.nan,
                'min_samples_leaf': np.nan,
                'max_features': np.nan,
                'num_leaves': np.nan,
                'learning_rate': np.nan,
                'min_child_samples': np.nan,
                'subsample': np.nan,
                'colsample_bytree': np.nan,
                'reg_alpha': np.nan,
                'reg_lambda': np.nan,
                'solver': np.nan,
                'penalty': np.nan,
                'max_iter': np.nan,
                'AUC': np.nan,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'acceptable_accuracy': acceptable_accuracy,
                'f1': f1,
                'precision': precision
            }
        
            for param, value in best_params.items():
                if param in result_row:
                    result_row[param] = value
        
            # Store the result row for dynamic saving
            self.result_row = result_row
            self.results.append(result_row)
    def extract_feat_myhc(self, train_patients, test_patients):

        # Prepare training data
        train_videos = []
        train_labels = []
        for patient in train_patients:
            video_list = ast.literal_eval(self.id2vid[self.id2vid[0] == patient].iloc[0, 1])
            for video in video_list:
                matching_rows = self.video_labels[self.video_labels['video_path'].str.contains(video, regex=False)]
                if not matching_rows.empty:
                    for _, row in matching_rows.iterrows():
                        train_videos.append(row['video_path'])
                        train_labels.append(row['score'])

        # Prepare test data
        test_videos = []
        test_labels = []
        for patient in test_patients:
            video_list = eval(self.id2vid[self.id2vid[0] == patient].iloc[0, 1])
            for video in video_list:
                matching_rows = self.video_labels[self.video_labels['video_path'].str.contains(video, regex=False)]
                if not matching_rows.empty:
                    for _, row in matching_rows.iterrows():
                        test_videos.append(row['video_path'])
                        test_labels.append(row['score'])

        print(f"Train set: {len(train_videos)} videos, Train labels: {len(train_labels)}")
        print(f"Test set: {len(test_videos)} videos, Test labels: {len(test_labels)}")
        
        train_videos = [path.replace('//chansey.umcn.nl', '/data').replace('\\', '/') for path in train_videos]
        test_videos = [path.replace('//chansey.umcn.nl', '/data').replace('\\', '/') for path in test_videos]
        
        train_videos = [path.replace('Video\\', 'Video/', 1) for path in train_videos]
        test_videos = [path.replace('Video\\', 'Video/', 1) for path in test_videos]
        
        train_features = []
        modified_train_labels = []
        for vid, label in tqdm(zip(train_videos, train_labels), total=len(train_videos), desc="Processing training videos"):
    

            feature = self._extract_features_from_csv(vid)                    
            if feature is not None:
                train_features.append(feature)
                modified_train_labels.append(label)
                
        
        test_features = []
        modified_test_labels = []
        for vid, label in tqdm(zip(test_videos,test_labels), total=len(test_videos), desc="Processing test videos"):
            feature = self._extract_features_from_csv(vid)                    
            if feature is not None:
                test_features.append(feature)
                modified_test_labels.append(label)
        # Standardize the features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)        
        
        return train_features_scaled, modified_train_labels, test_features_scaled, modified_test_labels
    


    
    def run_cross_validation(self):
       
        # Replace KFold with LeaveOneOut
        # Leave-One-Out setup
        loo = LeaveOneOut()
        total_folds = loo.get_n_splits(self.patient_ids)
        
        for fold, (train_idx, test_idx) in enumerate(loo.split(self.patient_ids), 1):
            percent = (fold / total_folds) * 100
            if fold>=171:
                print(f"Fold {fold}/{total_folds} ({percent:.1f}%)")
                self.patient_ids = np.array(self.patient_ids)
                # Select actual patient IDs from indices
                train_patients = self.patient_ids[train_idx]
                test_patients = self.patient_ids[test_idx]
                
                self.test_patients = test_patients
    
                # Multi-class
                self.train_and_evaluate_models(train_patients, test_patients, fold, classification_type='multi')
            
                # Ordinal
                self.train_and_evaluate_models(train_patients, test_patients, fold, classification_type='ordinal')


        
    def optimize_hyperparameters_multi(self, model_name, X_train, y_train, fold):
    
        y_train = np.array(y_train)
        
        def objective(trial):
            if model_name == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                }
                model = RandomForestClassifier(random_state=self.random_state, **params)

           
            elif model_name == 'Logistic Regression':
                solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
                
                # Adjust penalty options based on solver
                if solver in ['liblinear']:
                    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
                else:
                    penalty = 'l2'  # lbfgs only supports 'l2'
                
                params = {
                    'C': trial.suggest_float('C', 0.01, 10, log=True),
                    'solver': solver,
                    'penalty': penalty,
                    'max_iter': trial.suggest_int('max_iter', 100, 300)  
                }
                
                #model = LogisticRegression(random_state=42, **params)
                model = LogisticRegression(random_state=self.random_state, **params)

            elif model_name == 'LightGBM':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
                }
                
                model = lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1, **params)

            if self.stratified_split==True:
                inner_skf = StratifiedKFold(n_splits=3, shuffle=True)
                scores = []
                for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
                   
                    inner_X_train, inner_X_val = X_train[inner_train_idx], X_train[inner_val_idx]
                    inner_y_train, inner_y_val = y_train[inner_train_idx], y_train[inner_val_idx]
                    
                    model.fit(inner_X_train, inner_y_train)
                    y_pred = model.predict(inner_X_val)
                    balanced_acc = balanced_accuracy_score(inner_y_val, y_pred)
                    acc = accuracy_score(inner_y_val, y_pred)
                    score = acc
                    scores.append(score)
                
            if self.stratified_split==False:    
                # Inner cross-validation with a larger number of folds to ensure robustness
                inner_kf = KFold(n_splits=3, shuffle=True)
                scores = []
                for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                   
                    inner_X_train, inner_X_val = X_train[inner_train_idx], X_train[inner_val_idx]
                    inner_y_train, inner_y_val = y_train[inner_train_idx], y_train[inner_val_idx]
                    
                    model.fit(inner_X_train, inner_y_train)
                    y_pred = model.predict(inner_X_val)
                    balanced_acc = balanced_accuracy_score(inner_y_val, y_pred)
                    acc = accuracy_score(inner_y_val, y_pred)
                    score = acc
                    scores.append(score)
                
                
                
            
            return np.mean(scores)
    
        # Create an Optuna study with pruning callback (useful for extensive tuning)
        study = optuna.create_study(study_name=f'multi_{model_name}', direction='maximize')
        study.optimize(objective, n_trials=self.trials, n_jobs= self.n_jobs, gc_after_trial = True)  
    
        # Save the best hyperparameters found for this model
        best_params = study.best_params
        del study
        gc.collect()
        return best_params
    def optimize_hyperparameters_for_ordinal(self, model_name, X_train, y_train, fold):

        y_train = np.array(y_train)
        thresholds = [1, 2, 3, 4]  # Define thresholds for ordinal classification
    
        def objective(trial):
            # Define a shared set of hyperparameters for all binary classifiers
            if model_name == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                }
                model_class = RandomForestClassifier
            
           
            elif model_name == 'Logistic Regression':
                solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
                
                # Adjust penalty options based on solver
                if solver in ['liblinear']:
                    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
                else:
                    penalty = 'l2'  # lbfgs only supports 'l2'
                
                params = {
                    'C': trial.suggest_float('C', 0.01, 10, log=True),
                    'solver': solver,
                    'penalty': penalty,
                    'max_iter': trial.suggest_int('max_iter', 100, 300)  
                }
                model_class = LogisticRegression
            
            elif model_name == 'LightGBM':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
                }
                model_class = lgb.LGBMClassifier
    
            scores = []
            splitter = StratifiedKFold(n_splits=3, shuffle=True) if self.stratified_split else KFold(n_splits=3, shuffle=True)
    
            for inner_train_idx, inner_val_idx in splitter.split(X_train, y_train):
                inner_X_train, inner_X_val = X_train[inner_train_idx], X_train[inner_val_idx]
                inner_y_train, inner_y_val = y_train[inner_train_idx], y_train[inner_val_idx]
    
                models = []
                for threshold in thresholds:
                    y_train_binary = (inner_y_train >= threshold).astype(int)
                    model = model_class(random_state=self.random_state, **params)
                    model.fit(inner_X_train, y_train_binary)
                    models.append(model)
    
                # Inference using probabilistic logic
                proba_thresholds = [model.predict_proba(inner_X_val)[:, 1] for model in models]  # shape: 4 x N
                proba_thresholds = np.vstack(proba_thresholds)
                n_samples = proba_thresholds.shape[1]
    
                ordinal_probs = np.zeros((n_samples, 5))
                ordinal_probs[:, 0] = 1 - proba_thresholds[0]
                for i in range(1, 4):
                    ordinal_probs[:, i] = proba_thresholds[i - 1] - proba_thresholds[i]
                ordinal_probs[:, 4] = proba_thresholds[3]
    
                y_pred_val = np.argmax(ordinal_probs, axis=1)
    
                acc = accuracy_score(inner_y_val, y_pred_val)
                scores.append(acc)
    
            return np.mean(scores)
    
    
        # Create the Optuna study and optimize
        study = optuna.create_study(study_name=f'ordinal_{model_name}', direction='maximize')
        study.optimize(objective, n_trials=self.trials, n_jobs= self.n_jobs, gc_after_trial = True)
        best_params = study.best_params  # Return the shared set of best hyperparameters
        del study
        gc.collect()
        return best_params
    def train_and_evaluate_models(self, train, test, fold, classification_type):
        """Train and evaluate multiple models with Optuna hyperparameter tuning."""
        models = {
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1)
        }
        X_train, y_train, X_test, y_test  = self.extract_feat_myhc(train, test)


        # Evaluate the model using the tuned parameters
        if classification_type == 'multi':
            for model_name, model in models.items():
                print(f"Tuning hyperparameters for fold {fold}:{classification_type}_{model_name}...")
                best_params = self.optimize_hyperparameters_multi(model_name, X_train, y_train, fold)
                print(f"Best parameters for {model_name} in fold {fold}: {best_params}")

                # Update model with the best parameters
                model.set_params(**best_params)
                self.evaluate_model(model_name, model, X_train, y_train, X_test, y_test, fold, classification_type, best_params)
                
                # Save the results immediately after evaluation
                self.save_results_dynamically(fold, classification_type, model_name, self.result_row)
        elif classification_type == 'ordinal':
           
            for model_name, model in models.items():
                print(f"Tuning hyperparameters for fold {fold}: {classification_type}_{model_name}...")
                best_params = self.optimize_hyperparameters_for_ordinal(model_name, X_train, y_train, fold)
                print(f"Best parameters for {model_name} in fold {fold}: {best_params}")

                # Update model with the best parameters
                model.set_params(**best_params)
           
                self.ordinal_binary_classification_fs_our(model_name, best_params, X_train, y_train, X_test, y_test, fold, classification_type)
                # Save the results immediately after evaluation
                self.save_results_dynamically(fold, classification_type, model_name, self.result_row)

    def save_results_dynamically(self, fold, classification_type, model_name, result_row):
        """Append results dynamically to a CSV file after each evaluation."""
        result_df = pd.DataFrame([result_row])
        # Write header only if the file does not exist yet
        write_header = not os.path.exists(self.dynamic_csv)
        result_df.to_csv(self.dynamic_csv, mode='a', header=write_header, index=False)
        print(f"Results for fold {fold}, {classification_type} - {model_name} saved dynamically.")
    
           
    def load_and_plot_results(self):

        """Compute averages, standard deviations, and plot metrics with confidence intervals for both multi and ordinal classification."""
        # Load the results CSV
        if not os.path.exists(self.dynamic_csv):
            print(f"Dynamic results CSV not found at {self.dynamic_csv}. Please ensure that results are saved before plotting.")
            return
    
        # Load the saved results
        data = pd.read_csv(self.dynamic_csv)
        
        # Define the metrics to average
        metrics_to_average = ['accuracy', 'precision', 'f1', 'balanced_accuracy', 'acceptable_accuracy', 'AUC']
        
        # Compute average results per model and classification type
        avg_results = data.groupby(['model', 'classification_type'])[metrics_to_average].mean().reset_index()
        # Compute standard deviation of results
        std_results = data.groupby(['model', 'classification_type'])[metrics_to_average].std().reset_index()
        # Merge for easy plotting
        merged_results = avg_results.merge(std_results, on=['model', 'classification_type'], suffixes=('_mean', '_std'))
        num_folds = len(data['fold'].unique())  # Number of outer folds
        
        # Calculate 95% confidence intervals
        ci_factor = 1.96  # For 95% confidence intervals
        for metric in metrics_to_average:
            merged_results[f'{metric}_ci'] = ci_factor * (merged_results[f'{metric}_std'] / np.sqrt(num_folds))
    
    
        # Plotting logic remains the same (including rotation and offset adjustments)
        sns.set(style="whitegrid", palette="Blues", font_scale=1.1)  # A simpler color palette for academic style
        classifiers = merged_results['model'].unique()
        for classifier in classifiers:
            classifier_data = merged_results[merged_results['model'] == classifier]
    
            plt.figure(figsize=(10, 5))
            bar_width = 0.35
            x = range(len(metrics_to_average))
            
            # Plot multi-class metrics with error bars
            multi_class_metrics = classifier_data[classifier_data['classification_type'] == 'multi']
            if not multi_class_metrics.empty:
                multi_means = multi_class_metrics[[f'{metric}_mean' for metric in metrics_to_average]].values.flatten()
                multi_se = multi_class_metrics[[f'{metric}_ci' for metric in metrics_to_average]].values.flatten()
                plt.bar(x, multi_means, width=bar_width, label='Multi', align='center', yerr=multi_se, capsize=5, alpha=0.7)
            
            # Plot ordinal metrics with error bars next to multi-class bars
            ordinal_metrics = classifier_data[classifier_data['classification_type'] == 'ordinal']
            if not ordinal_metrics.empty:
                ordinal_means = ordinal_metrics[[f'{metric}_mean' for metric in metrics_to_average]].values.flatten()
                ordinal_se = ordinal_metrics[[f'{metric}_ci' for metric in metrics_to_average]].values.flatten()
                plt.bar([pos + bar_width for pos in x], ordinal_means, width=bar_width, label='Ordinal', align='center',
                        yerr=ordinal_se, capsize=5, alpha=0.7)
            
            # Adjust labels for better readability
            x_labels = ['accuracy', 'precision', 'f1', 'balanced\naccuracy', 'acceptable\naccuracy', 'AUC']
            plt.xticks([pos + bar_width / 2 for pos in x], x_labels)
            plt.title(f'Performance Metrics for {classifier}', fontsize=16, fontweight='bold', pad=20)
            
            # Adjust the y-axis limits to ensure numbers fit comfortably within the boundary
            max_y_value = max(multi_means.max() + multi_se.max(), ordinal_means.max() + ordinal_se.max()) if not ordinal_metrics.empty else multi_means.max() + multi_se.max()
            plt.ylim(0, max_y_value + 0.15)  # Add extra space above the highest bar
    
            plt.xlabel('Metrics', fontsize=12)
            plt.ylabel('Average Score', fontsize=12)
            plt.legend(title='Classification Type', fontsize=10)
            plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
            
            # Annotate with means and confidence intervals with an angle
            for pos, metric in enumerate(metrics_to_average):
                if not multi_class_metrics.empty:
                    plt.text(x=pos, y=multi_means[pos] + 0.02,
                             s=f"{multi_means[pos]:.2f} ± {multi_se[pos]:.2f}", 
                             ha='center', va='bottom', fontsize=8, color='black', rotation=45)
                if not ordinal_metrics.empty:
                    plt.text(x=pos + bar_width, y=ordinal_means[pos] + 0.02,
                             s=f"{ordinal_means[pos]:.2f} ± {ordinal_se[pos]:.2f}", 
                             ha='center', va='bottom', fontsize=8, color='black', rotation=45)
            
            # Tight layout for cleaner visualization
            plt.tight_layout()
    
            # Save the plot to a file
            plot_save_path = os.path.join(self.config['save_path'], f'{classifier}_performance_metrics.png')
            plt.savefig(plot_save_path, dpi=300)
            print(f"Plot saved to: {plot_save_path}")
    
            # Show the plot
            plt.show()
    
       
if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    CONFIG = {
        'id2vid': os.path.join(project_root, 'data/raw/id2vid.csv'),
        'ids': os.path.join(project_root, 'data/raw/patient_id_all.csv'),
        'vid2score': os.path.join(project_root, 'data/raw/segmented_ft_vid2score.csv'),
        'save_path': os.path.join(project_root, 'data/processed'),
        'n_splits': 5,  
    }

    os.makedirs(CONFIG['save_path'], exist_ok=True)

    trainer = ModelTrainer(CONFIG)
    trainer.run_cross_validation()
    trainer.load_and_plot_results()
