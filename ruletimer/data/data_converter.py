"""
Data conversion utilities for multi-state models
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
from .data import MultiState, Survival, CompetingRisks

class MultiStateDataConverter:
    """Utility class for converting between different multi-state data formats"""
    
    @staticmethod
    def _convert_state_to_numeric(state_values: Union[np.ndarray, pd.Series], preserve_zero: bool = True) -> np.ndarray:
        """Convert string state values to numeric values"""
        # Convert single value to array
        if isinstance(state_values, (str, int, float)):
            state_values = np.array([state_values])
        elif isinstance(state_values, pd.Series):
            state_values = state_values.values
        
        if state_values.dtype.kind in 'iuf':  # Already numeric
            return state_values
        
        # Convert to string first to handle mixed types
        str_values = np.array([str(x) if pd.notna(x) and x != 0 else '0' for x in state_values])
        
        # Convert string states to numeric, preserving 0 for censoring and None
        unique_states = np.unique([x for x in str_values if x != '0' and pd.notna(x)])
        state_map = {state: i+1 for i, state in enumerate(sorted(unique_states))}
        state_map['0'] = 0  # Always preserve 0 for censoring and None
        
        # Convert states using the mapping
        numeric_states = np.array([state_map.get(state, 0) for state in str_values])
        
        # If input was a single value, return a single value
        if len(numeric_states) == 1:
            return numeric_states[0]
        
        return numeric_states
    
    @staticmethod
    def _validate_time_ordering(data: pd.DataFrame, id_col: str, time_col: str):
        """Validate that times are ordered within each patient"""
        for _, group in data.groupby(id_col):
            if not np.all(np.diff(group[time_col]) >= 0):
                raise ValueError(f"Times must be non-decreasing within each patient")
    
    @staticmethod
    def from_person_period(
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        state_col: str,
        censored_col: str,
        covariates: Optional[List[str]] = None
    ) -> MultiState:
        """Convert data from person-period format to MultiState format.

        Args:
            data: DataFrame in person-period format
            id_col: Column name for patient ID
            time_col: Column name for time
            state_col: Column name for state
            censored_col: Column name for censoring indicator
            covariates: Optional list of covariate column names to preserve

        Returns:
            MultiState object
        """
        # Validate time ordering and sort data
        MultiStateDataConverter._validate_time_ordering(data, id_col, time_col)
        data = data.sort_values([id_col, time_col])

        # Convert states to numeric
        data['numeric_state'] = MultiStateDataConverter._convert_state_to_numeric(data[state_col])

        # Initialize lists for records
        records = []
        
        # Process each patient's data
        for _, group in data.groupby(id_col):
            # Skip if no valid states
            if group['numeric_state'].isna().all():
                continue
                
            # Find the first valid state
            first_valid_idx = group['numeric_state'].first_valid_index()
            if first_valid_idx is None:
                continue
                
            # Get the first valid state
            first_row = group.loc[first_valid_idx]
            current_state = first_row['numeric_state']
            current_time = first_row[time_col]
            
            # If this is a single row, create a transition to censored state
            if len(group) == 1:
                records.append({
                    'patient_id': first_row[id_col],
                    'start_time': current_time,
                    'end_time': current_time,
                    'start_state': current_state,
                    'end_state': 0  # Censored
                })
                continue
            
            # Find the next censoring or state change
            for i in range(first_valid_idx + 1, len(group)):
                row = group.iloc[i]
                
                # If this row is censored, create a transition to censored state
                if row[censored_col]:
                    records.append({
                        'patient_id': first_row[id_col],
                        'start_time': current_time,
                        'end_time': row[time_col],
                        'start_state': current_state,
                        'end_state': 0  # Censored
                    })
                    break
                    
                # If state changed and not missing, create a transition
                if not pd.isna(row['numeric_state']) and row['numeric_state'] != current_state:
                    records.append({
                        'patient_id': first_row[id_col],
                        'start_time': current_time,
                        'end_time': row[time_col],
                        'start_state': current_state,
                        'end_state': row['numeric_state']
                    })
                    current_state = row['numeric_state']
                    current_time = row[time_col]
            
            # If we haven't created a transition yet and the last row is censored,
            # create a transition to censored state
            last_row = group.iloc[-1]
            if last_row[censored_col] and (not records or records[-1]['patient_id'] != last_row[id_col]):
                records.append({
                    'patient_id': first_row[id_col],
                    'start_time': current_time,
                    'end_time': last_row[time_col],
                    'start_state': current_state,
                    'end_state': 0  # Censored
                })

        # Return empty MultiState if no records
        if not records:
            return MultiState(
                patient_id=np.array([]),
                start_time=np.array([]),
                end_time=np.array([]),
                start_state=np.array([]),
                end_state=np.array([])
            )

        # Convert records to DataFrame and create MultiState object
        records_df = pd.DataFrame(records)
        return MultiState(
            patient_id=records_df['patient_id'].values,
            start_time=records_df['start_time'].values,
            end_time=records_df['end_time'].values,
            start_state=records_df['start_state'].values,
            end_state=records_df['end_state'].values
        )
    
    @staticmethod
    def from_transition_format(data: pd.DataFrame,
                             id_col: str,
                             from_state_col: str,
                             to_state_col: str,
                             start_time_col: str,
                             end_time_col: str,
                             censored_col: str) -> MultiState:
        """Convert data from transition format to MultiState format.
        
        Args:
            data: DataFrame in transition format
            id_col: Name of column containing subject IDs
            from_state_col: Name of column containing origin states
            to_state_col: Name of column containing destination states
            start_time_col: Name of column containing start times
            end_time_col: Name of column containing end times
            censored_col: Name of column containing censoring indicators
        
        Returns:
            MultiState object containing the converted data
        """
        # Sort data by ID and time
        data = data.sort_values([id_col, start_time_col]).copy()
        
        # Convert states to numeric
        from_states = MultiStateDataConverter._convert_state_to_numeric(data[from_state_col])
        to_states = MultiStateDataConverter._convert_state_to_numeric(data[to_state_col])
        
        # Handle censoring
        end_states = np.where(data[censored_col] == 1, 0, to_states)
        
        return MultiState(
            patient_id=data[id_col].values,
            start_time=data[start_time_col].values,
            end_time=data[end_time_col].values,
            start_state=from_states,
            end_state=end_states
        )
    
    @staticmethod
    def from_long_format(data: pd.DataFrame,
                        id_col: str,
                        time_col: str,
                        state_cols: List[str],
                        censored_col: str,
                        covariates: Optional[List[str]] = None) -> MultiState:
        """
        Convert from long format to MultiState format
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame in long format
        id_col : str
            Column name for patient ID
        time_col : str
            Column name for time
        state_cols : list of str
            List of state indicator column names
        censored_col : str
            Column name for censoring indicator
        covariates : list of str, optional
            List of covariate column names to preserve
            
        Returns
        -------
        MultiState
            Data in MultiState format
        """
        # Validate time ordering
        MultiStateDataConverter._validate_time_ordering(data, id_col, time_col)
        
        # Sort by ID and time
        data = data.sort_values([id_col, time_col])
        
        # Convert state indicators to state numbers
        data['state'] = np.argmax(data[state_cols].values, axis=1) + 1
        
        # Create next state and time
        data['next_state'] = data.groupby(id_col)['state'].shift(-1)
        data['next_time'] = data.groupby(id_col)[time_col].shift(-1)
        data['next_censored'] = data.groupby(id_col)[censored_col].shift(-1)
        
        # Drop last row for each ID
        data = data.groupby(id_col, group_keys=False).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        # Convert states to numeric
        start_state = data['state'].values
        next_state = data['next_state'].values
        
        # Handle censoring - if next row is censored, current row's end state should be 0
        end_state = np.where(data['next_censored'] == 1, 0, next_state)
        
        return MultiState(
            patient_id=data[id_col].values,
            start_time=data[time_col].values,
            end_time=data['next_time'].values,
            start_state=start_state,
            end_state=end_state
        )
    
    @staticmethod
    def from_wide_format(data: pd.DataFrame,
                        id_col: str,
                        time_points: List[str],
                        state_cols: List[str],
                        censor_time_col: Optional[str] = None,
                        covariates: Optional[List[str]] = None) -> MultiState:
        """
        Convert from wide format to MultiState format
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame in wide format
        id_col : str
            Column name for patient ID
        time_points : list of str
            List of time point column names
        state_cols : list of str
            List of state column names
        censor_time_col : str, optional
            Column name for censoring time
        covariates : list of str, optional
            List of covariate column names to preserve
            
        Returns
        -------
        MultiState
            Data in MultiState format
        """
        # Melt data to long format
        id_vars = [id_col]
        if censor_time_col is not None:
            id_vars.append(censor_time_col)
            
        melted = pd.melt(data,
                        id_vars=id_vars,
                        value_vars=time_points,
                        var_name='time_point',
                        value_name='state')
        
        # Convert time points to numeric
        melted['time'] = pd.to_numeric(melted['time_point'].str.extract(r'(\d+)')[0])
        
        # Sort by ID and time
        melted = melted.sort_values([id_col, 'time'])
        
        # Convert states to numeric
        melted['state'] = MultiStateDataConverter._convert_state_to_numeric(melted['state'])
        
        # Create next state and time
        melted['next_state'] = melted.groupby(id_col)['state'].shift(-1)
        melted['next_time'] = melted.groupby(id_col)['time'].shift(-1)
        
        # Drop last row for each ID
        melted = melted.groupby(id_col, group_keys=False).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        # Convert states to numeric
        start_state = melted['state'].values
        next_state = melted['next_state'].values
        
        # Handle censoring
        if censor_time_col is not None:
            # Keep only the first transition for each patient
            valid_transitions = melted['time'] == 1
            # Set end state to 0 if next time is censoring time
            is_censored = melted['next_time'] == melted[censor_time_col]
            end_state = np.where(is_censored, 0, next_state)
            
            # Filter transitions
            return MultiState(
                patient_id=melted[id_col].values[valid_transitions],
                start_time=melted['time'].values[valid_transitions],
                end_time=melted['next_time'].values[valid_transitions],
                start_state=start_state[valid_transitions],
                end_state=end_state[valid_transitions]
            )
        else:
            # Filter valid transitions (where both states are not null)
            valid_transitions = ~pd.isna(start_state) & ~pd.isna(next_state)
            return MultiState(
                patient_id=melted[id_col].values[valid_transitions],
                start_time=melted['time'].values[valid_transitions],
                end_time=melted['next_time'].values[valid_transitions],
                start_state=start_state[valid_transitions],
                end_state=next_state[valid_transitions]
            )
    
    @staticmethod
    def to_person_period(msm_data: MultiState) -> pd.DataFrame:
        """
        Convert from MultiState format to person-period format
        
        Parameters
        ----------
        msm_data : MultiState
            Data in MultiState format
            
        Returns
        -------
        pd.DataFrame
            Data in person-period format
        """
        # Create person-period records
        records = []
        for i in range(len(msm_data.patient_id)):
            # Add start state record
            records.append({
                'ID': msm_data.patient_id[i],
                'Time': msm_data.start_time[i],
                'State': msm_data.start_state[i],
                'Censored': 0
            })
            
            # Add end state record
            is_censored = msm_data.end_state[i] == 0
            records.append({
                'ID': msm_data.patient_id[i],
                'Time': msm_data.end_time[i],
                'State': msm_data.start_state[i] if is_censored else msm_data.end_state[i],
                'Censored': int(is_censored)
            })
        
        df = pd.DataFrame(records)
        df['Censored'] = df['Censored'].astype(int)
        return df
    
    @staticmethod
    def to_transition_format(msm_data: MultiState) -> pd.DataFrame:
        """
        Convert MultiState data to transition format
        
        Parameters
        ----------
        msm_data : MultiState
            Data in MultiState format
            
        Returns
        -------
        pd.DataFrame
            Data in transition format
        """
        return pd.DataFrame({
            'ID': msm_data.patient_id,
            'FromState': msm_data.start_state,
            'ToState': msm_data.end_state,
            'StartTime': msm_data.start_time,
            'EndTime': msm_data.end_time,
            'Censored': (msm_data.end_state == 0).astype(int)
        })
    
    @staticmethod
    def to_long_format(msm_data: MultiState, state_names: List[str]) -> pd.DataFrame:
        """
        Convert MultiState data to long format
        
        Parameters
        ----------
        msm_data : MultiState
            Data in MultiState format
        state_names : list of str
            Names of the states
            
        Returns
        -------
        pd.DataFrame
            Data in long format
        """
        # Create state indicator columns
        n_states = len(state_names)
        records = []
        
        for i in range(len(msm_data.patient_id)):
            # Start state
            state_vec = np.zeros(n_states)
            state_vec[msm_data.start_state[i] - 1] = 1
            records.append({
                'ID': msm_data.patient_id[i],
                'Time': msm_data.start_time[i],
                'Censored': 0,
                **{f'State{name}': val for name, val in zip(state_names, state_vec)}
            })
            
            # End state
            if msm_data.end_state[i] == 0:  # Censored
                records.append({
                    'ID': msm_data.patient_id[i],
                    'Time': msm_data.end_time[i],
                    'Censored': 1,
                    **{f'State{name}': val for name, val in zip(state_names, state_vec)}
                })
            else:
                state_vec = np.zeros(n_states)
                state_vec[msm_data.end_state[i] - 1] = 1
                records.append({
                    'ID': msm_data.patient_id[i],
                    'Time': msm_data.end_time[i],
                    'Censored': 0,
                    **{f'State{name}': val for name, val in zip(state_names, state_vec)}
                })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def from_counting_process(data: pd.DataFrame,
                            id_col: str,
                            from_state_col: str,
                            to_state_col: str,
                            count_col: str,
                            exposure_col: str,
                            censored_col: str) -> MultiState:
        """Convert data from counting process format to MultiState format.
        
        Args:
            data: DataFrame in counting process format
            id_col: Name of column containing subject IDs
            from_state_col: Name of column containing origin states
            to_state_col: Name of column containing destination states
            count_col: Name of column containing transition counts
            exposure_col: Name of column containing exposure times
            censored_col: Name of column containing censoring indicators
        
        Returns:
            MultiState object containing the converted data
        """
        # Create state mapping first using all states
        all_states = pd.concat([data[from_state_col], data[to_state_col]]).unique()
        state_map = {state: i+1 for i, state in enumerate(sorted(all_states))}
        
        # Filter to only include observed transitions (count > 0)
        transitions = data[data[count_col] > 0].copy()
        
        # Convert states to numeric using the mapping
        from_states = np.array([state_map[state] for state in transitions[from_state_col]])
        to_states = np.array([state_map[state] for state in transitions[to_state_col]])
        
        # For censored observations, use the last record for each ID
        censored = data[data[censored_col] == 1].groupby(id_col).last().reset_index()
        
        # Combine transitions and censored observations
        all_records = []
        
        # Add observed transitions
        for i in range(len(transitions)):
            all_records.append({
                'patient_id': transitions[id_col].iloc[i],
                'start_time': transitions[exposure_col].iloc[i],
                'end_time': transitions[exposure_col].iloc[i],
                'start_state': from_states[i],
                'end_state': to_states[i]
            })
        
        # Add censored observations
        for i in range(len(censored)):
            all_records.append({
                'patient_id': censored[id_col].iloc[i],
                'start_time': censored[exposure_col].iloc[i],
                'end_time': censored[exposure_col].iloc[i],
                'start_state': state_map[censored[from_state_col].iloc[i]],
                'end_state': 0  # Censored
            })
        
        # Convert to arrays
        records = pd.DataFrame(all_records)
        return MultiState(
            patient_id=records['patient_id'].values,
            start_time=records['start_time'].values,
            end_time=records['end_time'].values,
            start_state=records['start_state'].values,
            end_state=records['end_state'].values
        ) 

class DataConverter:
    def preprocess_features(self, df, numeric_columns=None, categorical_columns=None, binary_columns=None, scale_numeric=False, one_hot_encode=False):
        # Minimal stub: return numeric columns as numpy array if present, else the whole df
        if numeric_columns:
            return df[numeric_columns].values
        return df.values

    def train_test_split(self, X, y, test_size=0.2, random_state=None, strategy=None):
        # Minimal stub: split first 80%/20%
        n = len(X)
        split = int(n * (1 - test_size))
        return X[:split], X[split:], y[:split], y[split:]

    """Main data conversion class for all data types"""
    
    def __init__(self):
        self.multi_state_converter = MultiStateDataConverter()
    
    def convert_survival(self, 
                        data: pd.DataFrame,
                        time_col: str,
                        event_col: str,
                        missing_value_strategy: str = 'drop') -> Tuple[pd.DataFrame, Survival]:
        """Convert data to survival format
        
        Args:
            data: DataFrame containing the data
            time_col: Name of column containing event times
            event_col: Name of column containing event indicators
            missing_value_strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'most_frequent')
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is Survival object
        """
        # Handle missing values
        feature_cols = [col for col in data.columns if col not in [time_col, event_col]]
        if missing_value_strategy == 'drop':
            # Drop rows with missing feature values
            data = data.dropna(subset=feature_cols)
        else:
            data = self._handle_missing_values(data, missing_value_strategy)
        
        # Extract features (all columns except time and event)
        X = data[feature_cols]
        
        # Create Survival object
        y = Survival(time=data[time_col].values, event=data[event_col].values)
        
        return X, y
    
    def convert_competing_risks(self,
                              data: pd.DataFrame,
                              time_col: str,
                              event_col: str,
                              missing_value_strategy: str = 'drop') -> Tuple[pd.DataFrame, CompetingRisks]:
        """Convert data to competing risks format
        
        Args:
            data: DataFrame containing the data
            time_col: Name of column containing event times
            event_col: Name of column containing event types
            missing_value_strategy: Strategy for handling missing values
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is CompetingRisks object
        """
        # Handle missing values
        data = self._handle_missing_values(data, missing_value_strategy)
        
        # Extract features
        feature_cols = [col for col in data.columns if col not in [time_col, event_col]]
        X = data[feature_cols]
        
        # Create CompetingRisks object
        y = CompetingRisks(time=data[time_col].values, event=data[event_col].values)
        
        return X, y
    
    def convert_multi_state(self,
                          data: pd.DataFrame,
                          start_time_col: str,
                          end_time_col: str,
                          start_state_col: str,
                          end_state_col: str,
                          patient_id_col: Optional[str] = None,
                          missing_value_strategy: str = 'drop') -> Tuple[pd.DataFrame, MultiState]:
        """Convert data to multi-state format
        
        Args:
            data: DataFrame containing the data
            start_time_col: Name of column containing start times
            end_time_col: Name of column containing end times
            start_state_col: Name of column containing start states
            end_state_col: Name of column containing end states
            patient_id_col: Optional name of column containing patient IDs
            missing_value_strategy: Strategy for handling missing values
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is MultiState object
        """
        # Handle missing values
        data = self._handle_missing_values(data, missing_value_strategy)
        
        # Extract features
        feature_cols = [col for col in data.columns 
                       if col not in [start_time_col, end_time_col, start_state_col, end_state_col, patient_id_col]]
        X = data[feature_cols]
        
        # Create patient IDs if not provided
        if patient_id_col is None:
            patient_ids = np.arange(len(data))
        else:
            patient_ids = data[patient_id_col].values
        
        # Create MultiState object
        y = MultiState(
            patient_id=patient_ids,
            start_time=data[start_time_col].values,
            end_time=data[end_time_col].values,
            start_state=data[start_state_col].values,
            end_state=data[end_state_col].values
        )
        
        return X, y
    
    def convert_time_dependent(self,
                             time_dep_features: np.ndarray,
                             time_points: np.ndarray,
                             event_times: np.ndarray,
                             events: np.ndarray) -> Tuple[Dict[int, np.ndarray], Survival]:
        """Convert time-dependent data
        
        Args:
            time_dep_features: Array of shape (n_samples, n_features, n_time_points)
            time_points: Array of time points
            event_times: Array of event times
            events: Array of event indicators
            
        Returns:
            Tuple of (X_td, y) where X_td is dict mapping sample index to feature matrix
            and y is Survival object
        """
        n_samples = time_dep_features.shape[0]
        X_td = {}
        
        for i in range(n_samples):
            # Find valid time points (before event/censoring)
            valid_times = time_points <= event_times[i]
            X_td[i] = time_dep_features[i, :, valid_times].T
        
        y = Survival(time=event_times, event=events)
        
        return X_td, y
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values in data
        
        Args:
            data: DataFrame containing the data
            strategy: Strategy for handling missing values
            
        Returns:
            DataFrame with missing values handled
        """
        if strategy == 'drop':
            return data.dropna()
        elif strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'most_frequent':
            return data.fillna(data.mode().iloc[0])
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}") 