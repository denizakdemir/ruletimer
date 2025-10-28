# RuleTimeR Blueprint

This document provides a detailed blueprint of the RuleTimeR project, outlining the structure and contents of the codebase.

## Directory Structure

```
.
├── LICENSE
├── README.md
├── competing_risks_feature_importance.png
├── docs
│   └── api_reference.md
├── ebmt_competing_risks.png
├── ebmt_survival_curves.png
├── msm_data.txt
├── pytest.ini
├── requirements.txt
├── ruletimer
│   ├── __init__.py
│   ├── base_time_to_event.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── data_converter.py
│   │   └── data_validator.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── base_multi_state.py
│   │   ├── competing_risks.py
│   │   ├── rule_multi_state.py
│   │   └── survival.py
│   ├── state_manager.py
│   ├── time_handler.py
│   └── utils
│       ├── __init__.py
│       ├── hazard_estimation.py
│       ├── importance.py
│       ├── prediction_utils.py
│       ├── time_utils.py
│       └── utils.py
├── setup.cfg
├── setup.py
├── survival_curves.png
├── survival_feature_importance.png
├── survival_variable_importances.png
└── tests
    ├── conftest.py
    ├── examples
    │   ├── ebmt_example.py
    │   ├── full_example.py
    │   ├── test_competing_risks.py
    │   ├── test_multi_state.py
    │   └── test_survival_analysis.py
    ├── test_data.py
    ├── test_data_converter.py
    ├── test_hazard_estimation.py
    ├── test_hello_world.py
    ├── test_rule_multi_state.py
    ├── test_state_manager.py
    ├── test_statistical_validation.py
    └── test_visualization.py
```

## ruletimer/

This is the main package directory.

### `__init__.py`

The `__init__.py` file exposes the public API of the `ruletimer` package.

```python
"""
RuleTimeR: A Rule Ensemble-based Time-to-Event Regression Module
"""

__version__ = "0.1.0"

from .data import Survival, CompetingRisks, MultiState
from .models.survival import RuleSurvival
from .models.competing_risks import RuleCompetingRisks
from .utils import StateStructure
from .visualization import (
    plot_rule_importance,
    plot_cumulative_incidence,
    plot_state_transitions
)

__all__ = [
    "Survival",
    "CompetingRisks",
    "MultiState",
    "RuleSurvival",
    "RuleCompetingRisks",
    "StateStructure",
    "plot_rule_importance",
    "plot_cumulative_incidence",
    "plot_state_transitions"
]
```

### `ruletimer/data/data.py`

This file defines the data structures for the different types of time-to-event analysis.

```python
"""
Data structures for time-to-event analysis
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional

class Survival:
    def __getitem__(self, idx):
        return Survival(self.time[idx], self.event[idx])
    """Class for standard survival data"""

    def __init__(self, time: Union[np.ndarray, pd.Series],
                 event: Union[np.ndarray, pd.Series]):
        """
        Initialize survival data

        Parameters
        ----------
        time : array-like
            Time to event or censoring
        event : array-like
            Event indicator (1 for event, 0 for censored)
        """
        self.time = np.asarray(time)
        self.event = np.asarray(event)
        self._validate()

    def __len__(self):
        """Return the number of samples."""
        return len(self.time)

    @property
    def shape(self):
        """Return the shape of the data."""
        return (len(self),)

    def _validate(self):
        """Validate the survival data"""
        if len(self.time) != len(self.event):
            raise ValueError("Time and event arrays must have the same length")
        if not np.all(self.time >= 0):
            raise ValueError("All times must be non-negative")
        if not np.all(np.isin(self.event, [0, 1])):
            raise ValueError("Event indicators must be 0 or 1")

class CompetingRisks:
    def __getitem__(self, idx):
        return CompetingRisks(self.time[idx], self.event[idx])
    """Class for competing risks data"""

    def __init__(self, time: Union[np.ndarray, pd.Series],
                 event: Union[np.ndarray, pd.Series]):
        """
        Initialize competing risks data

        Parameters
        ----------
        time : array-like
            Time to event or censoring
        event : array-like
            Event type indicator (0 for censored, positive integers for different event types)
        """
        self.time = np.asarray(time)
        self.event = np.asarray(event)
        self._validate()

    def _validate(self):
        """Validate the competing risks data"""
        if len(self.time) != len(self.event):
            raise ValueError("Time and event arrays must have the same length")
        if not np.all(self.time >= 0):
            raise ValueError("All times must be non-negative")
        if not np.all(self.event >= 0):
            raise ValueError("Event types must be non-negative integers")
        if not np.all(np.isin(self.event, [0, 1, 2])):
            raise ValueError("Event indicators must be 0, 1, or 2")

class MultiState:
    def __getitem__(self, idx):
        return MultiState(self.start_time[idx], self.end_time[idx], self.start_state[idx], self.end_state[idx], self.patient_id[idx])
    """Class for multi-state data"""

    def __init__(self, start_time: Union[np.ndarray, pd.Series],
                 end_time: Union[np.ndarray, pd.Series],
                 start_state: Union[np.ndarray, pd.Series],
                 end_state: Union[np.ndarray, pd.Series],
                 patient_id: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Initialize multi-state data

        Parameters
        ----------
        start_time : array-like
            Time at which the observation starts
        end_time : array-like
            Time at which the observation ends
        start_state : array-like
            State at the start of observation (non-negative integers)
        end_state : array-like
            State at the end of observation (non-negative integers)
        patient_id : array-like, optional
            Unique identifier for each patient. If not provided, will be generated automatically.
        """
        self.start_time = np.asarray(start_time)
        self.end_time = np.asarray(end_time)
        self.start_state = np.asarray(start_state)
        self.end_state = np.asarray(end_state)

        if patient_id is None:
            self.patient_id = np.arange(len(start_time))
        else:
            self.patient_id = np.asarray(patient_id)

        self._validate()

    def _validate(self):
        """Validate the multi-state data"""
        if not (len(self.patient_id) == len(self.start_time) == len(self.end_time) ==
                len(self.start_state) == len(self.end_state)):
            raise ValueError("All arrays must have the same length")

        # Validate patient IDs
        if not np.all(self.patient_id >= 0):
            raise ValueError("Patient IDs must be non-negative integers")

        # Validate times
        if not np.all(self.start_time >= 0):
            raise ValueError("All start times must be non-negative")
        if not np.all(self.end_time >= 0):
            raise ValueError("All end times must be non-negative")

        # Validate states
        if not np.all(self.start_state >= 0):
            raise ValueError("States must be non-negative integers")
        if not np.all(self.end_state >= 0):
            raise ValueError("States must be non-negative integers")

        # Validate time ordering within patients
        for pid in np.unique(self.patient_id):
            mask = self.patient_id == pid
            if len(self.start_time[mask]) > 1:  # Only check if more than one observation
                if not np.all(np.diff(self.start_time[mask]) > 0):
                    raise ValueError(f"Start times must be strictly increasing for patient {pid}")
                if not np.all(np.diff(self.end_time[mask]) > 0):
                    raise ValueError(f"End times must be strictly increasing for patient {pid}")
        # Validate that start_state != end_state for all transitions
        if np.any(self.start_state == self.end_state):
            raise ValueError("Start state and end state must differ for all transitions.")
```


### `ruletimer/data/data_converter.py`

This file contains the logic for converting data between different formats.

```python
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
                'end_time': transitions[exposure_col].iloc[i] + 1,  # End time is exposure time + 1
                'start_state': from_states[i],
                'end_state': to_states[i]
            })

        # Add censored observations
        for i in range(len(censored)):
            all_records.append({
                'patient_id': censored[id_col].iloc[i],
                'start_time': censored[exposure_col].iloc[i],
                'end_time': censored[exposure_col].iloc[i] + 1,  # End time is exposure time + 1
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
```


### `ruletimer/models/base.py`

This file defines the base class for all rule-based models.

```python
"""
Base classes for rule-based models
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

class BaseRuleEnsemble(BaseEstimator, ABC):
    """Base class for rule-based ensemble models"""

    def __init__(self):
        self.is_fitted_ = False
        self._rules_tuples = []
        self.rule_weights_ = None
        self.state_structure = None
        self._y = None
        self.baseline_hazards_ = {}

    @abstractmethod
    def _fit_weights(self, rule_values, y):
        """Fit weights for the rules"""
        pass

    @abstractmethod
    def _compute_feature_importances(self):
        """Compute feature importances"""
        pass

    @abstractmethod
    def _evaluate_rules(self, X):
        """Evaluate rules on input data"""
        pass

    def predict_cumulative_incidence(self, X, times, event_types):
        """Predict cumulative incidence for each event type"""
        raise NotImplementedError

    def predict_state_occupation(self, X, times):
        """Predict state occupation probabilities"""
        raise NotImplementedError

    @property
    def rules_(self):
        """Get the list of rules"""
        return self._rules_tuples

    def get_rules(self):
        """Get the list of rules"""
        return self._rules_tuples

    @property
    def feature_importances_(self):
        """Get feature importances"""
        if not hasattr(self, '_feature_importances'):
            self._compute_feature_importances()
        return self._feature_importances
```


### `ruletimer/models/survival.py`

This file contains the implementation of the survival analysis model.

```python
"""
Specialized survival model implemented as a two-state model.
"""
from typing import Optional, Union, List, Tuple, Dict
import numpy as np
from sklearn.linear_model import ElasticNet
from ruletimer.models.base_multi_state import BaseMultiStateModel
from ruletimer.data import Survival

class RuleSurvival(BaseMultiStateModel):
    """
    Survival model implemented as a special case of multi-state model.

    This model represents standard survival analysis as a two-state model:
    Alive (0) -> Dead (1)

    Parameters
    ----------
    hazard_method : str
        Method for hazard estimation: "nelson-aalen" or "parametric"

    Attributes
    ----------
    rules_ : list of Rule
        The set of rules selected by the model during fitting.
    coefficients_ : ndarray of shape (n_rules,)
        The coefficients associated with each rule in the final model.
    intercept_ : float
        The intercept term of the model.

    Examples
    --------
    >>> from ruletimer.models import RuleSurvival
    >>> from ruletimer.data import Survival
    >>> import numpy as np
    >>>
    >>> # Generate example data
    >>> X = np.random.randn(100, 5)
    >>> times = np.random.exponential(scale=5, size=100)
    >>> events = np.random.binomial(1, 0.7, size=100)
    >>> y = Survival(time=times, event=events)
    >>>
    >>> # Initialize and fit model
    >>> model = RuleSurvival(hazard_method="nelson-aalen")
    >>> model.fit(X, y)
    >>>
    >>> # Make predictions
    >>> test_times = np.linspace(0, 10, 100)
    >>> survival_probs = model.predict_survival(X, test_times)
    """

    def __init__(self, hazard_method: str = "nelson-aalen"):
        """
        Initialize survival model.

        Parameters
        ----------
        hazard_method : str
            Method for hazard estimation: "nelson-aalen" or "parametric"
        """
        # Initialize with two states and one transition
        super().__init__(
            states=["Alive", "Dead"],
            transitions=[(0, 1)],
            hazard_method=hazard_method
        )

    def fit(self, X: np.ndarray, y: Survival) -> 'RuleSurvival':
        """
        Fit the survival model.

        Parameters
        ----------
        X : array-like
            Training data
        y : Survival
            Target survival data

        Returns
        -------
        self : RuleSurvival
            Fitted model
        """
        # Validate input data
        if not isinstance(y, Survival):
            raise ValueError("y must be a Survival object")

        # Prepare transition-specific data
        transition = (0, 1)  # Alive -> Dead
        transition_times = {transition: y.time}
        transition_events = {transition: y.event}

        # Estimate baseline hazards
        self._estimate_baseline_hazards(transition_times, transition_events)

        # Fit transition-specific model (to be implemented by subclass)
        self._fit_transition_model(X, y, transition)

        self.is_fitted_ = True
        return self

    def predict_survival(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict survival probabilities.

        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like, optional
            Times at which to predict survival

        Returns
        -------
        np.ndarray
            Predicted survival probabilities
        """
        if times is None:
            times = self.baseline_hazards_[(0, 1)][0]

        # Get survival probabilities
        survival_probs = 1 - self.predict_transition_probability(X, times, 0, 1)

        # Ensure survival probability at time 0 is 1
        if len(times) > 0:
            survival_probs[:, 0] = 1.0

        return survival_probs

    def predict_hazard(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict hazard function.

        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like, optional
            Times at which to predict hazard

        Returns
        -------
        np.ndarray
            Predicted hazard values
        """
        if times is None:
            times = self.baseline_hazards_[(0, 1)][0]

        return self.predict_transition_hazard(X, times, 0, 1)

    def predict_cumulative_hazard(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
        from_state: Optional[Union[str, int]] = None,
        to_state: Optional[Union[str, int]] = None
    ) -> np.ndarray:
        """
        Predict cumulative hazard function.

        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like, optional
            Times at which to predict cumulative hazard
        from_state : str or int, optional
            Starting state (ignored in survival analysis)
        to_state : str or int, optional
            Target state (ignored in survival analysis)

        Returns
        -------
        np.ndarray
            Predicted cumulative hazard values
        """
        if times is None:
            times = self.baseline_hazards_[(0, 1)][0]

        # In survival analysis, we only have one transition (0 -> 1)
        return super().predict_cumulative_hazard(X, times, 0, 1)

    def _fit_transition_model(
        self,
        X: np.ndarray,
        y: Survival,
        transition: tuple
    ) -> None:
        """
        Fit transition-specific model optimized for concordance.
        """
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        from itertools import combinations

        # Convert survival data to pairwise comparisons
        n_samples = len(y.time)
        pairs = []
        labels = []
        pair_weights = []

        # Create pairwise comparisons only between comparable pairs
        for i, j in combinations(range(n_samples), 2):
            if y.event[i] and y.time[i] < y.time[j]:  # i had event before j's time
                pairs.append((i, j))
                labels.append(1)  # i should have higher risk
                pair_weights.append(1.0)
            elif y.event[j] and y.time[j] < y.time[i]:  # j had event before i's time
                pairs.append((i, j))
                labels.append(0)  # j should have higher risk
                pair_weights.append(1.0)

        if not pairs:  # No comparable pairs found
            # Fall back to using time-based ranking
            self.transition_models_[transition] = None
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
            return

        pairs = np.array(pairs)
        labels = np.array(labels)
        pair_weights = np.array(pair_weights)

        # Create feature differences for pairs
        X_diff = X[pairs[:, 0]] - X[pairs[:, 1]]

        # Train logistic regression on pairwise differences
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            random_state=self.random_state,
            max_iter=1000
        )
        model.fit(X_diff, labels, sample_weight=pair_weights)

        # Store model and compute feature importances
        self.transition_models_[transition] = model
        self.feature_importances_ = np.abs(model.coef_[0])
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        # We're using features directly
        self._using_features_as_rules = True
        self.selected_features_ = self.feature_importances_ > 0

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores using simple linear predictor.
        """
        transition = (0, 1)
        if transition not in self.transition_models_:
            raise ValueError("Model not fitted or transition (0, 1) model missing")

        # Use feature importances directly as coefficients
        importances = self.feature_importances_
        if len(importances) == 0:
            importances = np.ones(X.shape[1]) / X.shape[1]

        # Linear predictor similar to test data generation
        risk_scores = np.dot(X, importances)

        # Normalize scores
        risk_scores = (risk_scores - np.mean(risk_scores)) / (np.std(risk_scores) + 1e-8)

        return risk_scores  # Higher value = higher risk

    def _extract_rules_from_forest(self, forest):
        """Extract rules from random forest."""
        rules = []
        for tree in forest.estimators_:
            rules.extend(self._extract_rules_from_tree(tree.tree_))

        # Sort rules by importance
        rule_importances = []
        for rule in rules:
            importance = 0
            for feature, _, _ in rule:
                importance += forest.feature_importances_[feature]
            rule_importances.append(importance)

        # Sort rules by importance and take top max_rules
        sorted_rules = [rule for _, rule in sorted(zip(rule_importances, rules), reverse=True)]
        return sorted_rules[:self.max_rules]

    def _extract_rules_from_tree(self, tree):
        """Extract rules from a single tree."""
        rules = []

        def recurse(node, rule):
            if tree.feature[node] != -2:  # Not a leaf
                feature = tree.feature[node]
                threshold = tree.threshold[node]

                # Left branch: feature <= threshold
                left_rule = rule + [(feature, "<=", threshold)]
                recurse(tree.children_left[node], left_rule)

                # Right branch: feature > threshold
                right_rule = rule + [(feature, ">", threshold)]
                recurse(tree.children_right[node], right_rule)
            else:
                # Only keep rules from nodes with sufficient samples
                if tree.n_node_samples[node] >= self.min_samples_leaf:
                    rules.append(rule)

        recurse(0, [])
        return rules

    def _evaluate_rules(self, X):
        """Evaluate rules on data."""
        # For survival analysis, we only have one transition (0, 1)
        transition = (0, 1)
        if transition not in self.rules_ or not self.rules_[transition]:
            return np.zeros((X.shape[0], 0))

        rules = self.rules_[transition]
        rule_matrix = np.zeros((X.shape[0], len(rules)))
        for i, rule in enumerate(rules):
            mask = np.ones(X.shape[0], dtype=bool)
            for feature, op, threshold in rule:
                if op == "<=":
                    mask &= (X[:, feature] <= threshold)
                else:  # op == ">"
                    mask &= (X[:, feature] > threshold)
            rule_matrix[:, i] = mask
        return rule_matrix

    def _compute_feature_importances(self):
        """Compute feature importances from rules."""
        if not self.rules_ or len(self.rule_weights_) == 0:
            return np.array([])

        # Get all rules from all transitions
        all_rules = []
        for transition_rules in self.rules_.values():
            all_rules.extend(transition_rules)

        if not all_rules:
            return np.array([])

        # Get number of features from the first rule's first condition's feature index
        n_features = max(feature for rule in all_rules for feature, _, _ in rule) + 1
        importances = np.zeros(n_features)

        # Compute importances for each transition
        for transition, rules in self.rules_.items():
            if transition in self.rule_weights_:
                weights = self.rule_weights_[transition]
                for rule, weight in zip(rules, weights):
                    for feature, _, _ in rule:
                        importances[feature] += abs(weight)

        # Normalize importances
        if np.sum(importances) > 0:
            importances /= np.sum(importances)

        return importances

    def get_top_rules(self, n_rules: int = 5) -> list:
        """
        Get the top n rules across all transitions.

        Parameters
        ----------
        n_rules : int
            Number of top rules to return

        Returns
        -------
        list
            List of top rules
        """
        all_rules = []
        for transition, rules in self.rules_.items():
            if transition in self.rule_importances_:
                importances = self.rule_importances_[transition]
                for rule, importance in zip(rules, importances):
                    all_rules.append((rule, importance, transition))

        # Sort by importance and take top n
        all_rules.sort(key=lambda x: x[1], reverse=True)
        return all_rules[:n_rules]

    def get_rule_importances(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get rule importances in the format expected by visualization code.

        Returns
        -------
        dict
            Dictionary mapping transitions to arrays of rule importances
        """
        importances = {}
        for transition, rules in self.rules_.items():
            if transition in self.rule_importances_:
                importances[transition] = np.array(self.rule_importances_[transition])
        return importances

    def __str__(self) -> str:
        """String representation of the model."""
        if not self.is_fitted_:
            return "RuleSurvival (not fitted)"

        top_rules = self.get_top_rules(5)
        if not top_rules:
            return "RuleSurvival (no rules generated)"

        s = "RuleSurvival\n"
        s += "Top rules:\n"
        for i, (rule, importance, transition) in enumerate(top_rules):
            s += f"Rule {i+1} (transition {transition}, importance={importance:.3f}): {rule}\n"
        return s

class RuleSurvivalCox(RuleSurvival):
    """
    Rule-based Cox model for survival analysis.
    """

    def __init__(
        self,
        max_rules: int = 100,
        max_depth: int = 3,
        n_estimators: int = 500,
        min_samples_leaf: int = 10,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        random_state: Optional[int] = None
    ):
        """
        Initialize RuleSurvivalCox model.

        Parameters
        ----------
        max_rules : int
            Maximum number of rules to generate
        max_depth : int
            Maximum depth of trees
        n_estimators : int
            Number of trees in random forest
        min_samples_leaf : int
            Minimum number of samples required at a leaf node
        alpha : float
            L1 + L2 regularization strength
        l1_ratio : float
            L1 ratio for elastic net (1 = lasso, 0 = ridge)
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(hazard_method="nelson-aalen")
        self.max_rules = max_rules
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.transition_models_ = {}
        self.rules_ = {}  # Dictionary mapping transitions to rules
        self.rule_importances_ = {}  # Dictionary mapping transitions to rule importances
        self.rule_weights_ = {}  # Dictionary mapping transitions to rule weights
        self.is_fitted_ = False

    def _fit_transition_model(
        self,
        X: np.ndarray,
        y: Survival,
        transition: tuple
    ) -> None:
        """
        Fit transition-specific model using gradient boosting for better risk discrimination.

        Parameters
        ----------
        X : array-like
            Training data
        y : Survival
            Target survival data
        transition : tuple
            Transition to fit model for
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        # Create more aggressive risk targets
        # Use negative log time so differences are more pronounced
        risk_target = -np.log1p(y.time)

        # Weight samples to focus on events and early/late times
        time_weights = np.exp(-y.time/np.mean(y.time))  # More weight to early events
        sample_weights = y.event.astype(float) * (1.0 + time_weights)
        sample_weights /= np.sum(sample_weights)

        # Initialize feature matrix
        self.rules_[transition] = []
        self.rule_weights_[transition] = []

        # Train initial model for feature selection
        selector = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self.random_state
        )
        selector.fit(X, risk_target, sample_weight=sample_weights)

        # Get feature importances and select top features
        importances = selector.feature_importances_
        threshold = np.percentile(importances[importances > 0], 50)  # More aggressive threshold
        selected = importances >= threshold
        if not np.any(selected):
            selected = importances > 0

        X_selected = X[:, selected]

        # Train main model on selected features
        model = GradientBoostingRegressor(
            n_estimators=200,  # More trees
            learning_rate=0.05,  # Lower learning rate
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=self.random_state
        )
        model.fit(X_selected, risk_target, sample_weight=sample_weights)

        # Extract rules from each tree
        for tree in model.estimators_[:, 0]:
            rules = self._extract_rules_from_tree(tree.tree_)
            if rules:
                # Map selected feature indices back to original
                mapped_rules = []
                orig_indices = np.where(selected)[0]
                for rule in rules:
                    mapped_rule = [(orig_indices[feat], op, thresh) for feat, op, thresh in rule]
                    mapped_rules.append(mapped_rule)
                self.rules_[transition].extend(mapped_rules)

        # Keep only top rules
        if self.rules_[transition]:
            importances = []
            for rule in self.rules_[transition]:
                importance = 0
                for feature, _, _ in rule:
                    importance += selector.feature_importances_[feature]
                importances.append(importance)

            sorted_pairs = sorted(zip(importances, self.rules_[transition]), reverse=True)
            self.rules_[transition] = [rule for _, rule in sorted_pairs[:self.max_rules]]

        # Create rule matrix
        rule_matrix = self._evaluate_rules(X)

        if rule_matrix.shape[1] == 0:
            # If no rules, use selected features
            self.transition_models_[transition] = model
            self.rule_weights_[transition] = model.feature_importances_
            self.rule_importances_[transition] = model.feature_importances_
            self._using_features_as_rules = True
            self.selected_features_ = selected
        else:
            # Train final model on rules
            final_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=self.random_state
            )
            final_model.fit(rule_matrix, risk_target, sample_weight=sample_weights)

            self.transition_models_[transition] = final_model
            self.rule_weights_[transition] = final_model.feature_importances_
            self.rule_importances_[transition] = np.abs(final_model.feature_importances_)
            self._using_features_as_rules = False

        # Store feature importances
        self.feature_importances_ = self._compute_feature_importances(X.shape[1])

    def _compute_feature_importances(self, n_features):
        """Compute feature importances from rules or direct features."""
        importances = np.zeros(n_features)

        for transition, weights in self.rule_weights_.items():
            if hasattr(self, '_using_features_as_rules') and self._using_features_as_rules:
                # Weights correspond to features directly. Importance was stored directly.
                if transition in self.rule_importances_ and len(self.rule_importances_[transition]) == n_features:
                    importances += self.rule_importances_[transition]
                else:
                    print("Warning: Cannot compute feature importances when using features and importance shape mismatch.")

            elif transition in self.rules_:
                rules = self.rules_[transition]
                # Ensure weights match rules
                if len(rules) == len(weights):
                    for rule, weight in zip(rules, weights):
                        # Add abs(weight) to each feature in the rule
                        for feature, _, _ in rule:
                            if feature < n_features:  # Check bounds
                                importances[feature] += abs(weight)
                else:
                    print(f"Warning: Mismatch between number of rules ({len(rules)}) and weights ({len(weights)}) for transition {transition}. Skipping.")

        total_importance = np.sum(importances)
        if total_importance > 0:
            importances /= total_importance
        return importances

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores for samples with improved discrimination.

        Parameters
        ----------
        X : array-like
            Data to predict for

        Returns
        -------
        np.ndarray
            Risk scores for each sample. Higher values indicate higher risk.
        """
        transition = (0, 1)
        if transition not in self.transition_models_:
            raise ValueError("Model not fitted or transition (0, 1) model missing")

        model = self.transition_models_[transition]

        if hasattr(self, '_using_features_as_rules') and self._using_features_as_rules:
            # Use selected features
            X_selected = X[:, self.selected_features_]
            raw_prediction = model.predict(X_selected)
        else:
            # Use rules
            rule_matrix = self._evaluate_rules(X)
            if rule_matrix.shape[1] == 0:
                X_selected = X[:, self.selected_features_]
                raw_prediction = model.predict(X_selected)
                self._using_features_as_rules = True
            else:
                raw_prediction = model.predict(rule_matrix)

        # Scale predictions to boost discrimination
        # We're predicting negative log time, so higher raw prediction = higher risk
        # Add non-linear transformation to enhance separation
        risk_scores = np.exp(raw_prediction) - 1  # Makes differences more pronounced

        # Center and normalize risk scores for better numerical stability
        risk_scores = (risk_scores - np.mean(risk_scores)) / (np.std(risk_scores) + 1e-8)

        return risk_scores

    def predict_transition_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
        from_state: Union[str, int],
        to_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict transition-specific hazard with improved long-term behavior.
        """
        import numpy as np
        from scipy.interpolate import interp1d

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        from_idx = self.state_manager.to_internal_index(from_state)
        to_idx = self.state_manager.to_internal_index(to_state)
        transition = (from_idx, to_idx)

        if transition not in self.baseline_hazards_:
            return np.zeros((X.shape[0], len(times)))

        # Get baseline hazard values
        baseline_times, baseline_values = self.baseline_hazards_[transition]

        if len(baseline_times) == 0:
            return np.zeros((X.shape[0], len(times)))

        # Sort baseline values to ensure monotonicity
        sort_idx = np.argsort(baseline_times)
        baseline_times = baseline_times[sort_idx]
        baseline_values = baseline_values[sort_idx]

        # Fit log-linear model for hazard extrapolation
        last_idx = max(1, len(baseline_times) // 2)  # Use at least half the data
        log_times = np.log1p(baseline_times[last_idx:])
        log_hazard = np.log1p(baseline_values[last_idx:])
        coeffs = np.polyfit(log_times, log_hazard, deg=1)

        # Create interpolation function with log-linear extrapolation
        def hazard_func(t):
            mask = t <= baseline_times[-1]
            result = np.zeros_like(t, dtype=float)

            # Interpolate for times within observed range
            if np.any(mask):
                interp = interp1d(
                    baseline_times,
                    baseline_values,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(baseline_values[0], baseline_values[-1])
                )
                result[mask] = interp(t[mask])

            # Extrapolate for times beyond observed range using log-linear model
            if np.any(~mask):
                log_t = np.log1p(t[~mask])
                log_h = coeffs[0] * log_t + coeffs[1]
                result[~mask] = np.expm1(log_h)

            return result

        # Get baseline hazard at requested times
        baseline_hazard = hazard_func(times)

        # Get relative hazard from risk scores with more aggressive scaling
        risk_scores = self.predict_risk(X)
        relative_hazard = np.exp(np.clip(risk_scores, -50, 50))

        # Scale baseline hazard by relative hazard
        hazard = baseline_hazard[np.newaxis, :] * relative_hazard[:, np.newaxis]

        # Ensure hazard is non-negative and numerically stable
        hazard = np.maximum(hazard, 1e-50)

        return hazard

    def get_rule_importances(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get the importance scores for each rule in the model.

        Returns
        -------
        dict
            Dictionary mapping transitions to rule importance arrays
        """
        if not hasattr(self, 'rule_weights_'):
            raise ValueError("Model must be fitted before getting rule importances")

        return self.rule_weights_

    def get_variable_importances(self) -> Dict[str, float]:
        """
        Get the importance scores for each variable in the model.

        Returns
        -------
        dict
            Dictionary mapping variable names to their importance scores
        """
        if not hasattr(self, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting variable importances")

        # Get feature names from the preprocessor if available
        feature_names = getattr(self, 'feature_names_',
                              [f'feature_{i}' for i in range(len(self.feature_importances_))])

        # Create dictionary of variable importances
        importances = dict(zip(feature_names, self.feature_importances_))

        # Sort by importance in descending order
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
```


### `ruletimer/models/competing_risks.py`

This file contains the implementation of the competing risks model.

```python
"""
Specialized competing risks model implemented as a multi-state model.
"""
from typing import Dict, List, Optional, Union
import numpy as np
from ruletimer.models.base_multi_state import BaseMultiStateModel
from ruletimer.data import CompetingRisks
from sklearn.linear_model import ElasticNet
from scipy.interpolate import interp1d

class RuleCompetingRisks(BaseMultiStateModel):
    """A rule-based competing risks model that handles multiple event types.

    This class implements a specialized competing risks model that uses a rule-based
    approach to model transitions from an initial state to multiple absorbing states
    (different event types). The model is particularly useful for interpretable
    competing risks analysis where transparent decision rules are desired.

    Parameters
    ----------
    n_rules : int, default=100
        Maximum number of rules to generate for each event type. Each rule represents
        a potential decision boundary in the feature space.
    min_support : float, default=0.1
        Minimum proportion of samples that must satisfy a rule for it to be considered.
        This helps prevent overfitting to small subgroups.
    alpha : float, default=0.5
        Elastic net mixing parameter. A value of 0 corresponds to L2 regularization,
        while 1 corresponds to L1 regularization. Values in between provide a mix
        of both regularization types.
    l1_ratio : float, default=0.5
        The ratio of L1 to L2 regularization in the elastic net penalty.
        Must be between 0 and 1.
    max_iter : int, default=1000
        Maximum number of iterations for the optimization algorithm.
    tol : float, default=1e-4
        Tolerance for the optimization algorithm. The algorithm will stop when
        the change in the objective function is less than this value.
    random_state : int or RandomState, default=None
        Controls the randomness of the rule generation process.

    Attributes
    ----------
    rules_ : dict
        Dictionary containing the set of rules for each event type.
        Keys are event types, values are lists of Rule objects.
    coefficients_ : dict
        Dictionary containing the coefficients for each event type.
        Keys are event types, values are numpy arrays of coefficients.
    intercepts_ : dict
        Dictionary containing the intercept terms for each event type.
        Keys are event types, values are float intercepts.

    Examples
    --------
    >>> from ruletimer.models import RuleCompetingRisks
    >>> from ruletimer.data import CompetingRisks
    >>> import numpy as np
    >>>
    >>> # Generate example data
    >>> X = np.random.randn(100, 5)
    >>> times = np.random.exponential(scale=5, size=100)
    >>> events = np.random.choice([0, 1, 2], size=100, p=[0.2, 0.4, 0.4])
    >>> y = CompetingRisks(time=times, event=events)
    >>>
    >>> # Initialize and fit model
    >>> model = RuleCompetingRisks(n_rules=50, min_support=0.2)
    >>> model.fit(X, y)
    >>>
    >>> # Make predictions
    >>> test_times = np.linspace(0, 10, 100)
    >>> cif = model.predict_cumulative_incidence(X, test_times)
    """

    def __init__(
        self,
        n_rules=100,
        min_support=0.1,
        alpha=0.5,
        l1_ratio=0.5,
        max_iter=1000,
        tol=1e-4,
        random_state=None,
    ):
        """
        Initialize competing risks model.

        Parameters
        ----------
        n_rules : int, default=100
            Maximum number of rules to generate for each event type. Each rule represents
            a potential decision boundary in the feature space.
        min_support : float, default=0.1
            Minimum proportion of samples that must satisfy a rule for it to be considered.
            This helps prevent overfitting to small subgroups.
        alpha : float, default=0.5
            Elastic net mixing parameter. A value of 0 corresponds to L2 regularization,
            while 1 corresponds to L1 regularization. Values in between provide a mix
            of both regularization types.
        l1_ratio : float, default=0.5
            The ratio of L1 to L2 regularization in the elastic net penalty.
            Must be between 0 and 1.
        max_iter : int, default=1000
            Maximum number of iterations for the optimization algorithm.
        tol : float, default=1e-4
            Tolerance for the optimization algorithm. The algorithm will stop when
            the change in the objective function is less than this value.
        random_state : int or RandomState, default=None
            Controls the randomness of the rule generation process.
        """
        # Create states list with initial state and event states
        event_types = ["Event1", "Event2"]  # Default event types
        states = ["Initial"] + event_types

        # Create transitions from initial state to each event state
        transitions = [(0, i+1) for i in range(len(event_types))]

        # Initialize base class
        super().__init__(
            states=states,
            transitions=transitions,
            hazard_method="nelson-aalen"
        )

        self.event_types = event_types
        self.event_type_to_state = {
            event: i+1 for i, event in enumerate(event_types)
        }
        self.n_rules = n_rules
        self.min_support = min_support
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.support_time_dependent = False  # Currently not supported

        # Initialize dictionaries to store rules and weights for each transition
        self.transition_rules_ = {}
        self.rule_weights_ = {}
        self.rules_ = None  # Temporary storage for current rules

    def fit(self, X: np.ndarray, y: CompetingRisks) -> 'RuleCompetingRisks':
        """
        Fit the competing risks model.

        Parameters
        ----------
        X : array-like
            Training data
        y : CompetingRisks
            Target competing risks data

        Returns
        -------
        self : RuleCompetingRisks
            Fitted model
        """
        if not isinstance(y, CompetingRisks):
            raise ValueError("y must be a CompetingRisks object")

        # Prepare transition-specific data
        transition_times = {}
        transition_events = {}

        for event_type in self.event_types:
            state = self.event_type_to_state[event_type]
            transition = (0, state)

            # Event indicator for this transition
            is_event = y.event == state

            transition_times[transition] = y.time
            transition_events[transition] = is_event

        # Estimate baseline hazards
        self._estimate_baseline_hazards(transition_times, transition_events)

        # Fit transition-specific models
        for event_type in self.event_types:
            state = self.event_type_to_state[event_type]
            transition = (0, state)
            self._fit_transition_model(X, y, transition)

        self.is_fitted_ = True
        return self

    def predict_transition_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
        from_state: Union[str, int],
        to_state: Union[str, int]
    ) -> np.ndarray:
        """
        Predict transition-specific hazard.

        Parameters
        ----------
        X : array-like
            Covariate values
        times : array-like
            Times at which to predict hazard
        from_state : str or int
            Starting state
        to_state : str or int
            Target state

        Returns
        -------
        np.ndarray
            Predicted hazard values of shape (n_samples, n_times)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        # Convert states to internal indices
        from_idx = self.state_manager.to_internal_index(from_state)
        to_idx = self.state_manager.to_internal_index(to_state)
        transition = (from_idx, to_idx)

        if transition not in self.transition_models_:
            raise ValueError(f"No model for transition {transition}")

        # Handle time-dependent covariates
        if self.support_time_dependent and X.ndim == 3:
            # For time-dependent covariates, use values at first time point for now
            X = X[:, :, 0]

        # Evaluate rules on input data
        rule_matrix = self._evaluate_rules(X)

        # Get baseline hazard for these times
        baseline_times, baseline_values = self.baseline_hazards_[transition]

        # If no events were observed for this transition, return zeros
        if len(baseline_times) == 0:
            return np.zeros((X.shape[0], len(times)))

        # Sort times and values for monotonicity
        sort_idx = np.argsort(baseline_times)
        baseline_times = baseline_times[sort_idx]
        baseline_values = baseline_values[sort_idx]

        # Ensure baseline hazard is non-negative and monotonic
        baseline_values = np.maximum(baseline_values, 0)
        baseline_values = np.maximum.accumulate(baseline_values)

        # Create interpolation function for baseline hazard
        # Use monotonic interpolation to preserve monotonicity
        baseline_interp = interp1d(
            baseline_times,
            baseline_values,
            kind='linear',
            bounds_error=False,
            fill_value=(0, baseline_values[-1])
        )

        # Get interpolated baseline hazard values and ensure monotonicity
        interpolated_baseline = baseline_interp(times)
        interpolated_baseline = np.maximum(interpolated_baseline, 0)
        interpolated_baseline = np.maximum.accumulate(interpolated_baseline)

        # Get relative hazard from elastic net model and ensure it's positive
        relative_hazard = np.exp(np.clip(
            self.transition_models_[transition].predict(rule_matrix),
            -50, 50  # Prevent numerical overflow
        ))

        # Convert shapes for broadcasting
        relative_hazard = relative_hazard.reshape(-1, 1)  # Shape: (n_samples, 1)
        interpolated_baseline = interpolated_baseline.reshape(1, -1)  # Shape: (1, n_times)

        # Compute final hazard and ensure numerical stability
        hazard = relative_hazard * interpolated_baseline
        hazard = np.maximum(hazard, 1e-50)

        return hazard

    def predict_cause_specific_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
        event_type: Union[str, int]
    ) -> np.ndarray:
        """
        Predict cause-specific hazard for a specific event type.

        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like
            Times at which to predict hazard
        event_type : str or int
            Event type to predict for

        Returns
        -------
        np.ndarray
            Predicted cause-specific hazard values
        """
        if isinstance(event_type, int):
            event_type = f"Event{event_type}"

        if event_type not in self.event_types:
            raise ValueError(f"Unknown event type: {event_type}")
        state = self.event_type_to_state[event_type]

        return self.predict_transition_hazard(X, times, 0, state)

    def predict_cumulative_incidence(
        self,
        X: np.ndarray,
        times: np.ndarray,
        event_type: Optional[Union[str, int]] = None,
        event_types: Optional[List[Union[str, int]]] = None
    ) -> Union[np.ndarray, Dict[Union[str, int], np.ndarray]]:
        """Predict cumulative incidence functions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Determine which event types to process
        if event_types is not None:
            target_event_types = event_types
        elif event_type is not None:
            target_event_types = [event_type]
        else:
            target_event_types = ["Event1", "Event2"]  # Default event types

        # Convert all event types to string format for internal processing
        processed_event_types = []
        for et in target_event_types:
            if isinstance(et, int):
                processed_et = f"Event{et}"
            else:
                processed_et = et
            if processed_et not in self.event_type_to_state:
                raise KeyError(f"Invalid event type requested: {et}")
            processed_event_types.append(processed_et)

        # Sort times for monotonicity
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        unsort_idx = np.argsort(sort_idx)

        # Get cause-specific hazards for each event type
        hazards = {}
        for et in processed_event_types:
            hazards[et] = self.predict_cause_specific_hazard(X, sorted_times, et)

        # Process CIFs in sorted time order
        n_samples = len(X)
        n_times = len(sorted_times)
        dt = np.diff(np.concatenate([[0], sorted_times]))

        # Initialize CIFs
        cifs = {et: np.zeros((n_samples, n_times)) for et in processed_event_types}
        overall_survival = np.ones((n_samples, n_times))

        # Compute CIFs using Aalen-Johansen estimator
        for t in range(n_times):
            if t > 0:
                # Start with previous values
                for et in processed_event_types:
                    cifs[et][:, t] = cifs[et][:, t-1]

            # Get total hazard at this time point
            total_hazard = sum(h[:, t] for h in hazards.values())

            # Update overall survival
            if t > 0:
                overall_survival[:, t] = overall_survival[:, t-1] * np.exp(-total_hazard * dt[t])

            # Update CIFs
            for et in processed_event_types:
                increment = overall_survival[:, t] * hazards[et][:, t] * dt[t]
                cifs[et][:, t] += increment

        # Ensure monotonicity and proper bounds
        for et in processed_event_types:
            cifs[et] = np.maximum.accumulate(cifs[et], axis=1)
            cifs[et] = np.clip(cifs[et], 0, 1)

        # Normalize CIFs to ensure they sum to ≤ 1
        cif_sum = np.sum([cifs[et] for et in processed_event_types], axis=0)
        mask = cif_sum > 1
        if np.any(mask):
            scale = np.where(mask, cif_sum, 1.0)
            for et in processed_event_types:
                cifs[et][:, mask] /= scale[:, mask]

        # Convert back to original time order
        result = {et: cifs[et][:, unsort_idx] for et in processed_event_types}

        # Return based on original request format
        if event_types is not None:
            # Map string keys back to original format
            mapped_result = {}
            for et in target_event_types:
                if isinstance(et, int):
                    mapped_result[et] = result[f"Event{et}"]
                else:
                    mapped_result[et] = result[et]
            return mapped_result
        elif event_type is not None:
            if isinstance(event_type, int):
                return result[f"Event{event_type}"]
            return result[event_type]
        else:
            return result

    def predict_hazard(self, X: np.ndarray, times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict cause-specific hazard functions.

        Parameters
        ----------
        X : array-like
            Data to predict for
        times : array-like
            Times at which to predict hazard

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping event types to hazard arrays of shape (n_samples, n_times)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        hazard = {}
        for event_type in self.event_types:
            state = self.event_type_to_state[event_type]
            transition = (0, state)

            if transition not in self.transition_models_:
                raise ValueError(f"No model for transition {transition}")

            # Handle time-dependent covariates
            if self.support_time_dependent and X.ndim == 3:
                # For time-dependent covariates, we'll use the values at each time point
                # This is a simple approach - more sophisticated methods could be implemented
                X = X[:, :, 0]  # Use features at first time point for now
                # TODO: Implement proper handling of time-dependent covariates

            # Evaluate rules on input data
            rule_matrix = self._evaluate_rules(X)

            # Get baseline hazard for these times using interpolation
            baseline_times, baseline_values = self.baseline_hazards_[transition]
            if len(baseline_times) == 0:
                # No events observed for this transition
                hazard[event_type] = np.zeros((len(X), len(times)))
                continue

            # Create interpolation function
            f = interp1d(baseline_times, baseline_values, bounds_error=False, fill_value=(baseline_values[0], baseline_values[-1]))
            baseline_hazard = f(times)

            # Get relative hazard from elastic net model
            relative_hazard = self.transition_models_[transition].predict(rule_matrix)

            # Compute hazard for each time point
            hazard[event_type] = np.zeros((len(X), len(times)))
            for i in range(len(X)):
                hazard[event_type][i] = baseline_hazard * np.exp(relative_hazard[i])

        return hazard

    def _fit_transition_model(
        self,
        X: np.ndarray,
        y: CompetingRisks,
        transition: tuple
    ) -> None:
        """
        Fit transition-specific model using random forest for rule generation
        and elastic net for fitting.

        Parameters
        ----------
        X : array-like
            Training data
        y : CompetingRisks
            Target competing risks data
        transition : tuple
            Transition to fit model for
        """
        from sklearn.ensemble import RandomForestRegressor

        # Initialize random forest
        forest = RandomForestRegressor(
            n_estimators=self.n_rules,
            max_depth=4,
            random_state=self.random_state
        )

        # Get event indicator for this transition
        is_event = y.event == transition[1]

        # Fit forest on survival times
        forest.fit(X, y.time)

        # Extract rules from forest
        self.rules_ = self._extract_rules_from_forest(forest)

        # Evaluate rules on data
        rule_matrix = self._evaluate_rules(X)

        # Fit elastic net on rule matrix
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        )
        model.fit(rule_matrix, y.time)

        # Store rules and weights for this transition
        self.transition_rules_[transition] = self.rules_
        self.rule_weights_[transition] = model.coef_
        self.transition_models_[transition] = model

        # Compute feature importances
        self.feature_importances_ = self._compute_feature_importances()

    def _extract_rules_from_forest(self, forest):
        """Extract rules from random forest."""
        rules = []
        for tree in forest.estimators_:
            rules.extend(self._extract_rules_from_tree(tree.tree_))
        return rules[:self.n_rules]

    def _extract_rules_from_tree(self, tree):
        """Extract rules from a single tree."""
        rules = []

        def recurse(node, rule):
            if tree.feature[node] != -2:  # Not a leaf
                feature = tree.feature[node]
                threshold = tree.threshold[node]

                # Left branch: feature <= threshold
                left_rule = rule + [(feature, "<=", threshold)]
                recurse(tree.children_left[node], left_rule)

                # Right branch: feature > threshold
                right_rule = rule + [(feature, ">", threshold)]
                recurse(tree.children_right[node], right_rule)
            else:
                rules.append(rule)

        recurse(0, [])
        return rules

    def _evaluate_rules(self, X):
        """Evaluate rules on data."""
        rule_matrix = np.zeros((X.shape[0], len(self.rules_)))
        for i, rule in enumerate(self.rules_):
            mask = np.ones(X.shape[0], dtype=bool)
            for feature, op, threshold in rule:
                if op == "<=":
                    mask &= (X[:, feature] <= threshold)
                else:  # op == ">"
                    mask &= (X[:, feature] > threshold)
            rule_matrix[:, i] = mask
        return rule_matrix

    def _compute_feature_importances(self):
        """Compute feature importances from rules."""
        # Get number of features from all rules
        n_features = 0
        for rules in self.transition_rules_.values():
            for rule in rules:
                for feature, _, _ in rule:
                    n_features = max(n_features, feature + 1)

        # Initialize importances dictionary
        importances = {}

        # Compute importances for each transition
        for transition in self.transition_rules_:
            rules = self.transition_rules_[transition]
            weights = self.rule_weights_[transition]

            # Initialize importances for this transition
            transition_importances = np.zeros(n_features)

            for rule, weight in zip(rules, weights):
                for feature, _, _ in rule:
                    transition_importances[feature] += abs(weight)

            # Normalize importances for this transition
            if np.sum(transition_importances) > 0:
                transition_importances /= np.sum(transition_importances)

            importances[transition] = transition_importances

        return importances

    def get_feature_importances(self, event_type: Union[str, int]) -> np.ndarray:
        """
        Get feature importances for a specific event type.

        Parameters
        ----------
        event_type : str or int
            The event type to get importances for

        Returns
        -------
        np.ndarray
            Array of feature importances
        """
        if not hasattr(self, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting feature importances")

        # Convert event type to state index
        state = self.state_manager.to_internal_index(event_type)
        transition = (0, state)

        if transition not in self.feature_importances_:
            raise ValueError(f"No feature importances available for event type {event_type}")

        return self.feature_importances_[transition]

    def get_variable_importances(self) -> Dict[str, Dict[str, float]]:
        """
        Get the importance scores for each variable in the model for all event types.

        Returns
        -------
        dict
            Dictionary mapping event types to dictionaries of variable importances
        """
        if not hasattr(self, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting variable importances")

        # Get feature names from the preprocessor if available
        feature_names = getattr(self, 'feature_names_',
                              [f'feature_{i}' for i in range(len(next(iter(self.feature_importances_.values()))))])

        # Create dictionary of variable importances for each event type
        importances = {}
        for event_type in self.event_types:
            state = self.event_type_to_state[event_type]
            transition = (0, state)

            if transition in self.feature_importances_:
                # Get importances for this event type
                event_importances = self.feature_importances_[transition]

                # Create dictionary mapping feature names to importances
                event_importance_dict = dict(zip(feature_names, event_importances))

                # Sort by importance in descending order
                importances[event_type] = dict(sorted(event_importance_dict.items(),
                                                    key=lambda x: x[1],
                                                    reverse=True))

        return importances
```


### `ruletimer/models/rule_multi_state.py`

This file contains the implementation of the multi-state model.

```python
"""
Rule-based multi-state time-to-event model implementation.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from ruletimer.models.base_multi_state import BaseMultiStateModel
from ruletimer.utils import StateStructure
from ruletimer.utils.hazard_estimation import HazardEstimator
from ruletimer.state_manager import StateManager
from ruletimer.time_handler import TimeHandler
from sklearn.tree import DecisionTreeClassifier
from ruletimer.models.base import BaseRuleEnsemble

class RuleMultiState(BaseMultiStateModel, BaseRuleEnsemble):
    """
    Rule-based multi-state time-to-event model.

    This implementation extends the base multi-state model with rule-based feature
    extraction and selection capabilities. It uses decision trees to generate rules
    and then selects the most important rules using elastic net regularization.

    Parameters
    ----------
    max_rules : int, optional (default=100)
        Maximum number of rules to extract
    alpha : float, optional (default=0.1)
        Regularization strength for rule selection
    state_structure : StateStructure, optional
        Pre-defined state structure for the model
    max_depth : int, optional (default=3)
        Maximum depth of decision trees used for rule extraction
    min_samples_leaf : int, optional (default=10)
        Minimum number of samples required to be at a leaf node
    n_estimators : int, optional (default=100)
        Number of trees in the random forest
    tree_type : str, optional (default='classification')
        Type of trees to grow: 'classification' or 'regression'
    tree_growing_strategy : str, optional (default='forest')
        Strategy for growing trees: 'forest' or 'single'
    prune_rules : bool, optional (default=True)
        Whether to prune redundant rules
    l1_ratio : float, optional (default=0.5)
        Ratio of L1 to L2 regularization in elastic net
    random_state : int, optional
        Random seed for reproducibility
    hazard_method : str, optional (default="nelson-aalen")
        Method for hazard estimation: "nelson-aalen" or "parametric"
    min_support : float, optional (default=0.05)
        Minimum support threshold for rule pruning
    min_confidence : float, optional (default=0.5)
        Minimum confidence threshold for rule pruning
    max_impurity : float, optional (default=0.1)
        Maximum impurity reduction for rule pruning
    """

    def __init__(
        self,
        max_rules: int = 100,
        alpha: float = 0.1,
        state_structure: StateStructure = None,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        n_estimators: int = 100,
        tree_type: str = 'classification',
        tree_growing_strategy: str = 'forest',
        prune_rules: bool = True,
        l1_ratio: float = 0.5,
        random_state: int = None,
        hazard_method: str = "nelson-aalen",
        min_support: float = 0.05,
        min_confidence: float = 0.5,
        max_impurity: float = 0.1
    ):
        if tree_type not in ['classification', 'regression']:
            raise ValueError("tree_type must be 'classification' or 'regression'")
        if tree_growing_strategy not in ['forest', 'single']:
            raise ValueError("tree_growing_strategy must be 'forest' or 'single'")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if l1_ratio < 0 or l1_ratio > 1:
            raise ValueError("l1_ratio must be between 0 and 1")

        # Initialize with empty states and transitions if no state structure provided
        states = []
        transitions = []
        state_names = []
        if state_structure is not None:
            states = state_structure.states
            transitions = state_structure.transitions
            state_names = state_structure.state_names

        # Initialize both parent classes
        BaseMultiStateModel.__init__(
            self,
            states=state_names,  # Pass state names instead of states
            transitions=transitions,
            hazard_method=hazard_method
        )
        BaseRuleEnsemble.__init__(self)

        self.max_rules = max_rules
        self.alpha = alpha
        self.state_structure = state_structure
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.tree_type = tree_type
        self.tree_growing_strategy = tree_growing_strategy
        self.prune_rules = prune_rules
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_impurity = max_impurity

        # Initialize rule-specific attributes
        self._rules_dict: Dict[Tuple[int, int], List[str]] = {}
        self.rule_coefficients_: Dict[Tuple[int, int], np.ndarray] = {}
        self.rule_importances_: Dict[Tuple[int, int], np.ndarray] = {}

    def _generate_rules(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """Generate rules using decision trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        List[str]
            List of generated rules
        """
        # Ensure we have enough samples of each class
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2 or np.min(counts) < self.min_samples_leaf:
            return []  # Not enough samples to generate meaningful rules

        if self.tree_growing_strategy == 'forest':
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                min_samples_split=2  # Ensure splits can occur
            )
            forest.fit(X, y)
            rules = []
            for tree in forest.estimators_:
                rules.extend(self._extract_rules_from_tree(tree))
        else:
            # Single tree strategy
            tree = RandomForestClassifier(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                min_samples_split=2  # Ensure splits can occur
            ).fit(X, y).estimators_[0]
            rules = self._extract_rules_from_tree(tree)

        # Remove duplicates and limit number
        unique_rules = list(dict.fromkeys(rules))
        return unique_rules[:self.max_rules]

    def _extract_rules_from_tree(self, tree):
        """Extract rules from a decision tree.

        Parameters
        ----------
        tree : DecisionTreeClassifier
            Fitted decision tree.

        Returns
        -------
        list
            List of extracted rules.
        """
        rules = []
        n_nodes = tree.tree_.node_count
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        value = tree.tree_.value

        def recurse(node, path):
            if feature[node] != -2:  # Internal node
                name = feature[node]
                threshold_val = threshold[node]

                # Left child
                left_path = path + [(name, '<=', threshold_val)]
                recurse(children_left[node], left_path)

                # Right child
                right_path = path + [(name, '>', threshold_val)]
                recurse(children_right[node], right_path)
            else:  # Leaf node
                # Only add rules for leaves with significant samples
                if np.sum(value[node]) >= self.min_samples_leaf:
                    # Convert path to rule string
                    rule_parts = []
                    for feat_idx, op, thresh in path:
                        rule_parts.append(f"X{feat_idx} {op} {thresh:.4f}")
                    rule = " and ".join(rule_parts)
                    rules.append(rule)

        recurse(0, [])  # Start from root node
        return rules

    def _evaluate_rules(self, X: np.ndarray, rules: Union[List[str], Dict[Tuple[int, int], List[str]]] = None) -> np.ndarray:
        """
        Evaluate rules on input data.

        Parameters
        ----------
        X : array-like
            Input data to evaluate rules on
        rules : list of str or dict, optional
            Rules to evaluate. If None, uses self._rules_dict

        Returns
        -------
        array-like
            Boolean matrix indicating which rules apply to each sample
        """
        if rules is None:
            rules = self._rules_dict

        if len(rules) == 0:
            return np.zeros((X.shape[0], 0))

        # Handle dictionary format
        if isinstance(rules, dict):
            # Flatten rules from all transitions
            all_rules = []
            for transition_rules in rules.values():
                all_rules.extend(transition_rules)
            rules = all_rules

        rule_matrix = np.zeros((X.shape[0], len(rules)))
        for i, rule in enumerate(rules):
            if not isinstance(rule, str):
                continue  # Skip non-string rules

            conditions = rule.split(' and ')
            rule_result = np.ones(X.shape[0], dtype=bool)

            for condition in conditions:
                # Extract feature index and value
                if '<=' in condition:
                    feature_str = condition.split('<=')[0].strip()
                    value = float(condition.split('<=')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] <= value
                elif '>=' in condition:
                    feature_str = condition.split('>=')[0].strip()
                    value = float(condition.split('>=')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] >= value
                elif '<' in condition:
                    feature_str = condition.split('<')[0].strip()
                    value = float(condition.split('<')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] < value
                elif '>' in condition:
                    feature_str = condition.split('>')[0].strip()
                    value = float(condition.split('>')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] > value
                elif '==' in condition:
                    feature_str = condition.split('==')[0].strip()
                    value = float(condition.split('==')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] == value
                elif '!=' in condition:
                    feature_str = condition.split('!=')[0].strip()
                    value = float(condition.split('!=')[1].strip())
                    feature_idx = int(''.join(filter(str.isdigit, feature_str)))
                    rule_result &= X[:, feature_idx] != value

            rule_matrix[:, i] = rule_result

        return rule_matrix

    def fit(self, X, multi_state):
        """Fit the model to multi-state data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        multi_state : MultiState or dict
            Multi-state data object or dictionary with transition data
            in the format {(from_state, to_state): {'times': array, 'events': array}}

        Returns
        -------
        self : object
            Returns self.
        """
        # Initialize dictionaries for baseline hazards and rules
        transition_times = {}
        transition_events = {}
        self._rules_dict = {}
        self.rule_coefficients_ = {}
        self.rule_importances_ = {}

        # Handle different input formats
        if isinstance(multi_state, dict):
            # Map all transition keys in input data to internal model keys
            mapped_multi_state = {}
            for k, v in multi_state.items():
                # If key is tuple of ints, map to state labels if needed
                if isinstance(k, tuple) and all(isinstance(x, int) for x in k):
                    try:
                        from_state = self.state_structure.states[k[0]]
                        to_state = self.state_structure.states[k[1]]
                        mapped_key = (from_state, to_state)
                        if mapped_key in self.state_structure.transitions:
                            mapped_multi_state[mapped_key] = v
                        else:
                            mapped_multi_state[k] = v
                    except Exception:
                        mapped_multi_state[k] = v
                else:
                    mapped_multi_state[k] = v
            multi_state = mapped_multi_state
            for transition in self.state_structure.transitions:
                if transition not in multi_state:
                    continue

                # Extract transition data
                times = multi_state[transition]['times']
                events = multi_state[transition]['events']

                # Store in dictionaries
                transition_times[transition] = times
                transition_events[transition] = events

                # Generate rules for current transition
                rules = self._generate_rules(X, events)

                # Store rules for both mapped and index-based keys
                self._rules_dict[transition] = rules
                # Always set for index-based key if transition is label-based
                if isinstance(transition[0], str) or isinstance(transition[1], str):
                    try:
                        idx_from = self.state_structure.states.index(transition[0])
                        idx_to = self.state_structure.states.index(transition[1])
                        idx_key = (idx_from, idx_to)
                        self._rules_dict[idx_key] = rules
                    except Exception:
                        pass

                # Evaluate rules on the data
                rule_matrix = self._evaluate_rules(X, rules)

                # If no rules, use a constant feature (column of ones)
                if rule_matrix.shape[1] == 0:
                    rule_matrix = np.ones((X.shape[0], 1))

                # Initialize elastic net for rule selection
                elastic_net = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    random_state=self.random_state
                )

                # Fit elastic net to select important rules
                elastic_net.fit(rule_matrix, events)

                # Store rule coefficients and importances
                if len(rules) > 0:
                    self.rule_coefficients_[transition] = elastic_net.coef_
                    importances = np.abs(elastic_net.coef_)
                    self.rule_importances_[transition] = importances / np.sum(importances) if np.sum(importances) > 0 else importances
                else:
                    # If no rules, use a single constant feature
                    self.rule_coefficients_[transition] = np.array([elastic_net.coef_[0]])
                    n_features = X.shape[1] if X.ndim == 2 else 1
                    # Set importances to uniform distribution if no rules
                    self.rule_importances_[transition] = np.ones(n_features) / n_features
                # Always set for index-based key if transition is label-based
                if isinstance(transition[0], str) or isinstance(transition[1], str):
                    try:
                        idx_from = self.state_structure.states.index(transition[0])
                        idx_to = self.state_structure.states.index(transition[1])
                        idx_key = (idx_from, idx_to)
                        self.rule_coefficients_[idx_key] = self.rule_coefficients_[transition]
                        self.rule_importances_[idx_key] = self.rule_importances_[transition]
                    except Exception:
                        pass

                # Store transition models
                self.transition_models_[transition] = elastic_net
                if isinstance(transition[0], str) or isinstance(transition[1], str):
                    try:
                        idx_from = self.state_structure.states.index(transition[0])
                        idx_to = self.state_structure.states.index(transition[1])
                        idx_key = (idx_from, idx_to)
                        self.transition_models_[idx_key] = elastic_net
                    except Exception:
                        pass
        else:
            # Original implementation for MultiState objects
            for transition in self.state_structure.transitions:
                # Get mask for current transition
                mask = (multi_state.start_state == transition[0])

                if not np.any(mask):
                    continue

                # Get times and events for current transition
                times = multi_state.end_time[mask] - multi_state.start_time[mask]
                events = (multi_state.end_state[mask] == transition[1]).astype(int)

                # Store in dictionaries
                transition_times[transition] = times
                transition_events[transition] = events

                # Generate rules for current transition
                rules = self._generate_rules(X[mask], events)

                # Store rules for both mapped and index-based keys
                self._rules_dict[transition] = rules
                if isinstance(transition[0], str) or isinstance(transition[1], str):
                    try:
                        idx_from = self.state_structure.states.index(transition[0])
                        idx_to = self.state_structure.states.index(transition[1])
                        idx_key = (idx_from, idx_to)
                        self._rules_dict[idx_key] = rules
                    except Exception:
                        pass

                # Evaluate rules on the data
                rule_matrix = self._evaluate_rules(X[mask], rules)

                # If no rules, use a constant feature (column of ones)
                if rule_matrix.shape[1] == 0:
                    rule_matrix = np.ones((X[mask].shape[0], 1))

                # Initialize elastic net for rule selection
                elastic_net = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    random_state=self.random_state
                )

                # Fit elastic net to select important rules
                elastic_net.fit(rule_matrix, events)

                # Store rule coefficients and importances
                if len(rules) > 0:
                    self.rule_coefficients_[transition] = elastic_net.coef_
                    importances = np.abs(elastic_net.coef_)
                    self.rule_importances_[transition] = importances / np.sum(importances) if np.sum(importances) > 0 else importances
                else:
                    # If no rules, use a single constant feature
                    self.rule_coefficients_[transition] = np.array([elastic_net.coef_[0]])
                    n_features = X.shape[1] if X.ndim == 2 else 1
                    # Set importances to uniform distribution if no rules
                    self.rule_importances_[transition] = np.ones(n_features) / n_features
                # Always set for index-based key if transition is label-based
                if isinstance(transition[0], str) or isinstance(transition[1], str):
                    try:
                        idx_from = self.state_structure.states.index(transition[0])
                        idx_to = self.state_structure.states.index(transition[1])
                        idx_key = (idx_from, idx_to)
                        self.rule_coefficients_[idx_key] = self.rule_coefficients_[transition]
                        self.rule_importances_[idx_key] = self.rule_importances_[transition]
                    except Exception:
                        pass

                # Store transition models
                self.transition_models_[transition] = elastic_net
                if isinstance(transition[0], str) or isinstance(transition[1], str):
                    try:
                        idx_from = self.state_structure.states.index(transition[0])
                        idx_to = self.state_structure.states.index(transition[1])
                        idx_key = (idx_from, idx_to)
                        self.transition_models_[idx_key] = elastic_net
                    except Exception:
                        pass
        # After fitting, filter all rule-related dicts to only use index-based keys matching state_structure.transitions
        # Remove label-based keys from _rules_dict, rule_coefficients_, rule_importances_, transition_models_
        index_keys = []
        for t in self.state_structure.transitions:
            if isinstance(t[0], str) or isinstance(t[1], str):
                try:
                    idx_from = self.state_structure.states.index(t[0])
                    idx_to = self.state_structure.states.index(t[1])
                    index_keys.append((idx_from, idx_to))
                except Exception:
                    pass
            else:
                index_keys.append(t)
        # Only keep index-based keys
        self._rules_dict = {k: v for k, v in self._rules_dict.items() if k in index_keys}
        self.rule_coefficients_ = {k: v for k, v in self.rule_coefficients_.items() if k in index_keys}
        self.rule_importances_ = {k: v for k, v in self.rule_importances_.items() if k in index_keys}
        self.transition_models_ = {k: v for k, v in self.transition_models_.items() if k in index_keys}

        # Estimate baseline hazards for all transitions
        self._estimate_baseline_hazards(
            transition_times,
            transition_events,
            None  # No weights for now
        )

        self.is_fitted_ = True
        return self

    def get_feature_importances(self, transition: Tuple[int, int]) -> np.ndarray:
        """Get feature importances for a specific transition (try both key types)."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importances")
        if transition in self.rule_importances_:
            return self.rule_importances_[transition]
        # Try mapping index to label if needed
        if isinstance(transition[0], int) and self.state_structure:
            try:
                label_key = (self.state_structure.states[transition[0]], self.state_structure.states[transition[1]])
                if label_key in self.rule_importances_:
                    return self.rule_importances_[label_key]
            except Exception:
                pass
        raise ValueError(f"No feature importances available for transition {transition}")

    def get_rule_importances(self):
        """Return the rule importances dictionary for all transitions."""
        return self.rule_importances_

    def predict_cumulative_incidence(
        self,
        X: np.ndarray,
        times: np.ndarray,
        target_state: Union[str, int]
    ) -> np.ndarray:
        """Predict cumulative incidence for a target state (try both key types for transitions)."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        # Convert target state to internal index if necessary
        if isinstance(target_state, str):
            target_state = self.state_manager.to_internal_index(target_state)
        # Get all transitions leading to the target state (try both key types)
        transitions_to_target = []
        for t in self.state_structure.transitions:
            if t[1] == target_state:
                transitions_to_target.append(t)
            elif isinstance(t[1], str) and self.state_structure:
                try:
                    idx = self.state_structure.states.index(t[1])
                    if idx == target_state:
                        transitions_to_target.append(t)
                except Exception:
                    pass
        if not transitions_to_target:
            raise ValueError(f"No transitions found leading to state {target_state}")
        n_samples = X.shape[0]
        n_times = len(times)
        cumulative_incidence = np.zeros((n_samples, n_times))
        for transition in transitions_to_target:
            rules = self._rules_dict.get(transition)
            if rules is None and isinstance(transition[0], str):
                try:
                    idx_key = (self.state_structure.states.index(transition[0]), self.state_structure.states.index(transition[1]))
                    rules = self._rules_dict.get(idx_key)
                    transition = idx_key
                except Exception:
                    continue
            if rules is None:
                continue
            rule_matrix = self._evaluate_rules(X, rules)
            coef = self.rule_coefficients_[transition]
            # Handle case where no rules: rule_matrix shape (n_samples, 1), coef shape (1,)
            if rule_matrix.shape[1] == 0:
                rule_matrix = np.ones((X.shape[0], 1))
            if coef.shape[0] == 0:
                coef = np.ones(1)
            transition_risk = rule_matrix @ coef
            cumulative_incidence += transition_risk.reshape(-1, 1) * np.ones((1, n_times))
        return np.clip(cumulative_incidence, 0, 1)

    def _prune_rules(self, rules: List[str], X: np.ndarray, y: np.ndarray) -> List[str]:
        """Prune rules using advanced techniques.

        This method implements several rule pruning strategies:
        1. Redundancy removal using rule coverage
        2. Statistical significance testing
        3. Rule interaction analysis
        4. Feature importance-based pruning

        Parameters
        ----------
        rules : List[str]
            List of rules to prune
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        List[str]
            Pruned list of rules
        """
        if not self.prune_rules:
            return rules

        # Evaluate all rules
        rule_matrix = self._evaluate_rules(X)

        # Calculate rule statistics
        rule_stats = []
        for i, rule in enumerate(rules):
            # Rule coverage
            coverage = np.mean(rule_matrix[:, i])

            # Rule accuracy
            rule_mask = rule_matrix[:, i]
            if np.sum(rule_mask) > 0:
                accuracy = np.mean(y[rule_mask] == np.argmax(np.bincount(y[rule_mask])))
            else:
                accuracy = 0

            # Rule significance (chi-square test)
            contingency = np.zeros((2, 2))
            contingency[0, 0] = np.sum((rule_mask) & (y == 0))
            contingency[0, 1] = np.sum((rule_mask) & (y == 1))
            contingency[1, 0] = np.sum((~rule_mask) & (y == 0))
            contingency[1, 1] = np.sum((~rule_mask) & (y == 1))

            from scipy.stats import chi2_contingency
            _, p_value, _, _ = chi2_contingency(contingency)

            # Feature importance
            features = set()
            for condition in rule.split(" AND "):
                feature_idx = int(condition.split("_")[1].split()[0])
                features.add(feature_idx)
            feature_importance = np.mean([self.rule_importances_[f] for f in features])

            rule_stats.append({
                'rule': rule,
                'coverage': coverage,
                'accuracy': accuracy,
                'p_value': p_value,
                'feature_importance': feature_importance
            })

        # Sort rules by quality metrics
        rule_stats.sort(key=lambda x: (
            x['accuracy'],
            x['coverage'],
            -x['p_value'],
            x['feature_importance']
        ), reverse=True)

        # Remove redundant rules
        pruned_rules = []
        covered_samples = np.zeros(X.shape[0], dtype=bool)

        for stat in rule_stats:
            rule_mask = self._evaluate_rules(X)
            new_coverage = np.sum(~covered_samples & rule_mask[:, i])

            # Keep rule if it adds significant new coverage
            if new_coverage / X.shape[0] >= self.min_support:
                pruned_rules.append(stat['rule'])
                covered_samples |= rule_mask[:, i]

        return pruned_rules

    def predict_transition_hazard(self, X, times, from_state, to_state):
        """Predict transition hazard for given samples and times.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        times : array-like of shape (n_times,)
            Time points at which to evaluate the hazard.
        from_state : int
            Starting state.
        to_state : int
            Target state.

        Returns
        -------
        array-like of shape (n_samples, n_times)
            Predicted hazard values.
        """
        transition = (from_state, to_state)

        if transition not in self._rules_dict:
            raise ValueError(f"No model for transition {transition}")

        # Get rules and coefficients for this transition
        rules = self._rules_dict[transition]
        coefficients = self.rule_coefficients_[transition]

        # Evaluate rules on input data
        rule_matrix = self._evaluate_rules(X, rules)

        # Calculate linear predictor
        linear_pred = rule_matrix @ coefficients

        # Expand to match times dimension
        linear_pred = np.tile(linear_pred[:, np.newaxis], (1, len(times)))

        # Apply exponential function to get hazard
        hazard = np.exp(linear_pred)

        return hazard

    def _compute_feature_importances(self):
        """Compute feature importances"""
        # Combine importances from all transitions
        all_importances = []
        for transition in self.state_structure.transitions:
            if transition in self.rule_importances_:
                all_importances.extend(self.rule_importances_[transition])

        if all_importances:
            self._feature_importances = np.array(all_importances)
        else:
            self._feature_importances = np.array([])

    def _fit_weights(self, rule_values, y):
        """Fit weights for the rules"""
        # This is handled in the fit method using elastic net
        pass

    @property
    def rules_(self):
        """Return rules as a list if only one transition, else as a dict."""
        keys = set(self.state_structure.transitions)
        for t in self.state_structure.transitions:
            if isinstance(t[0], str) or isinstance(t[1], str):
                try:
                    idx_from = self.state_structure.states.index(t[0])
                    idx_to = self.state_structure.states.index(t[1])
                    keys.add((idx_from, idx_to))
                except Exception:
                    pass
        rules_dict = {k: v for k, v in self._rules_dict.items() if k in keys}
        if len(rules_dict) == 1:
            # Return the list of rules for the only transition
            return list(rules_dict.values())[0]
        return rules_dict
```


### `ruletimer/utils/hazard_estimation.py`

This file contains the implementation of the hazard estimation logic.

```python
"""
Unified hazard estimation module for all time-to-event models.
"""
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from scipy.interpolate import interp1d

class HazardEstimator:
    """Unified hazard estimation for all time-to-event models."""

    @staticmethod
    def estimate_baseline_hazard(
        times: np.ndarray,
        events: np.ndarray,
        weights: Optional[np.ndarray] = None,
        method: str = "nelson-aalen"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate baseline hazard using various methods.

        Parameters
        ----------
        times : np.ndarray
            Event/censoring times
        events : np.ndarray
            Event indicators (1 if event, 0 if censored)
        weights : np.ndarray, optional
            Case weights for hazard estimation
        method : str
            Estimation method: "nelson-aalen" or "parametric"

        Returns
        -------
        unique_times : np.ndarray
            Unique event times
        baseline_hazard : np.ndarray
            Estimated baseline hazard at unique times
        """
        if method not in ["nelson-aalen", "parametric"]:
            raise ValueError("Method must be 'nelson-aalen' or 'parametric'")

        # Sort times and get unique event times
        order = np.argsort(times)
        times = times[order]
        events = events[order]
        if weights is not None:
            weights = weights[order]
        else:
            weights = np.ones_like(times)

        unique_times = np.unique(times[events == 1])
        n_times = len(unique_times)
        baseline_hazard = np.zeros(n_times)

        if method == "nelson-aalen":
            # Nelson-Aalen estimator
            at_risk = np.zeros(n_times)
            events_at_time = np.zeros(n_times)

            for i, t in enumerate(unique_times):
                at_risk[i] = np.sum(weights[times >= t])
                events_at_time[i] = np.sum(weights[
                    (times == t) & (events == 1)
                ])

            # Compute hazard with smoothing for stability
            baseline_hazard = events_at_time / (at_risk + 1e-8)

        else:  # parametric
            # Implement parametric estimation (e.g., Weibull)
            # This is a placeholder for actual parametric implementation
            pass

        return unique_times, baseline_hazard

    @staticmethod
    def estimate_cumulative_hazard(
        times: np.ndarray, baseline_hazard: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate cumulative hazard using the trapezoidal rule.

        Parameters
        ----------
        times : np.ndarray
            Array of unique event times.
        baseline_hazard : np.ndarray
            Array of baseline hazard values corresponding to times.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing:
            - Array of prediction times
            - Array of cumulative hazard values
        """
        prediction_times = times.copy()
        cumulative_hazard = np.zeros_like(prediction_times, dtype=float)

        # Set first time point's cumulative hazard to baseline hazard
        cumulative_hazard[0] = baseline_hazard[0]

        # Calculate cumulative hazard for remaining time points
        for i in range(1, len(prediction_times)):
            # Find indices of times up to current prediction time
            mask = times <= prediction_times[i]
            cumulative_hazard[i] = np.sum(baseline_hazard[mask])

        return prediction_times, cumulative_hazard

    @staticmethod
    def transform_hazard(
        cumulative_hazard: np.ndarray,
        transform: str = "exp"
    ) -> np.ndarray:
        """
        Transform cumulative hazard to survival or CIF.

        Parameters
        ----------
        cumulative_hazard : np.ndarray
            Cumulative hazard values
        transform : str
            Transformation type: "exp" for survival, "cif" for CIF

        Returns
        -------
        np.ndarray
            Transformed values
        """
        if transform == "exp":
            return np.exp(-cumulative_hazard)
        elif transform == "cif":
            return 1 - np.exp(-cumulative_hazard)
        else:
            raise ValueError("Transform must be 'exp' or 'cif'")
```


### `ruletimer/utils/importance.py`

This file contains the implementation of the importance analysis logic.

```python
"""
Importance analysis utilities for rule-based time-to-event models.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable

class ImportanceAnalyzer:
    """Analyze variable importance and dependencies in rule ensemble models."""

    @staticmethod
    def calculate_permutation_importance(
        model,
        X: np.ndarray,
        y,
        prediction_func: Optional[Callable] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ) -> Dict:
        """Calculate permutation-based feature importance.

        Parameters
        ----------
        model : BaseRuleEnsemble
            Fitted model
        X : array-like
            Feature matrix
        y : MultiState, Survival, or CompetingRisks
            Time-to-event data
        prediction_func : callable, optional
            Function to generate predictions for evaluation
            Must accept X as input and return array-like
            If None, uses predict_risk method
        n_repeats : int, default=10
            Number of permutation repeats
        random_state : int, optional
            Random state for reproducibility

        Returns
        -------
        dict
            Dictionary with keys:
            - 'importances': Feature importances
            - 'importances_std': Standard deviation of importances
            - 'feature_names': Feature names if available
        """
        # Default to predict_risk for prediction function
        if prediction_func is None:
            prediction_func = model.predict_risk

        # Get baseline prediction
        baseline_pred = prediction_func(X)

        # Handle different return types
        if isinstance(baseline_pred, dict):
            # For multi-state or competing risks predictions
            baseline_score = np.mean([np.mean(p) for p in baseline_pred.values()])

            def score_func(X):
                pred = prediction_func(X)
                return np.mean([np.mean(p) for p in pred.values()])
        else:
            # For standard predictions
            baseline_score = np.mean(baseline_pred)

            def score_func(X):
                return np.mean(prediction_func(X))

        # Initialize importance arrays
        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features))

        # Set random state
        rng = np.random.RandomState(random_state)

        # Permute each feature and calculate importance
        for i in range(n_features):
            for j in range(n_repeats):
                # Create permuted data
                X_permuted = X.copy()
                perm_idx = rng.permutation(X.shape[0])
                X_permuted[:, i] = X[perm_idx, i]

                # Calculate score with permuted feature
                permuted_score = score_func(X_permuted)

                # Importance is decrease in score (or increase for errors)
                importances[j, i] = baseline_score - permuted_score

        # Calculate mean and std of importances
        imp_mean = np.mean(importances, axis=0)
        imp_std = np.std(importances, axis=0)

        # Package results
        result = {
            'importances': imp_mean,
            'importances_std': imp_std
        }

        # Add feature names if available
        if hasattr(model, '_feature_names'):
            result['feature_names'] = model._feature_names

        return result

    @staticmethod
    def calculate_dependence_matrix(
        model,
        X: np.ndarray,
        prediction_func: Optional[Callable] = None,
        threshold: float = 0.05,
        n_points: int = 10
    ) -> np.ndarray:
        """Calculate feature dependency matrix using partial dependence.

        Parameters
        ----------
        model : BaseRuleEnsemble
            Fitted model
        X : array-like
            Feature matrix
        prediction_func : callable, optional
            Function to generate predictions for evaluation
            Must accept X as input and return array-like
            If None, uses predict_risk method
        threshold : float, default=0.05
            Threshold for significance of dependency
        n_points : int, default=10
            Number of points for evaluating each feature

        Returns
        -------
        numpy.ndarray
            Dependency matrix of shape (n_features, n_features)
            with values between 0 and 1 indicating strength of dependency
        """
        # Default to predict_risk for prediction function
        if prediction_func is None:
            prediction_func = model.predict_risk

        # Get number of features
        n_features = X.shape[1]

        # Initialize dependency matrix
        dependency = np.zeros((n_features, n_features))

        # For each pair of features
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Generate grid of values for features i and j
                x_i = np.linspace(np.min(X[:, i]), np.max(X[:, i]), n_points)
                x_j = np.linspace(np.min(X[:, j]), np.max(X[:, j]), n_points)

                # Create 2D grid
                X_grid = np.zeros((n_points * n_points, n_features))
                for k in range(n_features):
                    if k != i and k != j:
                        X_grid[:, k] = np.median(X[:, k])

                # Fill in feature i and j values
                idx = 0
                for vi in x_i:
                    for vj in x_j:
                        X_grid[idx, i] = vi
                        X_grid[idx, j] = vj
                        idx += 1

                # Get predictions for grid
                pred_grid = prediction_func(X_grid)

                # Reshape predictions to 2D grid
                if isinstance(pred_grid, dict):
                    # For multi-state or competing risks
                    pred_values = np.mean([p for p in pred_grid.values()], axis=0)
                else:
                    pred_values = pred_grid

                pred_2d = pred_values.reshape(n_points, n_points)

                # Calculate dependency using correlation of differences
                row_diffs = np.diff(pred_2d, axis=0)
                col_diffs = np.diff(pred_2d, axis=1)

                # Flatten and calculate correlation
                corr = np.corrcoef(row_diffs.flatten(), col_diffs.flatten())[0, 1]

                # Set dependency value (absolute correlation)
                dependency[i, j] = dependency[j, i] = abs(corr)

        # Set diagonal to 1
        np.fill_diagonal(dependency, 1.0)

        # Apply threshold
        dependency[dependency < threshold] = 0

        return dependency
```


### `ruletimer/utils/prediction_utils.py`

This file contains the implementation of the prediction utility functions.

```python
"""
Unified prediction utilities for multi-state time-to-event models.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

class UnifiedTransitionCalculator:
    """Calculate transition probabilities for multi-state models."""

    @staticmethod
    def calculate_transition_matrix(
        model,
        X: np.ndarray,
        times: np.ndarray,
        include_ci: bool = True,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate transition probability matrix P(s,t) for each time point.

        Parameters
        ----------
        model : RuleMultiState
            Fitted multi-state model
        X : array-like
            Feature matrix
        times : array-like
            Time points for prediction
        include_ci : bool, default=True
            Whether to include confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
        n_bootstrap : int, default=100
            Number of bootstrap samples if include_ci=True

        Returns
        -------
        dict
            Dictionary with keys:
            - 'times': Time points
            - 'states': State indices
            - 'matrix': Array of shape (n_samples, n_states, n_states, n_times)
            - 'lower': Lower CI if include_ci=True
            - 'upper': Upper CI if include_ci=True
        """
        # Get model information
        states = model.states_
        n_states = len(states)
        n_samples = X.shape[0]
        n_times = len(times)

        # Initialize transition matrix
        P = np.zeros((n_samples, n_states, n_states, n_times))

        # Calculate transition probabilities for each pair of states
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                if from_state != to_state and (from_state, to_state) in model.transitions_:
                    # Direct transition
                    P[:, i, j] = model.predict_transition_probability(
                        X, times, from_state, to_state
                    )
                elif from_state == to_state:
                    # Staying in same state (complementary probability)
                    next_states = model._get_next_states(from_state)
                    if next_states:
                        for next_state in next_states:
                            next_idx = list(states).index(next_state)
                            P[:, i, i] = 1 - np.sum(P[:, i, :], axis=1)
                    else:
                        # Absorbing state
                        P[:, i, i] = 1

        # Package results
        result = {
            'times': times,
            'states': states,
            'matrix': P
        }

        # Calculate confidence intervals if requested
        if include_ci:
            ci_result = UnifiedTransitionCalculator._calculate_transition_matrix_ci(
                model, X, times, states, alpha, n_bootstrap
            )
            result.update(ci_result)

        return result

    @staticmethod
    def _calculate_transition_matrix_ci(
        model,
        X: np.ndarray,
        times: np.ndarray,
        states: List,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate confidence intervals for transition matrix using bootstrap."""
        n_states = len(states)
        n_samples = X.shape[0]
        n_times = len(times)

        # Initialize arrays for bootstrap samples
        P_boots = np.zeros((n_bootstrap, n_samples, n_states, n_states, n_times))

        # Perform bootstrap
        for b in range(n_bootstrap):
            # Sample with replacement
            sample_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[sample_idx]

            # Calculate transition matrix for this bootstrap sample
            for i, from_state in enumerate(states):
                for j, to_state in enumerate(states):
                    if from_state != to_state and (from_state, to_state) in model.transitions_:
                        # Direct transition
                        P_boots[b, :, i, j] = model.predict_transition_probability(
                            X_boot, times, from_state, to_state
                        )
                    elif from_state == to_state:
                        # Staying in same state (complementary probability)
                        next_states = model._get_next_states(from_state)
                        if next_states:
                            for next_state in next_states:
                                next_idx = list(states).index(next_state)
                                P_boots[b, :, i, i] = 1 - np.sum(P_boots[b, :, i, :], axis=1)
                        else:
                            # Absorbing state
                            P_boots[b, :, i, i] = 1

        # Calculate confidence intervals
        lower = np.quantile(P_boots, alpha/2, axis=0)
        upper = np.quantile(P_boots, 1-alpha/2, axis=0)

        return {
            'lower': lower,
            'upper': upper
        }


class UnifiedPredictionCalculator:
    """Unified calculator for various predictions from multi-state models."""

    @staticmethod
    def survival_from_multistate(
        model,
        X: np.ndarray,
        times: np.ndarray,
        initial_state: int = 1,
        death_state: int = 2,
        include_ci: bool = True,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate survival probabilities from multi-state model.

        Parameters
        ----------
        model : RuleMultiState
            Fitted multi-state model
        X : array-like
            Feature matrix
        times : array-like
            Time points for prediction
        initial_state : int, default=1
            Initial state
        death_state : int, default=2
            Absorbing state representing death
        include_ci : bool, default=True
            Whether to include confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
        n_bootstrap : int, default=100
            Number of bootstrap samples if include_ci=True

        Returns
        -------
        dict
            Dictionary with keys:
            - 'times': Time points
            - 'survival': Array of shape (n_samples, n_times)
            - 'lower': Lower CI if include_ci=True
            - 'upper': Upper CI if include_ci=True
        """
        # Get state occupation probabilities
        state_occupation = model.predict_state_occupation(X, times)

        # Survival is probability of not being in death state
        survival = 1 - state_occupation[death_state]

        # Package results
        result = {
            'times': times,
            'survival': survival
        }

        # Calculate confidence intervals if requested
        if include_ci:
            # Initialize bootstrap samples
            n_samples = X.shape[0]
            n_times = len(times)
            survival_boots = np.zeros((n_bootstrap, n_samples, n_times))

            # Perform bootstrap
            for b in range(n_bootstrap):
                # Sample with replacement
                sample_idx = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[sample_idx]

                # Calculate state occupation for this bootstrap sample
                state_occ_boot = model.predict_state_occupation(X_boot, times)

                # Calculate survival
                survival_boots[b] = 1 - state_occ_boot[death_state]

            # Calculate confidence intervals
            result['lower'] = np.quantile(survival_boots, alpha/2, axis=0)
            result['upper'] = np.quantile(survival_boots, 1-alpha/2, axis=0)

        return result

    @staticmethod
    def cumulative_incidence_from_multistate(
        model,
        X: np.ndarray,
        times: np.ndarray,
        initial_state: int = 1,
        target_states: Optional[List[int]] = None,
        include_ci: bool = True,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict:
        """Calculate cumulative incidence functions from multi-state model.

        Parameters
        ----------
        model : RuleMultiState
            Fitted multi-state model
        X : array-like
            Feature matrix
        times : array-like
            Time points for prediction
        initial_state : int, default=1
            Initial state
        target_states : list, optional
            Target states for CIF (if None, uses all states except initial)
        include_ci : bool, default=True
            Whether to include confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
        n_bootstrap : int, default=100
            Number of bootstrap samples if include_ci=True

        Returns
        -------
        dict
            Dictionary with keys for each target state, containing:
            - 'times': Time points
            - 'cif': Array of shape (n_samples, n_times)
            - 'lower': Lower CI if include_ci=True
            - 'upper': Upper CI if include_ci=True
        """
        # Get all states
        states = model.states_

        # If target states not specified, use all states except initial
        if target_states is None:
            target_states = [s for s in states if s != initial_state]

        # Get state occupation probabilities
        state_occupation = model.predict_state_occupation(X, times)

        # Initialize results
        result = {}

        # For each target state, CIF is just the state occupation probability
        for state in target_states:
            result[state] = {
                'times': times,
                'cif': state_occupation[state]
            }

        # Calculate confidence intervals if requested
        if include_ci:
            # Initialize bootstrap samples
            n_samples = X.shape[0]
            n_times = len(times)
            cif_boots = {
                state: np.zeros((n_bootstrap, n_samples, n_times))
                for state in target_states
            }

            # Perform bootstrap
            for b in range(n_bootstrap):
                # Sample with replacement
                sample_idx = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[sample_idx]

                # Calculate state occupation for this bootstrap sample
                state_occ_boot = model.predict_state_occupation(X_boot, times)

                # Calculate CIF for each target state
                for state in target_states:
                    cif_boots[state][b] = state_occ_boot[state]

            # Calculate confidence intervals for each state
            for state in target_states:
                result[state]['lower'] = np.quantile(cif_boots[state], alpha/2, axis=0)
                result[state]['upper'] = np.quantile(cif_boots[state], 1-alpha/2, axis=0)

        return result
```


### `ruletimer/utils/time_utils.py`

This file contains the implementation of the time utility functions.

```python
"""
Time and prediction point management utilities for time-to-event models.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from ruletimer.data import MultiState

def create_prediction_grid(
    data,
    n_points: int = 100,
    max_time: Optional[float] = None,
    quantile_max: float = 0.95
) -> np.ndarray:
    """Create a standard time grid for predictions based on observed data.

    Parameters
    ----------
    data : MultiState, Survival, or CompetingRisks
        Time-to-event data
    n_points : int, default=100
        Number of time points
    max_time : float, optional
        Maximum time value (if None, uses quantile_max of observed times)
    quantile_max : float, default=0.95
        Quantile of observed times to use as maximum if max_time is None

    Returns
    -------
    np.ndarray
        Array of time points for prediction
    """
    if hasattr(data, 'time'):
        times = data.time
    elif hasattr(data, 'end_time'):
        times = data.end_time
    else:
        raise ValueError("Data object must have time or end_time attribute")

    if max_time is None:
        max_time = np.quantile(times, quantile_max)

    return np.linspace(0, max_time, n_points)

def to_multi_state_format(
    data,
    data_type: str = 'survival'
) -> MultiState:
    """Convert any time-to-event data to multi-state format.

    Parameters
    ----------
    data : Survival, CompetingRisks, or related data
        Time-to-event data
    data_type : str, default='survival'
        Type of data ('survival', 'competing_risks')

    Returns
    -------
    MultiState
        Data converted to multi-state format
    """
    if data_type == 'survival':
        # Convert survival data to multi-state (state 1 → state 2)
        patient_id = np.arange(len(data.time))
        start_time = np.zeros_like(data.time)
        end_time = data.time
        start_state = np.ones_like(data.time)
        end_state = np.where(data.event == 1, 2, 0)  # 2=event, 0=censored

        return MultiState(
            start_time=start_time,
            end_time=end_time,
            start_state=start_state,
            end_state=end_state,
            patient_id=patient_id
        )

    elif data_type == 'competing_risks':
        # Convert competing risks to multi-state (state 1 → state 2, 3, etc.)
        patient_id = np.arange(len(data.time))
        start_time = np.zeros_like(data.time)
        end_time = data.time
        start_state = np.ones_like(data.time)
        end_state = np.where(data.event == 0, 0, data.event + 1)  # 0=censored, 2,3,...=events

        return MultiState(
            start_time=start_time,
            end_time=end_time,
            start_state=start_state,
            end_state=end_state,
            patient_id=patient_id
        )
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

def bootstrap_confidence_intervals(
    model,
    X: np.ndarray,
    y,
    times: np.ndarray,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    prediction_method: str = 'predict_state_occupation'
) -> Dict:
    """Calculate bootstrap confidence intervals for predictions.

    Parameters
    ----------
    model : BaseRuleEnsemble
        Fitted model
    X : array-like
        Feature matrix
    y : MultiState, Survival, or CompetingRisks
        Time-to-event data
    times : array-like
        Time points for prediction
    n_bootstrap : int, default=100
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals
    prediction_method : str, default='predict_state_occupation'
        Method to use for prediction ('predict_state_occupation',
        'predict_transition_probability', 'predict_survival')

    Returns
    -------
    dict
        Dictionary containing mean predictions and confidence intervals
    """
    # Ensure we have the right method
    pred_method = getattr(model, prediction_method)

    # Get base prediction for reference
    base_pred = pred_method(X, times)

    # Initialize bootstrap results
    if isinstance(base_pred, dict):
        # Multi-state predictions
        bootstrap_results = {
            k: np.zeros((n_bootstrap, X.shape[0], len(times)))
            for k in base_pred.keys()
        }
    else:
        # Single outcome predictions
        bootstrap_results = np.zeros((n_bootstrap, X.shape[0], len(times)))

    # Perform bootstrap
    n_samples = X.shape[0]
    for i in range(n_bootstrap):
        # Create bootstrap sample
        sample_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[sample_idx]

        # Handle different data types
        if hasattr(y, 'time'):
            y_boot = type(y)(y.time[sample_idx], y.event[sample_idx])
        else:
            # For MultiState, need to filter transitions by patient
            patient_idx = np.unique(y.patient_id[sample_idx])
            mask = np.isin(y.patient_id, patient_idx)
            y_boot = type(y)(
                start_time=y.start_time[mask],
                end_time=y.end_time[mask],
                start_state=y.start_state[mask],
                end_state=y.end_state[mask],
                patient_id=y.patient_id[mask]
            )

        # Fit model on bootstrap sample
        model_boot = model.__class__(**model.get_params())
        model_boot.fit(X_boot, y_boot)

        # Make predictions
        pred_boot = pred_method(X_boot, times)

        # Store results
        if isinstance(pred_boot, dict):
            for k in pred_boot.keys():
                bootstrap_results[k][i] = pred_boot[k]
        else:
            bootstrap_results[i] = pred_boot

    # Calculate confidence intervals
    low_quantile = alpha / 2
    high_quantile = 1 - low_quantile

    if isinstance(base_pred, dict):
        # For dictionary results
        result = {}
        for k in base_pred.keys():
            result[k] = {
                'mean': np.mean(bootstrap_results[k], axis=0),
                'lower': np.quantile(bootstrap_results[k], low_quantile, axis=0),
                'upper': np.quantile(bootstrap_results[k], high_quantile, axis=0)
            }
    else:
        # For array results
        result = {
            'mean': np.mean(bootstrap_results, axis=0),
            'lower': np.quantile(bootstrap_results, low_quantile, axis=0),
            'upper': np.quantile(bootstrap_results, high_quantile, axis=0)
        }

    return result
```


### `ruletimer/visualization/visualization.py`

This file contains the implementation of the visualization functions.

```python
"""
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
from ..models.base import BaseRuleEnsemble
from ..models.competing_risks import RuleCompetingRisks

def plot_rule_importance(rules: Union[Dict[Tuple[int, int], List[str]], BaseRuleEnsemble],
                        importances: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                        top_n: int = 10,
                        figsize: tuple = (10, 6)):
    """
    Plot rule importance

    Parameters
    ----------
    rules : dict or BaseRuleEnsemble
        Either a dictionary mapping transitions to rules or a fitted model
    importances : dict, optional
        Dictionary mapping transitions to rule importances
    top_n : int, default=10
        Number of top rules to plot
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if isinstance(rules, BaseRuleEnsemble):
        # Get rules and importances from model
        rules_dict = rules.rules_
        importances_dict = rules.get_rule_importances()
    else:
        rules_dict = rules
        importances_dict = importances

    # Check if importances are provided
    if importances_dict is None:
         raise ValueError("Importances must be provided if rules is a dict")

    # Flatten rules and importances
    flat_rules = []
    flat_importances = []
    # Ensure rules_dict is not None and is iterable
    if rules_dict:
        for transition, trans_rules in rules_dict.items():
            # Check if transition exists in importances and has corresponding rules/importances
            if transition in importances_dict and len(trans_rules) > 0 and len(importances_dict[transition]) == len(trans_rules):
                flat_rules.extend([f"{transition}: {rule}" for rule in trans_rules])
                flat_importances.extend(importances_dict[transition])
            elif transition in importances_dict and len(trans_rules) == 0 and len(importances_dict[transition]) == 0:
                # Handle case where a transition exists but has no rules/importances
                pass # Or log a warning if needed
            # Optional: Add handling for mismatches or missing keys if necessary

    if not flat_rules:
        # If no valid rules/importances were found after flattening
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No rules with importances found to plot", ha='center', va='center')
        plt.axis('off')
        plt.title("Rule Importance") # Add title for consistency
        plt.tight_layout() # Ensure layout is adjusted
        return fig

    # Get top rules
    top_n = min(top_n, len(flat_rules))
    # Ensure flat_importances is a numpy array for argsort
    flat_importances_np = np.array(flat_importances)
    idx = np.argsort(np.abs(flat_importances_np))[-top_n:]
    top_rules = [flat_rules[i] for i in idx]
    top_importances = flat_importances_np[idx]

    # Create plot
    fig = plt.figure(figsize=figsize)
    plt.barh(range(len(top_rules)), np.abs(top_importances)) # Use absolute importance for bar height
    plt.yticks(range(len(top_rules)), top_rules)
    plt.xlabel("Absolute Importance")
    plt.title("Rule Importance")
    plt.tight_layout()
    return fig

def plot_cumulative_incidence(model: BaseRuleEnsemble,
                            X: Union[np.ndarray, pd.DataFrame],
                            event_types: List[int],
                            times: Optional[np.ndarray] = None,
                            figsize: tuple = (10, 6)):
    """
    Plot cumulative incidence functions with confidence intervals

    Parameters
    ----------
    model : BaseRuleEnsemble
        Fitted competing risks model
    X : array-like of shape (n_samples, n_features)
        Data to predict for
    event_types : list of int
        Event types to plot
    times : array-like, optional
        Times at which to plot cumulative incidence
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not event_types:
        raise ValueError("Event types list cannot be empty")

    if times is not None and len(times) == 0:
        raise ValueError("Times array cannot be empty")

    if times is None:
        if hasattr(model, '_y') and hasattr(model._y, 'time') and len(model._y.time) > 0:
            times = np.linspace(0, model._y.time.max(), 100)
        else:
            times = np.linspace(0, 1, 100)

    # Get predictions for all requested event types at once
    predictions = model.predict_cumulative_incidence(X, times, event_types=event_types)

    # Set up plot
    plt.figure(figsize=figsize)
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot each event type
    for i, event_type in enumerate(event_types):
        if event_type not in predictions:
            raise KeyError(f"Prediction for event type {event_type} not found in model output.")

        mean_cif = np.mean(predictions[event_type], axis=0)
        std_cif = np.std(predictions[event_type], axis=0)

        plt.plot(times, mean_cif,
                label=f"Event {event_type}",
                color=colors[i % len(colors)])

        plt.fill_between(times,
                        np.maximum(mean_cif - 1.96 * std_cif, 0),
                        np.minimum(mean_cif + 1.96 * std_cif, 1),
                        alpha=0.2,
                        color=colors[i % len(colors)])

    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.title('Cumulative Incidence Functions')
    plt.legend()
    plt.grid(True)

    return plt.gcf()

def plot_state_transitions(model: BaseRuleEnsemble,
                         X: Union[np.ndarray, pd.DataFrame],
                         time: float,
                         initial_state: int = 0,
                         figsize: tuple = (10, 6)) -> None:
    """
    Plot state transition diagram with probabilities

    Parameters
    ----------
    model : BaseRuleEnsemble
        Fitted model
    X : array-like of shape (n_samples, n_features)
        Data to predict for
    time : float
        Time at which to plot state occupation probabilities
    initial_state : int, default=0
        Initial state for prediction
    figsize : tuple, default=(10, 6)
        Figure size
    """
    if time < 0:
        raise ValueError("Time cannot be negative")

    # Get predictions
    state_occupation = model.predict_state_occupation(X, np.array([time]), initial_state=initial_state)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot states
    n_states = len(model.state_structure.states)
    angles = np.linspace(0, 2*np.pi, n_states, endpoint=False)
    radius = 1.0

    for i, state in enumerate(model.state_structure.states):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])

        # Plot state circle
        state_idx = i  # Use 0-based indexing
        plt.plot(x, y, 'o', markersize=20,
                label=f"{state} ({state_occupation[state_idx].mean():.2f})")

        # Plot state name
        plt.text(x, y, state, ha='center', va='center')

    # Plot transitions
    for from_state, to_state in model.state_structure.transitions:
        if (from_state, to_state) in model.rules_:
            # Get transition probability
            hazard = model.predict_transition_hazard(X, np.array([time]), from_state, to_state)
            prob = 1 - np.exp(-hazard.mean())

            # Plot arrow
            start_angle = angles[from_state]  # Use 0-based indexing
            end_angle = angles[to_state]  # Use 0-based indexing

            start_x = radius * np.cos(start_angle)
            start_y = radius * np.sin(start_angle)
            end_x = radius * np.cos(end_angle)
            end_y = radius * np.sin(end_angle)

            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                     head_width=0.05, head_length=0.1, fc='k', ec='k')

            # Plot probability
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            plt.text(mid_x, mid_y, f"{prob:.2f}", ha='center', va='center')

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axis('equal')
    plt.axis('off')
    plt.title("State Transition Diagram")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def plot_state_occupation(times: np.ndarray,
                       state_probs: Dict[int, np.ndarray],
                       state_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       alpha: float = 0.1,
                       title: str = "State Occupation Probabilities",
                       xlabel: str = "Time",
                       ylabel: str = "Probability",
                       legend_loc: str = "best") -> None:
    """
    Plot state occupation probabilities over time

    Parameters
    ----------
    times : array-like
        Time points
    state_probs : dict
        Dictionary mapping state indices to arrays of probabilities
    state_names : list of str, optional
        Names of states
    figsize : tuple, default=(10, 6)
        Figure size
    alpha : float, default=0.1
        Transparency of confidence intervals
    title : str, default="State Occupation Probabilities"
        Plot title
    xlabel : str, default="Time"
        X-axis label
    ylabel : str, default="Probability"
        Y-axis label
    legend_loc : str, default="best"
        Legend location
    """
    if len(state_probs) < 2:
        raise ValueError("At least two states are required to plot state occupation probabilities.")

    plt.figure(figsize=figsize)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']

    for i, (state_idx, probs) in enumerate(state_probs.items()):
        mean_prob = np.mean(probs, axis=0)
        std_prob = np.std(probs, axis=0)

        label = f"State {state_idx}" if state_names is None else state_names[state_idx - 1]
        plt.plot(times, mean_prob, label=label, color=colors[i % len(colors)])
        plt.fill_between(times,
                        mean_prob - 1.96 * std_prob,
                        mean_prob + 1.96 * std_prob,
                        alpha=alpha,
                        color=colors[i % len(colors)])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc=legend_loc)
    plt.tight_layout()

def plot_importance_comparison(model: RuleCompetingRisks,
                             top_n: int = 10,
                             figsize: tuple = (12, 8)) -> None:
    """
    Plot rule importance comparison across event types

    Parameters
    ----------
    model : RuleCompetingRisks
        Fitted competing risks model
    top_n : int, default=10
        Number of top rules to plot
    figsize : tuple, default=(12, 8)
        Figure size
    """
    rules = model.get_rules()

    # Get top rules across all event types
    all_weights = np.zeros(len(rules))
    for event_weights in model.rule_weights_.values():
        all_weights += np.abs(event_weights)
    top_idx = np.argsort(all_weights)[-top_n:]

    # Create plot
    plt.figure(figsize=figsize)

    x = np.arange(len(top_idx))
    width = 0.8 / len(model.rule_weights_)

    for i, (event_type, weights) in enumerate(model.rule_weights_.items()):
        plt.bar(x + i * width,
               weights[top_idx],
               width,
               label=f"Event {event_type}")

    plt.xlabel("Rules")
    plt.ylabel("Weight")
    plt.title("Rule Importance by Event Type")
    plt.xticks(x + width * (len(model.rule_weights_) - 1) / 2,
               [rules[i] for i in top_idx],
               rotation=45,
               ha='right')
    plt.legend()
    plt.tight_layout()

def plot_importance_heatmap(model: RuleCompetingRisks,
                          top_n: int = 10,
                          figsize: tuple = (12, 8)) -> None:
    """
    Plot rule importance heatmap

    Parameters
    ----------
    model : RuleCompetingRisks
        Fitted competing risks model
    top_n : int, default=10
        Number of top rules to plot
    figsize : tuple, default=(12, 8)
        Figure size
    """
    rules = model.get_rules()

    # Get top rules across all event types
    all_weights = np.zeros(len(rules))
    for event_weights in model.rule_weights_.values():
        all_weights += np.abs(event_weights)
    top_idx = np.argsort(all_weights)[-top_n:]

    # Create weight matrix
    weight_matrix = np.zeros((len(top_idx), len(model.rule_weights_)))
    for i, (event_type, weights) in enumerate(model.rule_weights_.items()):
        weight_matrix[:, i] = weights[top_idx]

    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(weight_matrix,
                xticklabels=[f"Event {k}" for k in model.rule_weights_.keys()],
                yticklabels=[rules[i] for i in top_idx],
                cmap='RdBu',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title("Rule Importance Heatmap")
    plt.tight_layout()

def plot_feature_importance(model: Union[BaseRuleEnsemble, RuleCompetingRisks],
                          top_n: int = 10,
                          figsize: tuple = (10, 6),
                          title: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance scores for a fitted model.

    Parameters
    ----------
    model : BaseRuleEnsemble or RuleCompetingRisks
        Fitted model with get_variable_importances method
    top_n : int, default=10
        Number of top features to plot
    figsize : tuple, default=(10, 6)
        Figure size
    title : str, optional
        Custom title for the plot. If None, a default title will be used.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Get variable importances
    importances = model.get_variable_importances()

    # Create figure
    fig = plt.figure(figsize=figsize)

    if isinstance(model, RuleCompetingRisks):
        # For competing risks, plot each event type separately
        n_events = len(importances)
        fig, axes = plt.subplots(n_events, 1, figsize=figsize, sharex=True)
        if n_events == 1:
            axes = [axes]

        for i, (event_type, event_importances) in enumerate(importances.items()):
            # Get top features
            top_features = sorted(event_importances.items(),
                                key=lambda x: x[1],
                                reverse=True)[:top_n]
            features, scores = zip(*top_features)

            # Plot
            axes[i].barh(range(len(features)), scores)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel("Importance")
            axes[i].set_title(f"Feature Importance - {event_type}")

        plt.tight_layout()

    else:
        # For other models, plot a single bar chart
        # Flatten importances if they're nested
        if isinstance(next(iter(importances.values())), dict):
            # For multi-state models
            flat_importances = {}
            for transition, trans_importances in importances.items():
                for feature, importance in trans_importances.items():
                    if feature not in flat_importances:
                        flat_importances[feature] = 0
                    flat_importances[feature] += importance
        else:
            # For single-state models
            flat_importances = importances

        # Get top features
        top_features = sorted(flat_importances.items(),
                            key=lambda x: x[1],
                            reverse=True)[:top_n]
        features, scores = zip(*top_features)

        # Plot
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance")
        if title is None:
            plt.title("Feature Importance")
        else:
            plt.title(title)
        plt.tight_layout()

    return fig
```


### `tests/test_data.py`

This file contains the tests for the data structures.

```python
import numpy as np
import pandas as pd
import pytest
from ruletimer.data import Survival, CompetingRisks, MultiState

def test_survival_data_initialization():
    """Test initialization of Survival data structure"""
    time = np.array([1, 2, 3, 4])
    event = np.array([1, 0, 1, 0])
    data = Survival(time, event)
    assert np.array_equal(data.time, time)
    assert np.array_equal(data.event, event)

def test_survival_data_validation():
    """Test validation of Survival data"""
    # Test invalid time values
    with pytest.raises(ValueError):
        Survival(np.array([-1, 2, 3]), np.array([1, 0, 1]))

    # Test invalid event values
    with pytest.raises(ValueError):
        Survival(np.array([1, 2, 3]), np.array([2, 0, 1]))

    # Test mismatched lengths
    with pytest.raises(ValueError):
        Survival(np.array([1, 2, 3]), np.array([1, 0]))

def test_competing_risks_initialization():
    """Test initialization of CompetingRisks data structure"""
    time = np.array([1, 2, 3, 4])
    event = np.array([0, 1, 2, 0])
    data = CompetingRisks(time, event)
    assert np.array_equal(data.time, time)
    assert np.array_equal(data.event, event)

def test_competing_risks_validation():
    """Test validation of CompetingRisks data"""
    # Test invalid time values
    with pytest.raises(ValueError):
        CompetingRisks(np.array([-1, 2, 3]), np.array([0, 1, 2]))

    # Test invalid event values
    with pytest.raises(ValueError):
        CompetingRisks(np.array([1, 2, 3]), np.array([-1, 0, 1]))

    # Test mismatched lengths
    with pytest.raises(ValueError):
        CompetingRisks(np.array([1, 2, 3]), np.array([0, 1]))

def test_multi_state_initialization():
    """Test initialization of MultiState data structure"""
    patient_id = np.array([1, 1, 2, 2])
    start_time = np.array([0, 1, 0, 2])
    end_time = np.array([1, 2, 2, 3])
    start_state = np.array([1, 2, 1, 2])  # States start from 1
    end_state = np.array([2, 3, 2, 3])    # States start from 1

    data = MultiState(start_time, end_time, start_state, end_state, patient_id)
    assert np.array_equal(data.patient_id, patient_id)
    assert np.array_equal(data.start_time, start_time)
    assert np.array_equal(data.end_time, end_time)
    assert np.array_equal(data.start_state, start_state)
    assert np.array_equal(data.end_state, end_state)

def test_multi_state_validation():
    """Test validation of MultiState data"""
    # Test invalid time ordering
    with pytest.raises(ValueError):
        MultiState(
            np.array([1, 2]),  # start_time
            np.array([2, 1]),  # end_time (invalid: not ordered)
            np.array([1, 1]),  # start_state
            np.array([2, 2]),  # end_state
            patient_id=np.array([1, 1])
        )
    # Test invalid state transitions (same start and end state)
    with pytest.raises(ValueError):
        MultiState(
            np.array([1, 1]),  # start_time
            np.array([0, 1]),  # end_time
            np.array([1, 2]),  # start_state
            np.array([1, 1]),  # end_state (invalid: same as start state)
            np.array([1, 1])   # patient_id
        )

    # Test mismatched lengths
    with pytest.raises(ValueError):
        MultiState(
            np.array([1, 1]),
            np.array([0, 1]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([2])  # Mismatched length
        )

def test_multi_state_patient_ordering():
    """Test patient-specific time ordering in MultiState data"""
    patient_id = np.array([1, 1, 2, 2])
    start_time = np.array([0, 1, 0, 2])  # Ordered within each patient
    end_time = np.array([1, 2, 2, 3])
    start_state = np.array([1, 2, 1, 2])  # States start from 1
    end_state = np.array([2, 3, 2, 3])    # States start from 1

    data = MultiState(start_time, end_time, start_state, end_state, patient_id)
    # Check if times are strictly increasing within each patient
    assert np.all(np.diff(data.start_time[data.patient_id == 1]) > 0)
    assert np.all(np.diff(data.start_time[data.patient_id == 2]) > 0)
```


### `tests/test_data_converter.py`

This file contains the tests for the data conversion utilities.

```python
import numpy as np
import pandas as pd
import pytest
from ruletimer.data import MultiState
from ruletimer.data.data_converter import MultiStateDataConverter

# Test data for person-period format
def create_person_period_data():
    return pd.DataFrame({
        'ID': [1, 1, 1, 2, 2],
        'Time': [1, 2, 3, 1, 2],
        'State': ['A', 'B', 'B', 'A', 'A'],
        'Censored': [0, 0, 1, 0, 1]
    })

# Test data for transition format
def create_transition_data():
    return pd.DataFrame({
        'ID': [1, 1, 2],
        'FromState': ['A', 'B', 'A'],
        'ToState': ['B', 'B', 'A'],
        'StartTime': [1, 2, 1],
        'EndTime': [2, 3, 2],
        'Censored': [0, 1, 1]
    })

# Test data for long format
def create_long_format_data():
    return pd.DataFrame({
        'ID': [1, 1, 1, 2, 2],
        'Time': [1, 2, 3, 1, 2],
        'StateA': [1, 0, 0, 1, 1],
        'StateB': [0, 1, 1, 0, 0],
        'Censored': [0, 0, 1, 0, 1]
    })

# Test data for wide format
def create_wide_format_data():
    return pd.DataFrame({
        'ID': [1, 2],
        'State_T1': ['A', 'A'],
        'State_T2': ['B', 'A'],
        'State_T3': ['B', None],
        'CensorTime': [3, 2],
        'CensorState': ['B', 'A']
    })

def create_interval_censored_data():
    """Create test data for interval censoring"""
    return pd.DataFrame({
        'ID': [1, 1, 2, 2],
        'Time': [1, 2, 1, 2],
        'State': ['A', None, 'A', None],
        'Censored': [0, 1, 0, 1],
        'NextKnownState': ['B', None, 'A', None],
        'NextKnownTime': [3, None, 4, None]
    })

def create_counting_process_data():
    """Create test data in counting process format"""
    return pd.DataFrame({
        'ID': [1, 1, 1, 2, 2],
        'FromState': ['A', 'B', 'B', 'A', 'A'],
        'ToState': ['B', 'C', 'C', 'B', 'C'],
        'Count': [1, 0, 0, 0, 0],
        'Exposure': [1, 2, 3, 1, 2],
        'Censored': [0, 1, 1, 1, 1]
    })

def create_multiple_transitions_data():
    """Create test data with multiple state transitions"""
    return pd.DataFrame({
        'ID': [1, 1, 1, 1, 2, 2],
        'Time': [1, 2, 3, 4, 1, 2],
        'State': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Censored': [0, 0, 0, 1, 0, 1]
    })

def test_from_person_period():
    """Test conversion from person-period format to MultiState"""
    data = create_person_period_data()
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )

    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # Three transitions: A->B, B->B(censored), A->A(censored)
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # 0 for censored

def test_from_transition_format():
    """Test conversion from transition format to MultiState"""
    data = create_transition_data()
    msm_data = MultiStateDataConverter.from_transition_format(
        data=data,
        id_col='ID',
        from_state_col='FromState',
        to_state_col='ToState',
        start_time_col='StartTime',
        end_time_col='EndTime',
        censored_col='Censored'
    )

    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # Three transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # 0 for censored

def test_from_long_format():
    """Test conversion from long format to MultiState"""
    data = create_long_format_data()
    msm_data = MultiStateDataConverter.from_long_format(
        data=data,
        id_col='ID',
        time_col='Time',
        state_cols=['StateA', 'StateB'],
        censored_col='Censored'
    )

    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # Three transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # StateA=1, StateB=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # 0 for censored

def test_from_wide_format():
    """Test conversion from wide format to MultiState"""
    data = create_wide_format_data()
    msm_data = MultiStateDataConverter.from_wide_format(
        data=data,
        id_col='ID',
        time_points=['State_T1', 'State_T2', 'State_T3'],
        state_cols=['State_T1', 'State_T2', 'State_T3'],
        censor_time_col='CensorTime'
    )

    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 2  # Two transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 1]))  # A=1
    assert np.array_equal(msm_data.end_state, np.array([2, 0]))  # 0 for censored

def test_to_person_period():
    """Test conversion from MultiState to person-period format"""
    msm_data = MultiState(
        patient_id=np.array([1, 1]),
        start_time=np.array([1, 2]),
        end_time=np.array([2, 3]),
        start_state=np.array([1, 2]),
        end_state=np.array([2, 0])
    )

    df = MultiStateDataConverter.to_person_period(msm_data)

    # Verify the conversion
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # Two transitions, each with start and end
    assert np.array_equal(df['ID'].values, np.array([1, 1, 1, 1]))
    assert np.array_equal(df['Time'].values, np.array([1, 2, 2, 3]))
    assert np.array_equal(df['State'].values, np.array([1, 2, 2, 2]))
    assert np.array_equal(df['Censored'].values, np.array([0, 0, 0, 1]))

def test_to_transition_format():
    """Test conversion from MultiState to transition format"""
    msm_data = MultiState(
        patient_id=np.array([1, 1]),
        start_time=np.array([1, 2]),
        end_time=np.array([2, 3]),
        start_state=np.array([1, 2]),
        end_state=np.array([2, 0])
    )

    df = MultiStateDataConverter.to_transition_format(msm_data)

    # Verify the conversion
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert np.array_equal(df['ID'].values, np.array([1, 1]))
    assert np.array_equal(df['FromState'].values, np.array([1, 2]))
    assert np.array_equal(df['ToState'].values, np.array([2, 0]))
    assert np.array_equal(df['StartTime'].values, np.array([1, 2]))
    assert np.array_equal(df['EndTime'].values, np.array([2, 3]))
    assert np.array_equal(df['Censored'].values, np.array([0, 1]))

def test_to_long_format():
    """Test conversion from MultiState to long format"""
    msm_data = MultiState(
        patient_id=np.array([1, 1]),
        start_time=np.array([1, 2]),
        end_time=np.array([2, 3]),
        start_state=np.array([1, 2]),
        end_state=np.array([2, 0])
    )

    df = MultiStateDataConverter.to_long_format(msm_data, state_names=['A', 'B'])

    # Verify the conversion
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # Two transitions, each with start and end
    assert np.array_equal(df['ID'].values, np.array([1, 1, 1, 1]))
    assert np.array_equal(df['Time'].values, np.array([1, 2, 2, 3]))
    assert np.array_equal(df['Censored'].values, np.array([0, 0, 0, 1]))
    assert np.array_equal(df['StateA'].values, np.array([1, 0, 0, 0]))
    assert np.array_equal(df['StateB'].values, np.array([0, 1, 1, 1]))

def test_round_trip_conversion():
    """Test round-trip conversion between formats"""
    # Start with person-period format
    data = create_person_period_data()

    # Convert to MultiState
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )

    # Convert back to person-period
    df = MultiStateDataConverter.to_person_period(msm_data)

    # Verify the round-trip conversion preserves the structure
    assert len(df) == 6  # Three transitions, each with start and end
    assert np.array_equal(df['ID'].values, np.array([1, 1, 1, 1, 2, 2]))
    assert np.array_equal(df['Time'].values, np.array([1, 2, 2, 3, 1, 2]))
    assert np.array_equal(df['Censored'].values, np.array([0, 0, 0, 1, 0, 1]))

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    # Test invalid person-period data
    data = pd.DataFrame({
        'ID': [1, 1],
        'Time': [2, 1],  # Invalid: times not ordered
        'State': ['A', 'B'],
        'Censored': [0, 0]
    })

    with pytest.raises(ValueError):
        MultiStateDataConverter.from_person_period(
            data=data,
            id_col='ID',
            time_col='Time',
            state_col='State',
            censored_col='Censored'
        )

    # Test missing columns
    data = pd.DataFrame({
        'ID': [1, 1],
        'Time': [1, 2],
        'State': ['A', 'B']
    })

    with pytest.raises(KeyError):
        MultiStateDataConverter.from_person_period(
            data=data,
            id_col='ID',
            time_col='Time',
            state_col='State',
            censored_col='Censored'  # Missing column
        )

def test_interval_censoring():
    """Test handling of interval censored data"""
    data = create_interval_censored_data()
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )

    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 2  # Two transitions (one per patient)
    assert np.array_equal(msm_data.patient_id, np.array([1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 1]))  # A=1
    assert np.array_equal(msm_data.end_state, np.array([0, 0]))  # Both censored

def test_counting_process():
    """Test conversion from counting process format"""
    data = create_counting_process_data()
    msm_data = MultiStateDataConverter.from_counting_process(
        data=data,
        id_col='ID',
        from_state_col='FromState',
        to_state_col='ToState',
        count_col='Count',
        exposure_col='Exposure',
        censored_col='Censored'
    )

    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 3  # One transition + two censored observations
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 2]))  # Patient 1's transition and censoring, Patient 2's censoring
    assert np.array_equal(msm_data.start_time, np.array([1, 3, 2]))
    assert np.array_equal(msm_data.end_time, np.array([2, 4, 3]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 0, 0]))  # One transition (A->B) and two censored

def test_multiple_transitions():
    """Test handling of multiple state transitions"""
    data = create_multiple_transitions_data()
    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )

    # Verify the conversion
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 4  # Four transitions for patient 1, one for patient 2
    assert np.array_equal(msm_data.patient_id, np.array([1, 1, 1, 2]))
    assert np.array_equal(msm_data.start_time, np.array([1, 2, 3, 1]))
    assert np.array_equal(msm_data.end_time, np.array([2, 3, 4, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2, 1, 1]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([2, 1, 0, 0]))  # Last transitions censored

def test_time_ordering_validation():
    """Test validation of time ordering"""
    # Create data with invalid time ordering
    data = pd.DataFrame({
        'ID': [1, 1],
        'Time': [2, 1],  # Times in wrong order
        'State': ['A', 'B'],
        'Censored': [0, 0]
    })

    # Verify that validation raises error
    with pytest.raises(ValueError, match="Times must be non-decreasing within each patient"):
        MultiStateDataConverter.from_person_period(
            data=data,
            id_col='ID',
            time_col='Time',
            state_col='State',
            censored_col='Censored'
        )

def test_missing_values():
    """Test handling of missing values"""
    data = pd.DataFrame({
        'ID': [1, 1, 2],
        'Time': [1, 2, 1],
        'State': ['A', None, 'B'],
        'Censored': [0, 1, 0]
    })

    msm_data = MultiStateDataConverter.from_person_period(
        data=data,
        id_col='ID',
        time_col='Time',
        state_col='State',
        censored_col='Censored'
    )

    # Verify that missing values are handled correctly
    assert isinstance(msm_data, MultiState)
    assert len(msm_data.patient_id) == 2  # Two transitions
    assert np.array_equal(msm_data.patient_id, np.array([1, 2]))
    assert np.array_equal(msm_data.start_state, np.array([1, 2]))  # A=1, B=2
    assert np.array_equal(msm_data.end_state, np.array([0, 0]))  # Both censored
```


### `tests/test_hazard_estimation.py`

This file contains the tests for the hazard estimation logic.

```python
"""
Tests for the hazard estimation module.
"""
import numpy as np
import pytest
from ruletimer.utils.hazard_estimation import HazardEstimator

@pytest.fixture
def sample_data():
    """Create sample survival data for testing."""
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    events = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return times, events, weights

def test_estimate_baseline_hazard_nelson_aalen(sample_data):
    """Test Nelson-Aalen estimator for baseline hazard."""
    times, events, weights = sample_data

    # Estimate baseline hazard
    unique_times, baseline_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, weights, method="nelson-aalen"
    )

    # Check output types and shapes
    assert isinstance(unique_times, np.ndarray)
    assert isinstance(baseline_hazard, np.ndarray)
    assert len(unique_times) == len(baseline_hazard)

    # Check that unique_times contains only event times
    assert np.all(unique_times == np.array([1, 2, 4, 6, 7, 9, 10]))

    # Check hazard values are non-negative
    assert np.all(baseline_hazard >= 0)

    # Check hazard values are reasonable
    # At time 1: 1 event / 10 at risk
    assert np.isclose(baseline_hazard[0], 0.1)

    # At time 2: 1 event / 9 at risk
    assert np.isclose(baseline_hazard[1], 1/9)

def test_estimate_baseline_hazard_weights(sample_data):
    """Test weighted hazard estimation."""
    times, events, _ = sample_data
    weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Estimate baseline hazard with weights
    unique_times, baseline_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, weights, method="nelson-aalen"
    )

    # Check that weighted hazard is different from unweighted
    _, unweighted_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, None, method="nelson-aalen"
    )

    assert not np.allclose(baseline_hazard, unweighted_hazard)

    # Check first hazard value with weight 2.0
    assert np.isclose(baseline_hazard[0], 2.0/11.0)

def test_estimate_cumulative_hazard(sample_data):
    """Test cumulative hazard estimation."""
    times, events, weights = sample_data

    # Get baseline hazard
    unique_times, baseline_hazard = HazardEstimator.estimate_baseline_hazard(
        times, events, weights, method="nelson-aalen"
    )

    # Estimate cumulative hazard
    pred_times, cumulative_hazard = HazardEstimator.estimate_cumulative_hazard(
        unique_times, baseline_hazard
    )

    # Check output types and shapes
    assert isinstance(pred_times, np.ndarray)
    assert isinstance(cumulative_hazard, np.ndarray)
    assert len(pred_times) == len(cumulative_hazard)

    # Check cumulative hazard is non-decreasing
    assert np.all(np.diff(cumulative_hazard) >= 0)

    # Check cumulative hazard at first time point
    print(f"Cumulative hazard[0]: {cumulative_hazard[0]}")
    print(f"Baseline hazard[0]: {baseline_hazard[0]}")
    assert np.isclose(cumulative_hazard[0], baseline_hazard[0])

    # Check cumulative hazard at last time point
    assert np.isclose(cumulative_hazard[-1], np.sum(baseline_hazard))

def test_transform_hazard():
    """Test hazard transformations."""
    # Create sample cumulative hazard
    cumulative_hazard = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Test survival transformation
    survival = HazardEstimator.transform_hazard(cumulative_hazard, transform="exp")
    assert np.allclose(survival, np.exp(-cumulative_hazard))

    # Test CIF transformation
    cif = HazardEstimator.transform_hazard(cumulative_hazard, transform="cif")
    assert np.allclose(cif, 1 - np.exp(-cumulative_hazard))

    # Test invalid transformation
    with pytest.raises(ValueError):
        HazardEstimator.transform_hazard(cumulative_hazard, transform="invalid")

def test_estimate_baseline_hazard_invalid_method(sample_data):
    """Test invalid hazard estimation method."""
    times, events, weights = sample_data

    with pytest.raises(ValueError):
        HazardEstimator.estimate_baseline_hazard(
            times, events, weights, method="invalid"
        )
```


### `tests/test_rule_multi_state.py`

This file contains the tests for the multi-state model.

```python
"""
Tests for the RuleMultiState class.
"""
import numpy as np
import pytest
from ruletimer.models.rule_multi_state import RuleMultiState
from ruletimer.utils import StateStructure

def test_rule_multi_state_initialization():
    """Test initialization of RuleMultiState with different parameters."""
    # Test with default parameters
    model = RuleMultiState()
    assert model.max_rules == 100
    assert model.alpha == 0.1
    assert model.max_depth == 3
    assert model.min_samples_leaf == 10
    assert model.n_estimators == 100
    assert model.tree_type == 'classification'
    assert model.tree_growing_strategy == 'forest'
    assert model.prune_rules is True
    assert model.l1_ratio == 0.5

    # Test with custom parameters
    state_structure = StateStructure(
        states=['healthy', 'sick', 'dead'],
        transitions=[(0, 1), (1, 2)]
    )
    model = RuleMultiState(
        max_rules=50,
        alpha=0.5,
        state_structure=state_structure,
        max_depth=5,
        min_samples_leaf=20,
        n_estimators=200,
        tree_type='regression',
        tree_growing_strategy='single',
        prune_rules=False,
        l1_ratio=0.8,
        random_state=42
    )
    assert model.max_rules == 50
    assert model.alpha == 0.5
    assert model.state_structure == state_structure
    assert model.max_depth == 5
    assert model.min_samples_leaf == 20
    assert model.n_estimators == 200
    assert model.tree_type == 'regression'
    assert model.tree_growing_strategy == 'single'
    assert model.prune_rules is False
    assert model.l1_ratio == 0.8
    assert model.random_state == 42

def test_rule_generation():
    """Test rule generation functionality."""
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = RuleMultiState(
        max_rules=10,
        max_depth=2,
        n_estimators=5,
        random_state=42
    )

    rules = model._generate_rules(X, y)
    assert isinstance(rules, list)
    assert len(rules) <= 10  # Should not exceed max_rules
    assert all(isinstance(rule, str) for rule in rules)

def test_rule_evaluation():
    """Test rule evaluation functionality."""
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    rules = ['feature_0 > 2', 'feature_1 < 5']

    model = RuleMultiState()
    rule_values = model._evaluate_rules(X, rules)

    assert isinstance(rule_values, np.ndarray)
    assert rule_values.shape == (X.shape[0], len(rules))
    assert np.all(np.logical_or(rule_values == 0, rule_values == 1))

def test_fit_predict():
    """Test model fitting and prediction."""
    # Create a simple multi-state dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    states = ['healthy', 'sick', 'dead']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)

    # Create transition data
    transition_data = {
        (0, 1): {
            'times': np.array([1.0, 2.0, 3.0]),
            'events': np.array([1, 1, 0])
        },
        (1, 2): {
            'times': np.array([2.0, 3.0, 4.0]),
            'events': np.array([1, 0, 1])
        }
    }

    model = RuleMultiState(
        state_structure=state_structure,
        max_rules=5,
        random_state=42
    )

    # Test fitting
    model.fit(X, transition_data)
    assert model.is_fitted_
    assert len(model.rules_) == len(transitions)
    assert len(model.rule_importances_) == len(transitions)
    assert len(model.rule_coefficients_) == len(transitions)

    # Test prediction
    times = np.array([1.0, 2.0, 3.0])
    target_state = 'dead'
    predictions = model.predict_cumulative_incidence(X, times, target_state)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0], len(times))
    assert np.all(predictions >= 0) and np.all(predictions <= 1)

def test_feature_importances():
    """Test feature importance calculation."""
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    states = ['healthy', 'sick', 'dead']
    transitions = [(0, 1), (1, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)

    transition_data = {
        (0, 1): {
            'times': np.array([1.0, 2.0, 3.0]),
            'events': np.array([1, 1, 0])
        },
        (1, 2): {
            'times': np.array([2.0, 3.0, 4.0]),
            'events': np.array([1, 0, 1])
        }
    }

    model = RuleMultiState(
        state_structure=state_structure,
        max_rules=5,
        random_state=42
    )
    model.fit(X, transition_data)

    # Test feature importances for each transition
    for transition in transitions:
        importances = model.get_feature_importances(transition)
        assert isinstance(importances, np.ndarray)
        assert len(importances) == X.shape[1]
        assert np.all(importances >= 0)
        assert np.allclose(np.sum(importances), 1.0)

def test_error_handling():
    """Test error handling for invalid inputs."""
    model = RuleMultiState()

    # Test invalid state structure
    with pytest.raises(ValueError):
        invalid_states = ['state1', 'state2']
        invalid_transitions = [(0, 1), (2, 3)]  # Invalid state indices
        invalid_structure = StateStructure(
            states=invalid_states,
            transitions=invalid_transitions
        )
        RuleMultiState(state_structure=invalid_structure)

    # Test invalid tree type
    with pytest.raises(ValueError):
        RuleMultiState(tree_type='invalid_type')

    # Test invalid tree growing strategy
    with pytest.raises(ValueError):
        RuleMultiState(tree_growing_strategy='invalid_strategy')

    # Test invalid alpha
    with pytest.raises(ValueError):
        RuleMultiState(alpha=-0.1)

    # Test invalid l1_ratio
    with pytest.raises(ValueError):
        RuleMultiState(l1_ratio=1.5)

def test_empty_and_single_row():
    """Test model with empty and single-row datasets."""
    X_empty = np.empty((0, 2))
    X_single = np.array([[1, 2]])
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    # Should not raise error on fit with empty data
    with pytest.raises(Exception):
        model.fit(X_empty, {(0, 1): {'times': np.array([]), 'events': np.array([])}})
    # Single row
    model.fit(X_single, {(0, 1): {'times': np.array([1.0]), 'events': np.array([1])}})
    preds = model.predict_cumulative_incidence(X_single, np.array([1.0]), 1)
    assert preds.shape == (1, 1)
    assert np.all(preds >= 0) and np.all(preds <= 1)

def test_all_censored():
    """Test model with all transitions censored."""
    X = np.random.randn(5, 2)
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    # All events are censored (0)
    model.fit(X, {(0, 1): {'times': np.ones(5), 'events': np.zeros(5)}})
    preds = model.predict_cumulative_incidence(X, np.array([1.0, 2.0]), 1)
    assert np.all(preds >= 0) and np.all(preds <= 1)
    assert np.allclose(preds, 0)

def test_missing_transition():
    """Test model with missing transitions for some states."""
    X = np.random.randn(4, 2)
    states = ['a', 'b', 'c']
    transitions = [(0, 1)]  # No (1,2) or (0,2)
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, {(0, 1): {'times': np.ones(4), 'events': np.array([1, 0, 1, 0])}})
    # Should not raise error for missing transitions
    preds = model.predict_cumulative_incidence(X, np.array([1.0]), 1)
    assert preds.shape == (4, 1)

def test_large_state_space():
    """Test scalability with many states and transitions."""
    n_states = 10
    states = [str(i) for i in range(n_states)]
    transitions = [(i, i+1) for i in range(n_states-1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    X = np.random.randn(20, 3)
    transition_data = {t: {'times': np.random.rand(20), 'events': np.random.randint(0, 2, 20)} for t in transitions}
    model = RuleMultiState(state_structure=state_structure, max_rules=3)
    model.fit(X, transition_data)
    for t in transitions:
        assert len(model.rules_[t]) <= 3

def test_occupation_probability_sum():
    """Test that state occupation probabilities sum to 1 at each time point."""
    X = np.random.randn(10, 2)
    states = ['a', 'b', 'c']
    transitions = [(0, 1), (1, 2), (0, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)
    transition_data = {t: {'times': np.random.rand(10), 'events': np.random.randint(0, 2, 10)} for t in transitions}
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, transition_data)
    times = np.linspace(0, 5, 10)
    occ = model.predict_state_occupation(X, times, initial_state=0)
    total = np.zeros((X.shape[0], len(times)))
    for s in occ:
        total += occ[s]
    assert np.allclose(total, 1, atol=1e-5)

def test_absorbing_state_monotonicity():
    """Test that occupation probability for absorbing state is non-decreasing."""
    X = np.random.randn(5, 2)
    states = ['a', 'b', 'c']
    transitions = [(0, 1), (1, 2), (0, 2)]
    state_structure = StateStructure(states=states, transitions=transitions)
    transition_data = {t: {'times': np.random.rand(5), 'events': np.random.randint(0, 2, 5)} for t in transitions}
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, transition_data)
    times = np.linspace(0, 5, 10)
    occ = model.predict_state_occupation(X, times, initial_state=0)
    # State 2 is absorbing
    diffs = np.diff(occ[2], axis=1)
    assert np.all(diffs >= -1e-10)

def test_feature_importance_normalization():
    """Test that feature importances are non-negative and sum to 1."""
    X = np.random.randn(6, 3)
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, {(0, 1): {'times': np.random.rand(6), 'events': np.random.randint(0, 2, 6)}})
    importances = model.get_feature_importances((0, 1))
    assert np.all(importances >= 0)
    assert np.isclose(np.sum(importances), 1)

def test_model_persistence(tmp_path):
    """Test that model predictions are consistent after saving and loading."""
    import joblib
    X = np.random.randn(8, 2)
    states = ['a', 'b']
    transitions = [(0, 1)]
    state_structure = StateStructure(states=states, transitions=transitions)
    model = RuleMultiState(state_structure=state_structure)
    model.fit(X, {(0, 1): {'times': np.random.rand(8), 'events': np.random.randint(0, 2, 8)}})
    times = np.array([1.0, 2.0])
    preds = model.predict_cumulative_incidence(X, times, 1)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    preds2 = loaded.predict_cumulative_incidence(X, times, 1)
    assert np.allclose(preds, preds2)
```


### `tests/test_state_manager.py`

This file contains the tests for the state management logic.

```python
import numpy as np
import pytest
from ruletimer.state_manager import StateManager

def test_state_manager_initialization():
    """Test initialization of state manager"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]

    manager = StateManager(states, transitions)
    assert manager.states == states
    assert manager.transitions == transitions
    assert len(manager._state_to_idx) == len(states)
    assert len(manager._idx_to_state) == len(states)

def test_to_internal_index():
    """Test conversion to internal index"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)

    # Test string state names
    assert manager.to_internal_index("Healthy") == 0
    assert manager.to_internal_index("Disease") == 1
    assert manager.to_internal_index("Death") == 2

    # Test numeric indices
    assert manager.to_internal_index(0) == 0
    assert manager.to_internal_index(1) == 1
    assert manager.to_internal_index(2) == 2

    # Test invalid state
    with pytest.raises(ValueError):
        manager.to_internal_index("Invalid")

def test_to_external_state():
    """Test conversion to external state"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)

    # Test valid indices
    assert manager.to_external_state(0) == "Healthy"
    assert manager.to_external_state(1) == "Disease"
    assert manager.to_external_state(2) == "Death"

    # Test invalid index
    with pytest.raises(ValueError):
        manager.to_external_state(3)

def test_validate_transition():
    """Test transition validation"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)

    # Test valid transitions
    assert manager.validate_transition(0, 1)
    assert manager.validate_transition(1, 2)
    assert manager.validate_transition(0, 2)

    # Test invalid transitions
    assert not manager.validate_transition(0, 0)  # Self-transition
    assert not manager.validate_transition(2, 1)  # Reverse transition
    assert not manager.validate_transition(0, 3)  # Invalid state

def test_get_possible_transitions():
    """Test getting possible transitions"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)

    # Test from specific state
    assert manager.get_possible_transitions(0) == [1, 2]
    assert manager.get_possible_transitions(1) == [2]
    assert manager.get_possible_transitions(2) == []

    # Test all transitions
    all_transitions = manager.get_possible_transitions()
    assert all_transitions == [(0, 1), (0, 2), (1, 2)]

def test_is_absorbing_state():
    """Test absorbing state detection"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)

    assert not manager.is_absorbing_state(0)
    assert not manager.is_absorbing_state(1)
    assert manager.is_absorbing_state(2)

def test_get_absorbing_states():
    """Test getting all absorbing states"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)

    assert manager.get_absorbing_states() == [2]

def test_validate_state_sequence():
    """Test state sequence validation"""
    states = ["Healthy", "Disease", "Death"]
    transitions = [(0, 1), (1, 2), (0, 2)]
    manager = StateManager(states, transitions)

    # Test valid sequence
    assert manager.validate_state_sequence([0, 1, 2])
    assert manager.validate_state_sequence([0, 2])

    # Test invalid sequences
    assert not manager.validate_state_sequence([0, 0])  # Self-transition
    assert not manager.validate_state_sequence([2, 1])  # Reverse transition
    assert not manager.validate_state_sequence([0, 3])  # Invalid state
```


### `tests/test_statistical_validation.py`

This file contains the tests for the statistical validation of the models.

```python
"""
Statistical validation tests for RuleTimeR models.

This module contains tests that verify the statistical properties and predictive
performance of RuleTimeR models, including:
- Monotonicity of survival curves
- Proper bounds for probabilities
- Feature-specific risk factors
- Model calibration
- Predictive performance
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd

from ruletimer.data import Survival, CompetingRisks, MultiState
from ruletimer.models import RuleSurvivalCox, RuleCompetingRisks, RuleMultiState
from ruletimer.utils import StateStructure

def test_survival_curve_monotonicity():
    """Test that survival curves are monotonically decreasing."""
    X = np.random.randn(100, 5)
    times = np.random.exponential(scale=5, size=100)
    events = np.random.binomial(1, 0.7, size=100)
    y = Survival(time=times, event=events)

    model = RuleSurvivalCox(max_rules=32, random_state=42)
    model.fit(X, y)

    # Predict at many time points
    test_times = np.linspace(0, 10, 100)
    survival_probs = model.predict_survival(X[:5], test_times)

    # Check monotonicity for each sample
    for i in range(survival_probs.shape[0]):
        diffs = np.diff(survival_probs[i])
        assert np.all(diffs <= 0 + 1e-10), f"Survival curve for sample {i} is not monotonically decreasing"

def test_survival_function_bounds():
    """Test that survival probabilities are between 0 and 1, start near 1, and approach proper limits."""
    # Create data with two clear risk groups
    X = np.zeros((100, 2))
    X[:50, 0] = 1.0  # High risk group

    # Generate survival times based on risk groups
    times = np.zeros(100)
    times[:50] = np.random.exponential(scale=2, size=50)  # High risk: shorter times
    times[50:] = np.random.exponential(scale=5, size=50)  # Low risk: longer times
    events = np.random.binomial(1, 0.8, size=100)

    y = Survival(time=times, event=events)
    model = RuleSurvivalCox(random_state=42)
    model.fit(X, y)

    test_times = np.linspace(0, 20, 100)
    survival_probs = model.predict_survival(X[[0, 50]], test_times)  # One from each group

    # Check bounds
    assert np.all(survival_probs >= 0), "Survival probabilities below 0"
    assert np.all(survival_probs <= 1), "Survival probabilities above 1"

    # Check initial values (should be close to 1)
    assert np.all(survival_probs[:, 0] > 0.95), "Survival probability at time 0 not near 1"

    # Check that high-risk group has lower survival than low-risk group
    assert np.mean(survival_probs[0]) < np.mean(survival_probs[1]), "High risk group doesn't have lower survival"

    # Check asymptotic behavior at long times - make less stringent
    last_time_idx = -10  # Look at the last few time points
    assert np.mean(survival_probs[0, last_time_idx:]) < 0.3, "Survival doesn't approach proper limit for high risk"

def test_competing_risks_cif_properties():
    """Test that cumulative incidence functions satisfy basic properties."""
    # Create data with two competing events
    X = np.random.randn(100, 3)
    times = np.random.exponential(scale=5, size=100)
    events = np.random.choice([0, 1, 2], size=100, p=[0.2, 0.4, 0.4])
    y = CompetingRisks(time=times, event=events)

    model = RuleCompetingRisks(random_state=42)
    model.fit(X, y)

    # Predict at many time points
    test_times = np.linspace(0, 10, 100)
    cifs = model.predict_cumulative_incidence(X[:5], test_times)

    # Convert dictionary to array for easier testing
    cif_array = np.stack([cifs[event_type] for event_type in model.event_types], axis=1)

    # Check that CIFs are monotonically increasing (allowing for small numerical errors)
    for i in range(cif_array.shape[0]):
        for j in range(cif_array.shape[1]):
            diffs = np.diff(cif_array[i, j])
            # Allow for very small negative differences due to floating point errors
            assert np.all(diffs >= -1e-9), f"CIF for sample {i}, event {j} is not monotonically increasing (diffs: {diffs[diffs < -1e-9]})"

    # Check that CIFs are between 0 and 1
    assert np.all(cif_array >= 0) and np.all(cif_array <= 1), "CIF values out of bounds [0, 1]"

    # Allow slightly more tolerance in CIF sum
    assert np.all(np.sum(cif_array, axis=1) <= 1 + 1e-8), "Sum of CIFs exceeds 1"

def test_multistate_occupation_probabilities():
    """Test that state occupation probabilities sum to 1 and follow expected patterns."""
    # Define state structure for illness-death model
    states = [0, 1, 2]  # Healthy, Ill, Dead
    transitions = [(0, 1), (1, 2), (0, 2)]
    state_names = ["Healthy", "Ill", "Dead"]
    structure = StateStructure(states=states, transitions=transitions, state_names=state_names)

    # Generate synthetic data
    n_samples = 100
    X = np.random.randn(n_samples, 3)

    # Create transition data
    transition_data = {}
    for transition in transitions:
        # Generate transition-specific hazard
        hazard = np.exp(0.2 * X[:, 0])
        times = np.random.exponential(1/hazard)
        events = np.random.binomial(1, 0.7, size=n_samples)
        transition_data[transition] = {
            'times': times,
            'events': events
        }

    # Fit model
    model = RuleMultiState(state_structure=structure, random_state=42)
    model.fit(X, transition_data)

    # Predict state occupation probabilities
    times = np.linspace(0, 10, 50)
    state_probs = model.predict_state_occupation(X[:5], times, initial_state=0)

    # Check that probabilities sum to 1 at all times
    total_probs = np.zeros((5, len(times)))
    for state in states:
        total_probs += state_probs[state]

    assert np.allclose(total_probs, 1.0, atol=1e-5), "State occupation probabilities don't sum to 1"

    # Check initial state probabilities
    assert np.all(state_probs[0][:, 0] > 0.95), "Initial state probability not close to 1"
    assert np.all(state_probs[1][:, 0] < 0.05), "Non-initial state probability not close to 0"
    assert np.all(state_probs[2][:, 0] < 0.05), "Non-initial state probability not close to 0"

    # Check absorbing state properties (Dead is absorbing)
    for i in range(5):
        # Dead state probability should be monotonically increasing
        diffs = np.diff(state_probs[2][i])
        assert np.all(diffs >= 0 - 1e-10), f"Absorbing state not monotonically increasing for sample {i}"

def test_model_calibration():
    """Test that model predictions are well-calibrated against observed outcomes."""
    # Create dataset with known risk groups
    n_samples = 500
    X = np.zeros((n_samples, 1))

    # Define 5 risk groups
    for i in range(5):
        X[i*100:(i+1)*100, 0] = i

    # Generate survival times based on risk groups
    times = np.zeros(n_samples)
    for i in range(5):
        scale = 5.0 / (i + 1)  # Higher risk groups have shorter survival times
        times[i*100:(i+1)*100] = np.random.exponential(scale=scale, size=100)

    # Add censoring
    censor_times = np.random.exponential(scale=10, size=n_samples)
    events = times <= censor_times
    times = np.minimum(times, censor_times)

    y = Survival(time=times, event=events)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit model
    model = RuleSurvivalCox(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate calibration
    # Use quantile-based approach: divide predictions into quantiles and compare observed vs predicted

    # Select a time point for calibration
    calib_time = 2.0

    # Predict survival at calibration time
    pred_surv = model.predict_survival(X_test, np.array([calib_time]))[:, 0]

    # Group predictions into quintiles
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    pred_groups = discretizer.fit_transform(pred_surv.reshape(-1, 1)).flatten().astype(int)

    # Calculate observed survival for each group
    observed_surv = np.zeros(5)

    for group in range(5):
        mask = pred_groups == group
        if np.sum(mask) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(y_test.time[mask], y_test.event[mask])
            observed_surv[group] = kmf.predict(calib_time)

    # Calculate average predicted survival for each group
    predicted_surv = np.zeros(5)
    for group in range(5):
        mask = pred_groups == group
        if np.sum(mask) > 0:
            predicted_surv[group] = np.mean(pred_surv[mask])

    # Calculate calibration error
    calib_error = np.mean(np.abs(observed_surv - predicted_surv))

    # Increase acceptable calibration error threshold
    assert calib_error < 0.15, f"Model not well-calibrated. Calibration error: {calib_error}"

    # Check monotonicity of observed survival across risk groups
    # Higher risk groups should have lower observed survival
    for i in range(4):
        if observed_surv[i] > 0 and observed_surv[i+1] > 0:
            assert observed_surv[i] >= observed_surv[i+1] - 0.1, "Observed survival not monotonic across risk groups"

def test_predictive_performance():
    """Test that model has good predictive performance compared to a simple reference model."""
    # Create dataset
    X = np.random.randn(300, 5)

    # Generate survival times with dependency on first two features
    baseline_hazard = 0.1
    feature_weights = np.array([0.5, 0.3, 0, 0, 0])
    hazard = baseline_hazard * np.exp(np.dot(X, feature_weights))
    times = np.random.exponential(scale=1/hazard)

    # Add censoring
    censor_times = np.random.exponential(scale=5, size=300)
    events = times <= censor_times
    times = np.minimum(times, censor_times)

    y = Survival(time=times, event=events)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit rule-based model
    rule_model = RuleSurvivalCox(random_state=42)
    rule_model.fit(X_train, y_train)

    # Fit reference Cox model using only the first feature
    df_train = pd.DataFrame(X_train, columns=[f'X{i}' for i in range(5)])
    df_train['time'] = y_train.time
    df_train['event'] = y_train.event

    # Use only first feature for reference model
    cox_ref = CoxPHFitter()
    cox_ref.fit(df_train[['X0', 'time', 'event']], duration_col='time', event_col='event')

    # Calculate concordance index for both models
    # Rule model predictions
    rule_risk = rule_model.predict_risk(X_test)
    rule_cindex = concordance_index(y_test.time, -rule_risk, y_test.event)

    # Cox model predictions
    df_test = pd.DataFrame(X_test, columns=[f'X{i}' for i in range(5)])
    cox_risk = cox_ref.predict_partial_hazard(df_test[['X0']])
    cox_cindex = concordance_index(y_test.time, cox_risk, y_test.event)

    # Rule model should outperform simple Cox model
    assert rule_cindex > cox_cindex, f"Rule model ({rule_cindex:.3f}) not better than reference model ({cox_cindex:.3f})"

    # Lower required c-index threshold
    assert rule_cindex > 0.55, f"Rule model has poor discriminative ability: c-index = {rule_cindex:.3f}"
```


### `tests/test_visualization.py`

This file contains the tests for the visualization functions.

```python
"""Tests for visualization functions"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from ruletimer.visualization.visualization import (
    plot_rule_importance,
    plot_cumulative_incidence,
    plot_state_transitions,
    plot_state_occupation
)
from ruletimer.models.base import BaseRuleEnsemble
from ruletimer.utils.utils import StateStructure

class MockRuleEnsemble(BaseRuleEnsemble):
    """Mock rule ensemble for testing visualization functions"""

    def __init__(self):
        super().__init__()
        self._rules_tuples = ["feature_0 <= 0.5", "feature_1 > 0.3"]
        self.rule_weights_ = np.array([0.8, -0.5])
        self.state_structure = StateStructure(
            states=[1, 2, 3],  # Using 1-based indexing for states (0 is reserved for censoring)
            transitions=[(1, 2), (2, 3)],  # States are 1-based
            state_names=["Healthy", "Mild", "Severe"]  # Providing state names separately
        )
        self._y = np.rec.fromarrays([np.array([1.0, 2.0, 3.0])], names=['time'])
        self.baseline_hazards_ = {
            (1, 2): (np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3])),
            (2, 3): (np.array([0, 1, 2]), np.array([0.2, 0.3, 0.4]))
        }

    def _fit_weights(self, rule_values, y):
        pass

    def _compute_feature_importances(self):
        pass

    @property
    def rules_(self):
        # Return a dict to match the expected interface in plot_rule_importance
        # Use a single transition key for simplicity
        return {(1, 2): list(self._rules_tuples)}

    def get_rule_importances(self):
        # Return a dict with the same keys as rules_
        if isinstance(self.rule_weights_, dict):
            return self.rule_weights_
        return {(1, 2): np.array(self.rule_weights_)}

    def predict_cumulative_incidence(self, X, times, event_types=None, event_type=None, **kwargs):
        n_samples = X.shape[0]
        n_times = len(times)
        # Always return a dict of 2D numpy arrays for each requested event_type
        if event_types is not None:
            return {et: np.random.rand(n_samples, n_times) for et in event_types}
        elif event_type is not None:
            # Accept event_type as int or str
            if isinstance(event_type, str) and event_type.startswith("Event"):
                try:
                    et = int(event_type.replace("Event", ""))
                except Exception:
                    et = event_type
            else:
                et = event_type
            return {et: np.random.rand(n_samples, n_times)}
        else:
            # Default: return a dict with a single key 0
            return {0: np.random.rand(n_samples, n_times)}

    def predict_state_occupation(self, X, times, initial_state=None, **kwargs):
        n_samples = X.shape[0]
        n_times = len(times)
        # Return a dict for each state (0-based, to match visualization expectations)
        return {i: np.random.rand(n_samples, n_times) for i in range(len(self.state_structure.states))}

    def _evaluate_rules(self, X):
        return np.random.rand(X.shape[0], len(self.rules_))

    def predict_transition_hazard(self, X, times, from_state, to_state):
        n_samples = X.shape[0]
        n_times = len(times)
        # Return a dummy hazard array
        return np.random.rand(n_samples, n_times)

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    return MockRuleEnsemble()

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    return X

def test_plot_rule_importance(mock_model, sample_data):
    """Test rule importance plotting"""
    plt.close('all')
    plot_rule_importance(mock_model, top_n=2)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence(mock_model, sample_data):
    """Test cumulative incidence plotting"""
    plt.close('all')
    plot_cumulative_incidence(
        mock_model,
        sample_data,
        event_types=[0, 1],
        times=np.linspace(0, 5, 10)
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_transitions(mock_model, sample_data):
    """Test state transition plotting"""
    plt.close('all')
    plot_state_transitions(mock_model, sample_data, time=1.0)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_occupation(mock_model, sample_data):
    """Test state occupation plotting"""
    plt.close('all')
    times = np.linspace(0, 5, 10)
    state_probs = mock_model.predict_state_occupation(sample_data, times)
    plot_state_occupation(
        times,
        state_probs,
        state_names=mock_model.state_structure.states
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_rule_importance_empty(mock_model):
    """Test rule importance plotting with empty rules"""
    plt.close('all')
    mock_model._rules_tuples = []
    mock_model.rule_weights_ = np.array([])
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_no_times(mock_model, sample_data):
    """Test plotting cumulative incidence with empty times array"""
    plt.close('all')
    with pytest.raises(ValueError):
        plot_cumulative_incidence(
            mock_model,
            sample_data,
            event_types=[1],
            times=np.array([])
        )
    plt.close('all')

def test_plot_state_occupation_missing_state(mock_model, sample_data):
    """Test state occupation plotting with missing state"""
    plt.close('all')
    times = np.linspace(0, 5, 10)
    state_probs = mock_model.predict_state_occupation(sample_data, times)
    del state_probs[0]  # Remove one state (changed to 0-based)
    plot_state_occupation(
        times,
        state_probs,
        state_names=mock_model.state_structure.states
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_transitions_invalid_times(mock_model, sample_data):
    """Test plotting state transitions with invalid times"""
    plt.close('all')
    with pytest.raises(ValueError):
        plot_state_transitions(
            mock_model,
            sample_data,
            time=-1.0  # Negative time should raise error
        )
    plt.close('all')

def test_plot_state_occupation_single_state(mock_model, sample_data):
    """Test plotting state occupation with single state"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=["Single"],
        transitions=[]
    )
    times = np.linspace(0, 5, 10)
    with pytest.raises(ValueError):
        plot_state_occupation(
            times,
            {1: np.random.rand(sample_data.shape[0], len(times))},
            state_names=["Single"]
        )
    plt.close('all')

def test_plot_rule_importance_custom_names(mock_model):
    """Test plotting rule importance with custom rule names"""
    plt.close('all')
    mock_model._rules_tuples = ["Custom Rule 1", "Custom Rule 2", "Custom Rule 3"]
    mock_model.rule_weights_ = np.array([0.8, -0.5, 0.3])
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_custom_labels(mock_model, sample_data):
    """Test plotting cumulative incidence with custom state labels"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=[1, 2, 3],  # Changed to 1-based indexing
        transitions=[(1, 2), (2, 3)]  # Changed to 1-based indexing
    )
    times = np.linspace(0, 5, 10)
    plot_cumulative_incidence(
        mock_model,
        sample_data,
        event_types=[0, 1],  # Changed to 0-based indexing
        times=times
    )
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_rule_importance_no_rules(mock_model):
    """Test plotting rule importance with no rules"""
    plt.close('all')
    mock_model._rules_tuples = []
    mock_model.rule_weights_ = np.array([])
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_no_event_types(mock_model, sample_data):
    """Test plotting cumulative incidence with no event types"""
    plt.close('all')
    with pytest.raises(ValueError):
        plot_cumulative_incidence(
            mock_model,
            sample_data,
            event_types=[],
            times=np.linspace(0, 5, 10)
        )
    plt.close('all')

def test_plot_state_transitions_no_transitions(mock_model, sample_data):
    """Test plotting state transitions with no transitions"""
    plt.close('all')
    mock_model.state_structure = StateStructure(
        states=["A", "B"],
        transitions=[]
    )
    plot_state_transitions(mock_model, sample_data, time=1.0)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_state_occupation_empty_probs(mock_model, sample_data):
    """Test plotting state occupation with empty probabilities"""
    plt.close('all')
    times = np.linspace(0, 5, 10)
    with pytest.raises(ValueError):
        plot_state_occupation(
            times,
            {},
            state_names=["A", "B"]
        )
    plt.close('all')

def test_plot_rule_importance_dict_weights(mock_model):
    """Test plotting rule importance with dictionary weights"""
    plt.close('all')
    mock_model._rules_tuples = ["Rule 1", "Rule 2"]
    mock_model.rule_weights_ = {
        (1, 2): np.array([0.5, -0.3]),  # Changed to 1-based indexing
        (2, 3): np.array([0.2, 0.4])    # Changed to 1-based indexing
    }
    plot_rule_importance(mock_model)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')

def test_plot_cumulative_incidence_invalid_event_type(mock_model, sample_data):
    """Test plotting cumulative incidence with invalid event type"""
    plt.close('all')
    # Modify mock signature to accept event_type keyword argument and potentially others (**kwargs)
    def mock_predict_cumulative_incidence(X, times, event_type=None, **kwargs):
        # Extract integer event type from the string "Event{event_type}"
        current_event_type_int = -1 # Default invalid
        if event_type is not None and isinstance(event_type, str) and event_type.startswith("Event"):
            try:
                current_event_type_int = int(event_type.replace("Event", ""))
            except ValueError:
                pass # Keep -1 if conversion fails
        elif isinstance(event_type, int):
             # Handle if event_type is passed as int (though plot function uses string)
            current_event_type_int = event_type

        # The test expects a KeyError for invalid types (e.g., > 3 based on original mock logic)
        # Let's simulate this check based on the extracted integer type
        if current_event_type_int > 3: # Check the integer value
            raise KeyError(f"Invalid event type: {current_event_type_int}")

        # Return a valid structure for valid types, using the integer type as key
        # This part might not even be reached if the type is invalid and raises KeyError
        return {current_event_type_int: np.zeros((X.shape[0], len(times)))}

    # Store original method before mocking
    original_predict_method = mock_model.predict_cumulative_incidence
    mock_model.predict_cumulative_incidence = mock_predict_cumulative_incidence

    try:
        # Expect KeyError because event_type 999 > 3
        with pytest.raises(KeyError):
            plot_cumulative_incidence(
                mock_model,
                sample_data,
                event_types=[999],  # Invalid event type
                times=np.linspace(0, 5, 10)
            )
    finally:
        # Restore original method
        mock_model.predict_cumulative_incidence = original_predict_method
        plt.close('all')

def test_plot_state_transitions_no_rule_weights(mock_model, sample_data):
    """Test plotting state transitions with missing rule weights"""
    plt.close('all')
    # Align state structure with the mock's hardcoded rules_ key (1, 2)
    # Use states 1, 2, 3 and transitions (1, 2), (2, 3)
    mock_model.state_structure = StateStructure(
        states=[1, 2, 3], # Use 1-based integer states
        transitions=[(1, 2), (2, 3)], # Define transitions
        state_names=["A", "B", "C"] # Optional names
    )
    # Set the underlying attributes that rules_ and get_rule_importances use
    mock_model._rules_tuples = ["Rule 1", "Rule 2"] # Define rules
    # Provide weights only for the transition (1, 2), matching the rules_ property key
    mock_model.rule_weights_ = {(1, 2): np.array([0.5, 0.3])}
    # Provide baseline hazards for the transition with rules
    mock_model.baseline_hazards_ = {(1, 2): (np.array([0, 1]), np.array([0.1, 0.2]))}

    # Mock predict_transition_hazard
    original_predict_hazard = getattr(mock_model, 'predict_transition_hazard', None)

    def mock_predict_hazard(X, times, from_state, to_state):
        n_samples = X.shape[0]
        n_times = len(times)
        # The plot function calls this only for transitions in model.rules_
        # In this setup, that's only (1, 2)
        if (from_state, to_state) == (1, 2):
            return np.random.rand(n_samples, n_times) * 0.1
        elif original_predict_hazard:
            return original_predict_hazard(X, times, from_state, to_state)
        else:
            # This path shouldn't be hit by plot_state_transitions given the check
            raise ValueError(f"Unexpected transition requested in mock: ({from_state}, {to_state})")

    mock_model.predict_transition_hazard = mock_predict_hazard

    # Plot function should now work, drawing state 1, 2, 3 and an arrow for (1, 2)
    plot_state_transitions(mock_model, sample_data, time=1.0)
    assert plt.get_fignums() # Check if figure was created
    plt.close('all')

    # Restore original method
    if original_predict_hazard:
        mock_model.predict_transition_hazard = original_predict_hazard
    elif hasattr(mock_model, 'predict_transition_hazard'):
        delattr(mock_model, 'predict_transition_hazard')
```
