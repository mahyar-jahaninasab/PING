import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict,Any, Optional, Tuple
from dataclasses import dataclass
import re 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OptimizationConfig:
    state_error_weight: float = 1.0
    time_reward_weight: float = 1.0
    equation_weight: float = 1.0
    derivatives_weight: float = 1.0
    reward_weight: float = 1.0
    time_mode: str = 'min' 
    epsilon: float = 1e-10  


class SolutionImprovement:

    def __init__(self, 
                 model: nn.Module,
                 prior: torch.Tensor,
                 time: torch.Tensor,
                 component: Dict[str, bool],
                 problem_config: Optional[Dict[str, Any]] = None):
        self.time_inputs = time
        self.model = model
        self.problem_config = problem_config

    def calculate_dervative(self):
        all_dervatives = []
        t_input = self.time_inputs .clone().detach().requires_grad_(True)
        output_for_der = self.model(t_input)
        saved_derivatives_per_state = []
        for state_idx, state_col in enumerate(self.problem_config['ODE']['states']):
            highest_order = self.problem_config['ODE']['highest'][state_idx]
            orders_to_save = self.problem_config['ODE']['saved'][state_idx]
            current_output = output_for_der[:, state_col]
            derivs_by_order = {}
            current_grad = None
            for order in range(1, highest_order + 1):
                need_for_higher_order = order < highest_order
                current_grad = torch.autograd.grad(
                    outputs=current_output.sum(),
                    inputs=t_input,
                    create_graph=need_for_higher_order,  
                    retain_graph=True  
                )[0]
                if order in orders_to_save:
                    derivs_by_order[order] = current_grad
                
                if need_for_higher_order:
                    current_output = current_grad
                else:
                    del current_grad
                    break
            ordered_derivs_for_state = [derivs_by_order[order] for order in orders_to_save]
            saved_derivatives_per_state.append(ordered_derivs_for_state)
            del current_output, derivs_by_order
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if saved_derivatives_per_state:
                all_dervatives.append(saved_derivatives_per_state)
        return all_dervatives
    
    def _dervative_constrains(self, all_dervatives):
        loss_value = 0
        for condition in self.problem_config["check_gradients"]:
            which_state = int(re.search(r'(?<=state_)\d+$', condition['state']).group())
            which_order = int(condition['derivative_order']) -1
            is_range = condition['type']
            target_value = float(condition['value'])
            if is_range == "standard":
                which_element = re.search(r't = (-?\d+)$',  condition['condition'])
                time_step = int(which_element.group(1))
                current_value = all_dervatives[which_state][which_order][time_step,:]
                distance = torch.sqrt((current_value**2 - target_value**2).abs())
                loss_value += distance 
            else:
                current_value = all_dervatives[which_state][which_order]
                penalty = 1 / (1 + torch.exp(-10*((current_value - target_value).abs())))
                distance = penalty.mean()
                loss_value += distance

        return loss_value 
    

    def _ode_check(self,action_list, dervatives):
        pass

    

class SolutionSelector:

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()  
        self._validate_config()
    
    def _validate_config(self) -> None:
        valid_modes = {'min', 'max', 'fix'}
        if self.config.time_mode not in valid_modes:
            raise ValueError(f'time_mode must be one of {valid_modes}, got: {self.config.time_mode}')
        
        if self.config.epsilon <= 0:
            raise ValueError(f'epsilon must be positive, got: {self.config.epsilon}')
        
        weights = [
            self.config.state_error_weight,
            self.config.time_reward_weight,
            self.config.equation_weight,
            self.config.derivatives_weight,
            self.config.reward_weight
        ]
        if any(w < 0 for w in weights):
            raise ValueError(f"All weights must be non-negative")
        
    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:

        if len(arr) == 0:
            raise ValueError("Cannot normalize an empty array")
        
        arr = np.asarray(arr, dtype=float)
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        
        if arr_max - arr_min < self.config.epsilon:
            return np.ones_like(arr) * 0.5 
        
        normalized_arr = (arr - arr_min) / (arr_max - arr_min + self.config.epsilon)
        return np.clip(normalized_arr, 0, 1)
    
    def _flip_metric(self, normalized_arr: np.ndarray) -> np.ndarray:
        return 1 - normalized_arr
    
    def _compute_normalized_scores(self, variables: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        normalized_scores = {}
        
        for var_name in ['state_error', 'equation', 'derivatives']:  
            if var_name in variables: 
                normalized_scores[var_name] = self._flip_metric(
                    self._normalize_array(variables[var_name])
                )
        
        if 'reward' in variables:
            normalized_scores['reward'] = self._normalize_array(variables['reward'])
        if self.config.time_mode != 'fix' and 'time_reward' in variables:
            normalized = self._normalize_array(variables['time_reward'])
            if self.config.time_mode == 'min':
                normalized = self._flip_metric(normalized)
            normalized_scores['time_reward'] = normalized
            
        return normalized_scores
    
    def _compute_weighted_scores(
        self,
        normalized_scores: Dict[str, np.ndarray],
        n_candidates: int  
    ) -> np.ndarray:

        weighted_scores = np.zeros(n_candidates)
        total_weight = 0.0
        
        if 'state_error' in normalized_scores:
            weight = self.config.state_error_weight
            weighted_scores += weight * normalized_scores['state_error']
            total_weight += weight
        
        if 'equation' in normalized_scores:
            weight = self.config.equation_weight
            weighted_scores += weight * normalized_scores['equation']
            total_weight += weight
        
        if 'derivatives' in normalized_scores:
            weight = self.config.derivatives_weight
            weighted_scores += weight * normalized_scores['derivatives']
            total_weight += weight
        
        if 'reward' in normalized_scores:
            weight = self.config.reward_weight
            weighted_scores += weight * normalized_scores['reward']
            total_weight += weight
        
        if 'time_reward' in normalized_scores:
            weight = self.config.time_reward_weight
            weighted_scores += weight * normalized_scores['time_reward']
            total_weight += weight
        
        if total_weight > 0:
            weighted_scores /= total_weight
        else:
            raise ValueError("Total weight is zero. At least one weight must be positive.")
        return weighted_scores
    
    def _extract_best_values(
        self,
        best_index: int,
        provided_vars: Dict[str, np.ndarray],
        all_variables: Dict[str, Optional[np.ndarray]]
    ) -> Dict[str, float]:

        best_values = {}
        for var_name in all_variables:
            if var_name in provided_vars:
                best_values[var_name] = float(provided_vars[var_name][best_index])
        return best_values
    
    def select(
        self,
        state_error: Optional[np.ndarray] = None,
        time_reward: Optional[np.ndarray] = None,
        equation: Optional[np.ndarray] = None,
        derivatives: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None
    ) -> Tuple[int, Dict[str, float]]:
        variables = {
            'state_error': state_error,
            'time_reward': time_reward,
            'equation': equation,
            'derivatives': derivatives,
            'reward': reward
        }
        provided_vars = {k: v for k, v in variables.items() if v is not None}
        if not provided_vars:
            raise ValueError("At least one variable must be provided")
        for name, arr in provided_vars.items():
            provided_vars[name] = np.asarray(arr, dtype=float)
            if provided_vars[name].ndim != 1:
                raise ValueError(f"{name} must be a 1D array, got shape: {provided_vars[name].shape}")
        arrays = list(provided_vars.values())
        n_candidates = len(arrays[0])
        if n_candidates == 0:
            raise ValueError("Arrays must have at least one element")
        
        for name, arr in provided_vars.items():
            if len(arr) != n_candidates:
                raise ValueError(
                    f"All input arrays must have the same length. "
                    f"Expected {n_candidates}, but {name} has length {len(arr)}"
                )
        normalized_scores = self._compute_normalized_scores(provided_vars)
        scores = self._compute_weighted_scores(normalized_scores, n_candidates)
        best_index = int(np.argmax(scores))
        best_values = self._extract_best_values(best_index, provided_vars, variables)
        return best_index, best_values
    
    def select_with_details(
        self,
        state_error: Optional[np.ndarray] = None,
        time_reward: Optional[np.ndarray] = None,
        equation: Optional[np.ndarray] = None,
        derivatives: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None
    ) -> Dict:

        variables = {
            'state_error': state_error,
            'time_reward': time_reward,
            'equation': equation,
            'derivatives': derivatives,
            'reward': reward
        }
        
        provided_vars = {k: v for k, v in variables.items() if v is not None}
        if not provided_vars:
            raise ValueError("At least one variable must be provided")
        for name, arr in provided_vars.items():
            provided_vars[name] = np.asarray(arr, dtype=float)
            if provided_vars[name].ndim != 1:
                raise ValueError(f"{name} must be a 1D array, got shape: {provided_vars[name].shape}")
        arrays = list(provided_vars.values())
        n_candidates = len(arrays[0])
        if n_candidates == 0:
            raise ValueError("Arrays must have at least one element")
        for name, arr in provided_vars.items():
            if len(arr) != n_candidates:
                raise ValueError(
                    f"All input arrays must have the same length. "
                    f"Expected {n_candidates}, but {name} has length {len(arr)}"
                )
        normalized_scores = self._compute_normalized_scores(provided_vars)
        scores = self._compute_weighted_scores(normalized_scores, n_candidates)
        best_index = int(np.argmax(scores))
        best_values = self._extract_best_values(best_index, provided_vars, variables)
        ranking = np.argsort(-scores)
        return {
            'best_index': best_index,
            'best_values': best_values,
            'scores': scores,
            'normalized_scores': normalized_scores,
            'ranking': ranking
        }
    


def reorder_derivatives_by_state(all_dervatives):
    if not all_dervatives or not all_dervatives[0]:
        return {}
    num_states = len(all_dervatives[0])
    reordered_derivatives = {}
    for state_idx in range(num_states):
        num_orders = len(all_dervatives[0][state_idx])
        orders_for_state_concatenated = []
        for order_idx in range(num_orders):
            tensors_for_current_order = [
                all_dervatives[seg_idx][state_idx][order_idx]
                for seg_idx in range(len(all_dervatives))
            ]
            concatenated_tensor = torch.cat(tensors_for_current_order, dim=0)
            if concatenated_tensor.shape[-1] == 1:
                concatenated_tensor = concatenated_tensor.squeeze(-1)
            orders_for_state_concatenated.append(concatenated_tensor)
        state_matrix = torch.stack(orders_for_state_concatenated, dim=1)
        reordered_derivatives[f"state_{state_idx}"] = state_matrix.permute(0, 2, 1).to(DEVICE)
    return reordered_derivatives
