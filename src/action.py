
import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Dict, Any, Optional, Tuple, List
import re


class ActionOptimizer:

    def __init__(self, parser_config: Dict[str, Any]):
 
        self.config = parser_config
        self.action_info = None
        self.equations = []
        self.action_constraints = []
        self._extract_action_info()

    def _extract_action_info(self):
        self.action_constraints = self.config.get('#Action_Constraints', [])

        actions = set()
        equations = []

        if 'ode_equation' in self.config:
            equations = self.config['ode_equation']
        elif '#ODE' in self.config:
            equations = self.config['#ODE']

        for eq in equations:
            action_matches = re.findall(r'action_\d+', str(eq))
            actions.update(action_matches)

        for constraint in self.action_constraints:
            if 'action' in constraint:
                actions.add(constraint['action'])

        self.action_info = {
            'actions': sorted(list(actions)),
            'num_actions': len(actions),
            'constraints': self.action_constraints
        }
        self.equations = equations

    def solve(self,
              data_dict: Dict[str, np.ndarray],
              initial_guess: Optional[np.ndarray] = None,
              method: str = 'SLSQP',
              verbose: bool = True,
              max_iter: int = 1000,
              patience: int = 10,
              min_improvement: float = 1e-6) -> Tuple[Dict[str, np.ndarray], np.ndarray]:


        if not self.equations:
            raise ValueError("No equations found in config. Did you parse ODE equations?")

        if not self.action_info['actions']:
            raise ValueError("No action variables found in equations")

        sample_tensor = next(iter(data_dict.values()))
        if len(sample_tensor.shape) != 3:
            raise ValueError(f"Expected 3D data (batch, rows, columns), got shape {sample_tensor.shape}")

        batch_size, rows, columns = sample_tensor.shape
        elements_per_batch = rows * columns

        if verbose:
            print("=" * 70)
            print("ACTION OPTIMIZER (Batch-wise Independent)")
            print("=" * 70)
            print(f"Data shape: (batch={batch_size}, rows={rows}, columns={columns})")
            print(f"Elements per batch: {elements_per_batch}")
            print(f"Actions to optimize: {self.action_info['actions']}")
            print(f"Number of actions: {self.action_info['num_actions']}")
            print(f"Optimization variables per batch: {self.action_info['num_actions'] * elements_per_batch}")
            print(f"Number of equations: {len(self.equations)}")
            print(f"Number of action constraints: {len(self.action_constraints)}")

        all_batch_actions = {action: np.zeros((batch_size, rows, columns)) 
                            for action in self.action_info['actions']}
        satisfaction_scores = np.zeros(batch_size)

        bounds = self._setup_bounds(elements_per_batch, verbose=verbose and batch_size <= 5)

        if verbose:
            print(f"\nOptimizing {batch_size} batches independently...")
            print("-" * 70)

        for b in range(batch_size):
            batch_data = {name: arr[b:b+1, :, :] for name, arr in data_dict.items()}

            if initial_guess is None:
                batch_initial = np.zeros(self.action_info['num_actions'] * elements_per_batch)
            else:
                batch_initial = initial_guess.copy()

            best_loss = np.inf
            no_improve_count = 0
            iteration_count = [0]  
            losses = []

            def callback(xk):
                iteration_count[0] += 1
                current_loss = self._compute_residual_single_batch(
                    xk, batch_data, (1, rows, columns), verbose=False
                )
                losses.append(current_loss)

                nonlocal best_loss, no_improve_count

                if current_loss < best_loss - min_improvement:
                    best_loss = current_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    if verbose and batch_size <= 10:
                        print(f"  Batch {b}: Early stop at iteration {iteration_count[0]}")
                    return True 

                return False

            def objective(actions_flat):
                return self._compute_residual_single_batch(
                    actions_flat, batch_data, (1, rows, columns), verbose=False
                )

            try:
                result = minimize(
                    objective,
                    batch_initial,
                    method=method,
                    bounds=bounds,
                    callback=callback,
                    options={'disp': False, 'maxiter': max_iter}
                )

                batch_actions = self._unflatten_actions(
                    result.x, (1, rows, columns)
                )

                for action_name, action_arr in batch_actions.items():
                    all_batch_actions[action_name][b, :, :] = action_arr[0, :, :]

                max_possible_residual = elements_per_batch * len(self.equations)
                score = min(1.0, result.fun / max_possible_residual)
                satisfaction_scores[b] = score

                if verbose and (batch_size <= 10 or b % max(1, batch_size // 10) == 0):
                    print(f"  Batch {b:3d}/{batch_size}: "
                          f"iterations={iteration_count[0]:3d}, "
                          f"residual={result.fun:.6e}, "
                          f"score={score:.4f}, "
                          f"success={result.success}")

            except Exception as e:
                if verbose:
                    print(f"  Batch {b}: Optimization failed - {e}")
                satisfaction_scores[b] = 1.0  # Worst score

        if verbose:
            print("-" * 70)
            print(f"\nOptimization completed for all batches!")
            print(f"\nSatisfaction Score Statistics (0=perfect, 1=worst):")
            print(f"  Mean:   {satisfaction_scores.mean():.6f}")
            print(f"  Median: {np.median(satisfaction_scores):.6f}")
            print(f"  Std:    {satisfaction_scores.std():.6f}")
            print(f"  Min:    {satisfaction_scores.min():.6f}")
            print(f"  Max:    {satisfaction_scores.max():.6f}")
            print(f"  Perfect (score=0): {np.sum(satisfaction_scores < 1e-6)} batches")
            print(f"  Good (score<0.1):  {np.sum(satisfaction_scores < 0.1)} batches")

            print(f"\nOptimized action statistics (across all batches):")
            for name, arr in all_batch_actions.items():
                print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                      f"min={arr.min():.4f}, max={arr.max():.4f}")

        return all_batch_actions, satisfaction_scores

    def _compute_residual_single_batch(self,
                                      actions_flat: np.ndarray,
                                      data_dict: Dict[str, np.ndarray],
                                      tensor_shape: Tuple,
                                      verbose: bool = False) -> float:

        action_arrays = self._unflatten_actions(actions_flat, tensor_shape)
        namespace = {}
        namespace.update(data_dict)
        namespace.update(action_arrays)
        namespace['np'] = np

        total_residual = 0.0

        for i, eq_str in enumerate(self.equations):
            try:
                if '=' in eq_str:
                    lhs, rhs = eq_str.split('=', 1)
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    lhs_val = eval(lhs, {"__builtins__": {}}, namespace)
                    rhs_val = eval(rhs, {"__builtins__": {}}, namespace)
                    residual_array = lhs_val - rhs_val
                else:
                    residual_array = eval(eq_str, {"__builtins__": {}}, namespace)

                residual = np.sum(residual_array ** 2)
                total_residual += residual

            except Exception as e:
                if verbose:
                    print(f"Error evaluating equation {i}: {eq_str}")
                    print(f"Error: {e}")
                return 1e10  

        return total_residual

    def _unflatten_actions(self,
                          actions_flat: np.ndarray,
                          tensor_shape: Tuple) -> Dict[str, np.ndarray]:

        total_elements = np.prod(tensor_shape)
        action_arrays = {}

        for i, action_name in enumerate(self.action_info['actions']):
            start_idx = i * total_elements
            end_idx = (i + 1) * total_elements

            action_flat = actions_flat[start_idx:end_idx]
            action_reshaped = action_flat.reshape(tensor_shape)
            action_arrays[action_name] = action_reshaped

        return action_arrays

    def _setup_bounds(self, total_elements: int, verbose: bool = False) -> Bounds:

        lower_bounds = []
        upper_bounds = []

        for action_name in self.action_info['actions']:
            action_lower = -np.inf
            action_upper = np.inf

            for constraint in self.action_constraints:
                if constraint['action'] == action_name:
                    operator = constraint['operator']
                    bound_val = constraint['bound']

                    try:
                        bound_numeric = float(bound_val)
                    except:
                        try:
                            bound_numeric = eval(str(bound_val))
                        except:
                            bound_numeric = 0.0

                    # Apply constraint
                    if operator in ['>', '>=']:
                        action_lower = max(action_lower, bound_numeric)
                        if operator == '>':
                            action_lower += 1e-10 
                    elif operator in ['<', '<=']:
                        action_upper = min(action_upper, bound_numeric)
                        if operator == '<':
                            action_upper -= 1e-10 
                    elif operator == '==':
                        action_lower = bound_numeric
                        action_upper = bound_numeric

            lower_bounds.extend([action_lower] * total_elements)
            upper_bounds.extend([action_upper] * total_elements)

            if verbose and (action_lower > -np.inf or action_upper < np.inf):
                print(f"  {action_name}: [{action_lower}, {action_upper}]")

        return Bounds(lower_bounds, upper_bounds)

    def evaluate_equations(self,
                          data_dict: Dict[str, np.ndarray],
                          action_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        namespace = {}
        namespace.update(data_dict)
        namespace.update(action_dict)
        namespace['np'] = np

        results = {}

        for i, eq_str in enumerate(self.equations):
            try:
                if '=' in eq_str:
                    lhs, rhs = eq_str.split('=', 1)
                    lhs_val = eval(lhs.strip(), {"__builtins__": {}}, namespace)
                    rhs_val = eval(rhs.strip(), {"__builtins__": {}}, namespace)
                    residual = lhs_val - rhs_val
                else:
                    residual = eval(eq_str, {"__builtins__": {}}, namespace)

                results[f"equation_{i}"] = residual
            except Exception as e:
                print(f"Error evaluating equation {i}: {e}")
                results[f"equation_{i}"] = None

        return results

    def check_constraints(self,
                         action_dict: Dict[str, np.ndarray]) -> Dict[str, bool]:

        results = {}

        for action_name in self.action_info['actions']:
            satisfied = True
            action_values = action_dict.get(action_name)

            if action_values is None:
                results[action_name] = False
                continue

            for constraint in self.action_constraints:
                if constraint['action'] == action_name:
                    operator = constraint['operator']
                    bound = float(constraint['bound'])

                    if operator == '>':
                        satisfied = satisfied and np.all(action_values > bound)
                    elif operator == '>=':
                        satisfied = satisfied and np.all(action_values >= bound)
                    elif operator == '<':
                        satisfied = satisfied and np.all(action_values < bound)
                    elif operator == '<=':
                        satisfied = satisfied and np.all(action_values <= bound)
                    elif operator == '==':
                        satisfied = satisfied and np.allclose(action_values, bound)

            results[action_name] = satisfied

        return results

    def compute_batch_scores(self,
                            data_dict: Dict[str, np.ndarray],
                            action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        batch_size = next(iter(data_dict.values())).shape[0]
        rows = next(iter(data_dict.values())).shape[1]
        columns = next(iter(data_dict.values())).shape[2]
        elements_per_batch = rows * columns

        scores = np.zeros(batch_size)
        max_possible_residual = elements_per_batch * len(self.equations)

        for b in range(batch_size):
            batch_data = {name: arr[b:b+1, :, :] for name, arr in data_dict.items()}
            batch_actions = {name: arr[b:b+1, :, :] for name, arr in action_dict.items()}
            residuals = self.evaluate_equations(batch_data, batch_actions)
            total_residual = 0.0
            for eq_name, residual_arr in residuals.items():
                if residual_arr is not None:
                    total_residual += np.sum(residual_arr ** 2)

            scores[b] = min(1.0, total_residual / max_possible_residual)

        return scores